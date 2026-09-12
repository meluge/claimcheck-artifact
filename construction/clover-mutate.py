#!/usr/bin/env python3
"""
clover-mutate.py — meaning-preserving control mutations for the CloverBench slice.

Why this exists
---------------
The SQL slice separates a judge that reads from one that bets on provenance,
because 488 of its confirmed rows are mutated. The Dafny slice has no such
family: every confirmed row is an untouched ground truth, so "looks untouched,
therefore faithful" still scores perfectly there. This builds the missing
family.

The oracle is stronger here than in SQL
---------------------------------------
For SQL, execution can only ever prove that two queries DIFFER; equivalence had
to be argued algebraically by hand. Dafny can prove the equivalence outright.
For each candidate we emit

    lemma Equiv_<n><TP>(<params>, <returns>)
      requires <original preconditions>
      ensures  (<original postconditions>) <==> (<mutated postconditions>)
    {}

and keep the candidate only if Dafny discharges it with an empty body. A rule
that misfires produces a lemma that does not verify and the row is dropped, so
a buggy rewrite cannot silently enter the benchmark as `confirmed`. Mutations
that change a precondition get a second lemma for `pre_orig <==> pre_mut`.

Contracts using `old()` need a two-state context, so their lemma is emitted as
`twostate lemma` and the arrays are passed as parameters that the caller has
already havocked. Where that fails the row is simply dropped.

Usage:
    python3 construction/clover-mutate.py --dafny <path to dafny> --out construction/benchmark/clover-control.jsonl
    python3 construction/clover-mutate.py --keep-work    # leave the .dfy files for inspection
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLOVER = HERE / "benchmark" / "clover.jsonl"

SPEC_KW = ("requires", "ensures", "modifies", "reads", "decreases")


# ----------------------------------------------------------------- parsing

class Contract:
    """A Clover method contract, split into a signature and its clauses.

    Clover contracts are one method with no helper declarations, so a
    line-oriented split is exact. Everything before the first spec keyword is
    the signature; each spec line is one clause. A clause may wrap onto
    continuation lines, which are those that do not start with a spec keyword.
    """

    def __init__(self, text):
        self.raw = text
        lines = [l for l in text.splitlines() if l.strip()]
        sig, clauses, cur = [], [], None
        for line in lines:
            kw = next((k for k in SPEC_KW if re.match(rf"\s*{k}\b", line)), None)
            if kw:
                if cur:
                    clauses.append(cur)
                cur = [kw, line.strip()[len(kw):].strip()]
            elif cur:
                cur[1] += " " + line.strip()
            else:
                sig.append(line.strip())
        if cur:
            clauses.append(cur)
        self.signature = " ".join(sig)
        self.clauses = clauses  # list of [keyword, body]

    # -- signature pieces, needed to give the lemma the same variables -----
    @property
    def name(self):
        m = re.search(r"\bmethod\s+([A-Za-z_]\w*)", self.signature)
        return m.group(1) if m else None

    def _type_param_span(self):
        """(start, end) of the `<...>` after the method name, or None.

        A naive `<[^>]*>` is wrong and a naive `find('(')` is worse: type
        parameters carry characteristics written with parentheses, as in
        `<T(0)>`, `<T(==)>` and `<K(!new), V>`. Taking the first `(` in the
        signature then grabs the inside of a characteristic and the emitted
        lemma does not parse. Scanning for the balanced `>` is what makes the
        parameter list start in the right place.
        """
        m = re.search(r"\bmethod\s+[A-Za-z_]\w*", self.signature)
        if not m:
            return None
        i = m.end()
        while i < len(self.signature) and self.signature[i].isspace():
            i += 1
        if i >= len(self.signature) or self.signature[i] != "<":
            return None
        depth = 0
        for j in range(i, len(self.signature)):
            depth += (self.signature[j] == "<") - (self.signature[j] == ">")
            if depth == 0:
                return (i, j + 1)
        return None

    @property
    def type_params(self):
        span = self._type_param_span()
        return self.signature[span[0]:span[1]] if span else ""

    def _params_start(self):
        """Index to start looking for the value parameter list."""
        span = self._type_param_span()
        if span:
            return span[1]
        m = re.search(r"\bmethod\s+[A-Za-z_]\w*", self.signature)
        return m.end() if m else 0

    def _paren_after(self, idx):
        """The balanced parenthesised group starting at or after `idx`."""
        start = self.signature.find("(", idx)
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(self.signature)):
            depth += (self.signature[i] == "(") - (self.signature[i] == ")")
            if depth == 0:
                return self.signature[start + 1:i]
        return None

    @property
    def params(self):
        return (self._paren_after(self._params_start()) or "").strip()

    @property
    def returns(self):
        m = re.search(r"\breturns\b", self.signature)
        if not m:
            return ""
        return (self._paren_after(m.end()) or "").strip()

    def bodies(self, kw):
        return [b for k, b in self.clauses if k == kw]

    def conj(self, kw):
        """The clauses of one kind as a single parenthesised conjunction."""
        parts = self.bodies(kw)
        if not parts:
            return "true"
        return " && ".join(f"({p})" for p in parts)

    def uses_old(self):
        return "old(" in self.raw

    def render(self, clauses):
        out = [self.signature]
        for kw, body in clauses:
            out.append(f"  {kw} {body}")
        return "\n".join(out)


# ------------------------------------------------------------------- rules
#
# Every rule returns a new clause list or None if it does not apply. A rule
# must be an identity on any input, not merely on the cases Dafny happens to
# check. The emitted lemma is what enforces that, so a rule that overreaches is
# caught rather than shipped.
#
# Rules are ordered roughly by how much they LOOK like a meaning change. A
# control only does its job if a reader cannot dismiss it at a glance, which is
# the lesson from the SQL slice, where 79% of the controls turned out to be
# no-ops anyone could see.

def _first_clause(clauses, kw, pred):
    for i, (k, body) in enumerate(clauses):
        if k == kw and pred(body):
            return i
    return None


def _split_top(expr, op):
    """Split `expr` on the top-level occurrences of `op`, respecting nesting.

    Splitting on a naive `.split('==>')` would cut inside a quantifier body or
    a parenthesised group and produce two halves that are not expressions.
    """
    # A single bar is a cardinality or a set-comprehension guard, as in
    # `|set i | i in numbers && i < threshold|`, and it is not a bracket. Bars
    # cannot be matched by counting either: that comprehension has three of
    # them, so tracking parity puts the `&&` outside the comprehension and the
    # split still lands mid-expression. Refusing to split such a clause costs
    # two candidates and cannot go wrong.
    if re.search(r"(?<!\|)\|(?!\|)", expr):
        return [expr]

    out, depth, i, last = [], 0, 0, 0
    while i < len(expr):
        c = expr[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif depth == 0 and expr.startswith("::", i):
            # A quantifier body runs to the end of the expression unless
            # parenthesised, so every operator to the right of an unbracketed
            # `::` is inside the binder's scope. Splitting there detaches the
            # bound variable from its quantifier and the halves no longer
            # resolve. `(forall k :: P) && Q` keeps its `::` at depth 1 and is
            # still split correctly.
            break
        elif depth == 0 and expr.startswith(op, i):
            # `<==>` ends in `==>`, so a scan for implication finds one inside
            # every equivalence. Splitting there yields `(A) <` as a left half,
            # which is not an expression and does not parse.
            before = expr[i - 1] if i else " "
            if not (op == "==>" and before in "<="):
                out.append(expr[last:i])
                last = i + len(op)
                i += len(op)
                continue
        i += 1
    out.append(expr[last:])
    return out


def r_contrapositive(clauses):
    """`P ==> Q` becomes `!(Q) ==> !(P)`.

    Textually this is a large edit that reverses the direction of the arrow,
    which is what makes it a useful control. It is also the shape of a real
    specification bug, so a judge cannot clear it without following the logic.
    """
    for i, (k, body) in enumerate(clauses):
        if k != "ensures":
            continue
        parts = _split_top(body, "==>")
        if len(parts) != 2:
            continue
        p, q = parts[0].strip(), parts[1].strip()
        if not p or not q or "forall" in p or "exists" in p:
            continue
        new = list(clauses)
        new[i] = (k, f"!({q}) ==> !({p})")
        return new, "contrapositive"
    return None


def r_forall_to_not_exists(clauses):
    """`forall x :: R ==> P` becomes `!exists x :: R && !(P)`.

    The quantifier changes, the connective changes, and the body gains a
    negation. Nothing about the surface says these agree.
    """
    for i, (k, body) in enumerate(clauses):
        if k != "ensures" or not body.strip().startswith("forall"):
            continue
        m = re.match(r"forall\s+(.+?)\s*::\s*(.+)$", body.strip(), re.S)
        if not m:
            continue
        binders, inner = m.group(1), m.group(2)
        parts = _split_top(inner, "==>")
        if len(parts) != 2:
            continue
        r, p = parts[0].strip(), parts[1].strip()
        new = list(clauses)
        new[i] = (k, f"!(exists {binders} :: ({r}) && !({p}))")
        return new, "forall_to_not_exists"
    return None


def r_bound_shift(clauses):
    """`i < N` becomes `i <= N-1` inside a quantifier range.

    This is the control half of a pair. Its semantic twin changes `i < N` to
    `i <= N`, which reads almost identically and is the classic off-by-one. A
    judge that flags a shifted bound on sight is wrong here and right there.
    """
    for i, (k, body) in enumerate(clauses):
        if k != "ensures" or ("forall" not in body and "exists" not in body):
            continue
        m = re.search(r"<\s*([A-Za-z_][\w.\[\]]*(?:\.Length)?)\s*(==>|&&)", body)
        if not m:
            continue
        new_body = body[:m.start()] + f"<= {m.group(1)}-1 {m.group(2)}" + body[m.end():]
        new = list(clauses)
        new[i] = (k, new_body)
        return new, "bound_shift"
    return None


def r_iff_split(clauses):
    """`A <==> B` becomes `(A ==> B) && (B ==> A)`."""
    for i, (k, body) in enumerate(clauses):
        if k != "ensures":
            continue
        parts = _split_top(body, "<==>")
        if len(parts) != 2:
            continue
        a, b = parts[0].strip(), parts[1].strip()
        if not a or not b:
            continue
        new = list(clauses)
        new[i] = (k, f"(({a}) ==> ({b})) && (({b}) ==> ({a}))")
        return new, "iff_split"
    return None


def r_arith_move(clauses):
    """`X + Y == Z` becomes `X == Z - Y`.

    Dafny integers are unbounded, so this is an identity rather than an
    overflow hazard. The rewritten clause names a different quantity than the
    one the docstring mentions, which is what a reader has to see through.
    """
    # `+` is also sequence concatenation, and `-` is not defined on sequences,
    # so `a[..] + [b] == c[..]` would be rewritten into an expression that does
    # not typecheck. An operand qualifies only if it is an identifier with
    # optional field access and simple integer indexing, which rules out slices
    # like `a[..]` and displays like `[b]`.
    operand = r"[A-Za-z_]\w*(?:\.\w+|\[\s*[A-Za-z_0-9]+\s*\])*"
    for i, (k, body) in enumerate(clauses):
        if k != "ensures":
            continue
        m = re.match(rf"^\s*({operand})\s*\+\s*({operand})\s*==\s*({operand})\s*$", body)
        if not m:
            continue
        x, y, z = m.groups()
        new = list(clauses)
        new[i] = (k, f"{x} == {z} - {y}")
        return new, "arith_move"
    return None


def r_split_conjunct(clauses):
    """One `ensures P && Q` becomes two clauses.

    Multiple ensures are conjoined, so this is an identity, but the contract
    now has a different number of obligations than the docstring enumerates.
    """
    idx = _first_clause(clauses, "ensures", lambda b: len(_split_top(b, "&&")) == 2)
    if idx is None:
        return None
    a, b = (p.strip() for p in _split_top(clauses[idx][1], "&&"))
    if not a or not b:
        return None
    new = clauses[:idx] + [("ensures", a), ("ensures", b)] + clauses[idx + 1:]
    return new, "split_conjunct"


def r_merge_ensures(clauses):
    """Two `ensures` clauses become one conjunction. The inverse of the above."""
    idxs = [i for i, (k, _) in enumerate(clauses) if k == "ensures"]
    if len(idxs) < 2:
        return None
    i, j = idxs[0], idxs[1]
    merged = f"({clauses[i][1]}) && ({clauses[j][1]})"
    new = [c for n, c in enumerate(clauses) if n != j]
    new[i] = ("ensures", merged)
    return new, "merge_ensures"


def r_commute_eq(clauses):
    """`A == B` becomes `B == A`, the cheapest control and the least convincing."""
    for i, (k, body) in enumerate(clauses):
        if k != "ensures":
            continue
        m = re.match(r"^\s*([\w.\[\]]+)\s*==\s*([\w.\[\]]+)\s*$", body)
        if not m:
            continue
        new = list(clauses)
        new[i] = (k, f"{m.group(2)} == {m.group(1)}")
        return new, "commute_eq"
    return None


RULES = [r_contrapositive, r_forall_to_not_exists, r_bound_shift, r_iff_split,
         r_arith_move, r_split_conjunct, r_merge_ensures, r_commute_eq]


# -------------------------------------------------------------- verification

def vacuity_lemma_for(c, n):
    """The check that the equivalence proof was not vacuous.

    An equivalence `P <==> Q` holds trivially whenever P and Q are both
    unsatisfiable, and the postconditions here can be unsatisfiable for a
    reason that has nothing to do with the mutation. `fresh(a)` is the case
    that found this: it is a two-state predicate, so in a lemma's single state
    Dafny proves `!fresh(a)`, every conjunction containing it is false on both
    sides, and the equivalence goes through no matter what the rest of the
    clause says. `seq_to_array` accepted both a dropped lower bound and a
    `forall` turned into an `exists` for exactly that reason.

    If Dafny can prove the ORIGINAL postcondition false under the
    preconditions, the pair proves nothing and the candidate is dropped. This
    is the Dafny form of the vacuity problem the SQL slice has with controls
    whose gold query returns no rows.
    """
    params = re.sub(r"\bghost\s+", "",
                    ", ".join(p for p in (c.params, c.returns) if p))
    kind = "twostate lemma" if c.uses_old() else "lemma"
    return (
        f"{kind} Vac_{n}_{c.name}{c.type_params}({params})\n"
        f"  requires {c.conj('requires')}\n"
        f"  ensures !({c.conj('ensures')})\n"
        "{}\n"
    )


def lemma_for(c, mutated, n):
    """The proof obligation that the mutated contract means what the original does.

    Preconditions are assumed rather than compared when the rule left them
    alone, which is every rule here. They are still needed as hypotheses,
    because a postcondition may only be well formed under them.
    """
    mc = Contract(c.render(mutated))
    # A lemma is ghost throughout, so a `ghost` modifier on one of its formals
    # is rejected. `onlineMax` returns `ghost m: int`, which is legal on the
    # method and illegal once the same binder is carried onto the lemma.
    params = re.sub(r"\bghost\s+", "",
                    ", ".join(p for p in (c.params, c.returns) if p))
    pre = c.conj("requires")
    kind = "twostate lemma" if c.uses_old() else "lemma"
    return (
        f"{kind} Equiv_{n}_{c.name}{c.type_params}({params})\n"
        f"  requires {pre}\n"
        f"  ensures ({c.conj('ensures')})\n"
        f"      <==> ({mc.conj('ensures')})\n"
        "{}\n"
    )


def verify(dafny, path, timeout, expected):
    """Return the indices of candidates Dafny could not prove.

    A parse or resolution error aborts Dafny before it verifies anything, and
    the first version of this function read that as "everything not named in an
    error message was proved". It reported 126 of 140 proved on a file that had
    never been verified at all. Nothing may be treated as proved unless Dafny
    printed its own tally and that tally accounts for every candidate, so the
    two failure modes are now separated and only a real verification run
    returns a verdict.
    """
    try:
        p = subprocess.run([dafny, "verify", "--allow-warnings", str(path)],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise SystemExit(f"dafny timed out after {timeout}s on {path}")
    blob = p.stdout + p.stderr

    for fatal in ("parse errors detected", "resolution errors detected"):
        if fatal in blob:
            names = sorted(set(re.findall(r"Equiv_(\d+)_\w+", blob)), key=int)
            raise SystemExit(
                f"dafny reported {fatal} and verified nothing.\n"
                f"candidates implicated: {names[:20]}\n"
                f"{blob[-2000:]}")

    m = re.search(r"(\d+) verified, (\d+) error", blob)
    if not m:
        raise SystemExit(f"no verification tally from dafny:\n{blob[-2000:]}")
    errors = int(m.group(2))

    # Dafny's tally counts proof obligations, not lemmas, and one lemma raises
    # several. Attribution therefore goes by source line: `expected` maps a
    # starting line to the candidate emitted there, and an error belongs to the
    # last lemma that started at or before it.
    starts = sorted(expected)
    bad = set()
    reported = 0
    for line in (int(n) for n in re.findall(rf"{re.escape(path.name)}\((\d+),\d+\): Error", blob)):
        reported += 1
        owner = [s for s in starts if s <= line]
        if not owner:
            raise SystemExit(f"error on line {line} precedes every lemma:\n{blob[-2000:]}")
        bad.add(expected[owner[-1]])
    if reported == 0 and errors:
        raise SystemExit(f"dafny reported {errors} errors but none were locatable:\n{blob[-2000:]}")
    return bad


def selftest(dafny, work):
    """Refuse to run unless the oracle still discriminates.

    Two obligations that must come out opposite ways. If both pass, error
    attribution has drifted and every verdict this script produces is
    meaningless, which is how the vacuity hole was found in the first place.
    """
    p = work / "selftest.dfy"
    p.write_text(
        "lemma MustPass(n: int, a: int)\n"
        "  ensures (0 <= n && a == n) <==> (a == n && n >= 0)\n{}\n\n"
        "lemma MustFail(n: int, a: int)\n"
        "  ensures (0 <= n) <==> (0 < n)\n{}\n", encoding="utf8")
    lines = {1: ("x", "pass"), 5: ("x", "fail")}
    bad = verify(dafny, p, 120, lines)
    got = {v for _k, v in bad}
    if got != {"fail"}:
        raise SystemExit(f"oracle selftest failed: expected only the false "
                         f"equivalence to be rejected, got {got or 'nothing'}")
    print("oracle selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dafny", default="dafny")
    ap.add_argument("--data", type=Path, default=CLOVER)
    ap.add_argument("--out", type=Path, default=HERE / "benchmark" / "clover-control.jsonl")
    ap.add_argument("--work", type=Path, default=HERE / "clover-control-work")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.data.read_text(encoding="utf8").splitlines() if l.strip()]
    gt = [r for r in rows if r.get("clover_class") == "ground_truth"]
    print(f"{len(gt)} ground-truth contracts")

    args.work.mkdir(parents=True, exist_ok=True)
    selftest(args.dafny, args.work)

    # ---- generate candidates ------------------------------------------------
    cands, skipped = [], []
    for r in gt:
        c = Contract(r["formal"])
        if not c.name:
            skipped.append((r["name"], "no method signature"))
            continue
        for rule in RULES:
            got = rule(list(c.clauses))
            if not got:
                continue
            mutated, rule_name = got
            text = c.render(mutated)
            if text.strip() == c.raw.strip():
                continue
            cands.append(dict(row=r, contract=c, mutated=mutated,
                              rule=rule_name, formal=text))

    print(f"{len(cands)} candidates from {len(RULES)} rules")

    # ---- one file, one lemma per candidate ---------------------------------
    src = args.work / "equiv.dfy"
    body = ["// generated by clover-mutate.py -- two obligations per candidate\n"]
    line_of = {}  # first line of each lemma -> (kind, candidate index)
    for n, cd in enumerate(cands):
        body.append(f"// [{n}] {cd['row']['name']} :: {cd['rule']}")
        line_of[sum(b.count("\n") + 1 for b in body) + 1] = ("equiv", n)
        body.append(lemma_for(cd["contract"], cd["mutated"], n))
        line_of[sum(b.count("\n") + 1 for b in body) + 1] = ("vac", n)
        body.append(vacuity_lemma_for(cd["contract"], n))
    src.write_text("\n".join(body), encoding="utf8")
    print(f"wrote {src}")

    failed = verify(args.dafny, src, args.timeout, line_of)
    unproved_equiv = {n for kind, n in failed if kind == "equiv"}
    # A vacuity lemma that FAILS is the good outcome: it means Dafny could not
    # show the original postcondition false, so the equivalence said something.
    vacuous = {n for n in range(len(cands)) if ("vac", n) not in failed}
    kept = [cd for n, cd in enumerate(cands)
            if n not in unproved_equiv and n not in vacuous]
    print(f"{len(cands)} candidates: {len(unproved_equiv)} not proved equivalent, "
          f"{len(vacuous)} vacuous, {len(kept)} kept")

    # ---- emit ---------------------------------------------------------------
    out = []
    for cd in kept:
        r = cd["row"]
        name = f"{r['name']}__ctrl_{cd['rule']}"
        out.append(dict(
            id=f"clover/dafny/{name}", name=name,
            informal=r["informal"], formal=cd["formal"], context="",
            source="clover", language="dafny",
            expected="confirmed", variant="control_mutation",
            mutation_rule=cd["rule"], mutation_of=r["id"],
            category=None, meta_orig=r["formal"],
            clover_class="control", clover_category=None,
        ))
    args.out.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out),
                        encoding="utf8")
    print(f"wrote {len(out)} control rows -> {args.out}")

    by = {}
    for o in out:
        by[o["mutation_rule"]] = by.get(o["mutation_rule"], 0) + 1
    print("\nkept by rule:")
    for k in sorted(by, key=lambda x: -by[x]):
        print(f"  {k:<24}{by[k]:>4}")
    if skipped:
        print(f"\nskipped {len(skipped)}: {skipped[:5]}")


if __name__ == "__main__":
    main()
