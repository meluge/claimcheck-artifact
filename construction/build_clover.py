#!/usr/bin/env python3
"""Assemble the CloverBench faithfulness slice as a standalone dataset.

Emits the same schema as build_benchmark.py so this file can later be
concatenated into faithfulness.jsonl without a second pass, but it is built
and run on its own first.

CloverBench ships 62 hand-written textbook programs plus four hand-crafted
incorrect variants of each. Every variant still verifies in Dafny -- the
adversarial_incorrect/README.md is explicit about it -- so nothing here can be
caught by running the verifier. That is the point: this slice isolates the
cases where only a meaning check finds the defect.

  class          code   annotation  docstring   -> expected
  ground_truth   -      -           -              confirmed
  C1             -      -           mutated        disputed
  C2             -      mutated     -              disputed
  C6             mutated mutated    -              disputed
  C3             -      mutated     mutated        UNLABELLED, see below

C3 weakens annotation and docstring together, so the pair often still agrees
and would be `confirmed` for this benchmark's question even though Clover
calls it incorrect for its own. A sample showed that agreement is not
reliable (`find`'s docstring says "possibly empty array" while its annotation
requires `a.Length > 0`; `swap_in_array`'s docstring claims the untouched
elements equal -1 while the annotation says they are unchanged). Those 62 rows
are emitted with expected=None so a later hand-labelling pass can fill them in
without regenerating anything. They are excluded from the runnable set.

Two things this deliberately does NOT decide:

  * Directionality. Clover marks a variant incorrect whenever docstring and
    annotation differ in either direction. This benchmark may or may not want
    "the spec guarantees more than the docstring claims" to count as disputed.
    That covers C1's 20 `doc- too weak` rows and C2's `post- too strong` ones.
    `clover_category` carries Clover's own hand annotation so the decision can
    be made once, as a filter, rather than baked in here.
  * What to do with a docstring that is itself wrong (C1). Those rows are
    disputed because docstring and annotation disagree, which is the question
    asked -- but the defect is on the informal side, a polarity the rest of the
    benchmark does not have.

Usage:
    python3 construction/build_clover.py
    python3 construction/build_clover.py --include-c3 --out /tmp/clover-all.jsonl
"""

import argparse
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CLOVER = Path.home() / "Downloads" / "Clover-main" / "dataset" / "CloverBench"

# Where each class keeps its docstring and its .dfy, relative to the class dir.
# The filenames inside a class directory are not trustworthy -- C1's docstring
# lives in `<p>_C2.txt` and C2's annotation lives in `<p>_C1.dfy` -- so the
# parent directory is what identifies the class, never the filename.
# A `None` docstring means the class does not mutate it: take the ground truth's.
LAYOUT = {
    "ground_truth": ("textbook_algo",            "{p}_spec.txt", "{p}_strong.dfy"),
    "C1":           ("adversarial_incorrect/C1", "{p}_C2.txt",   "{p}_strong.dfy"),
    "C2":           ("adversarial_incorrect/C2", None,           "{p}_C1.dfy"),
    "C3":           ("adversarial_incorrect/C3", "{p}_doc.txt",  "{p}_C1.dfy"),
    "C6":           ("adversarial_incorrect/C6", None,           "{p}.dfy"),
}

EXPECTED = {"ground_truth": "confirmed", "C1": "disputed",
            "C2": "disputed", "C6": "disputed", "C3": None}

# `variant` in the shared schema only knows original / control_mutation /
# semantic_mutation. C1 mutates the natural language rather than the formal,
# which none of the existing values describe, so it gets its own.
VARIANT = {"ground_truth": "original", "C1": "informal_mutation",
           "C2": "semantic_mutation", "C6": "semantic_mutation",
           "C3": "paired_mutation"}

MUTATION_RULE = {"ground_truth": None, "C1": "doc_mutated",
                 "C2": "annotation_mutated", "C6": "code_and_annotation_mutated",
                 "C3": "doc_and_annotation_mutated"}

RUNNABLE = ["ground_truth", "C1", "C2", "C6"]

# Everything that can sit between the signature and the body, so the body
# detector does not mistake `ensures c == {1,2}` for the opening brace.
SPEC_KW = r"(requires|ensures|modifies|reads|decreases)"
# ...but only these two say what the method guarantees. `modifies s` on its own
# is a framing clause: there is nothing to informalize and nothing to compare a
# docstring against, so such a row cannot be scored either way.
BEHAVIOURAL_KW = r"(requires|ensures)"


def read(path):
    return path.read_text(encoding="utf8", errors="replace") if path.exists() else None


# ------------------------------------------------------------------- contract

def split_contract(dfy):
    """Return (signature + spec clauses, set of behavioural keywords present).

    `formal` has to be the contract and nothing else -- the body is the thing
    claimcheck is explicitly not auditing. Clover files are a single method
    with no helper declarations (verified: 0 of 62 define a function, predicate
    or datatype), so the contract is everything from the declaration up to the
    line that opens the body.
    """
    lines = dfy.splitlines()
    start = next((i for i, l in enumerate(lines)
                  if re.match(r"\s*(method|function|predicate|lemma)\b", l)), None)
    if start is None:
        return "", False

    out, depth = [], 0
    for line in lines[start:]:
        # The body opens at the first `{` that is not inside the parameter list
        # or a set/map display in a clause. Tracking paren depth keeps
        # `ensures c == {1,2}` from being mistaken for the body.
        depth += line.count("(") - line.count(")")
        brace = line.find("{")
        if brace != -1 and depth <= 0 and not re.match(rf"\s*{SPEC_KW}\b", line):
            head = line[:brace].rstrip()
            if head:
                out.append(head)
            break
        out.append(line.rstrip())

    text = "\n".join(l for l in out if l.strip())
    # `\b` after the keyword rather than `^\s*` before it, so a clause written
    # on the signature line still counts.
    kinds = set(re.findall(rf"\b{BEHAVIOURAL_KW}\b", text))
    return text.strip(), kinds


# ------------------------------------------------------------------ categories

def load_categories(root):
    """Clover's own hand annotation of what each mutation did.

    C1 and C2 ship a summary.json mapping category -> [program]; C6 puts a
    one-line category.txt in each program directory. The per-directory files in
    C2 duplicate summary.json but carry typos ('pre good post too weak', a
    stray quote, 'pre- too good'), so summary.json wins where both exist.
    """
    cats = {}
    for cls in ("C1", "C2"):
        path = root / "adversarial_incorrect" / cls / "summary.json"
        if not path.exists():
            continue
        for category, progs in json.loads(path.read_text(encoding="utf8")).items():
            for p in progs:
                cats[(cls, p)] = category

    c6 = root / "adversarial_incorrect" / "C6"
    if c6.is_dir():
        for d in sorted(x for x in c6.iterdir() if x.is_dir()):
            txt = read(d / "category.txt")
            if txt and txt.strip():
                cats[("C6", d.name)] = re.sub(r"\s+", " ", txt.strip())
    return cats


# ------------------------------------------------------------------------ rows

def row(**kw):
    base = dict(id=None, name=None, informal="", formal="", context="", source=None,
                language=None, expected=None, variant="original",
                mutation_rule=None, mutation_of=None, category=None, meta_orig=None,
                clover_class=None, clover_category=None)
    base.update(kw)
    return base


def build(root, include_c3):
    gt_dir = root / "textbook_algo"
    progs = sorted(d.name for d in gt_dir.iterdir() if d.is_dir())
    cats = load_categories(root)

    rows, dropped, missing, precondition_only = [], [], [], []
    classes = RUNNABLE + (["C3"] if include_c3 else [])

    for p in progs:
        gt_doc = read(gt_dir / p / f"{p}_spec.txt")
        gt_dfy = read(gt_dir / p / f"{p}_strong.dfy")
        if gt_doc is None or gt_dfy is None:
            missing.append(("ground_truth", p, "docstring or .dfy absent"))
            continue
        gt_formal, _ = split_contract(gt_dfy)

        for cls in classes:
            subdir, doc_tmpl, dfy_tmpl = LAYOUT[cls]
            d = root / subdir / p
            dfy = read(d / dfy_tmpl.format(p=p))
            if dfy is None:
                missing.append((cls, p, f"{dfy_tmpl.format(p=p)} absent"))
                continue

            # C2 and C6 leave the docstring alone, so it comes from the ground
            # truth rather than from the variant directory.
            doc = read(d / doc_tmpl.format(p=p)) if doc_tmpl else gt_doc
            if doc is None:
                missing.append((cls, p, f"{doc_tmpl.format(p=p)} absent"))
                continue

            formal, kinds = split_contract(dfy)
            if not kinds:
                # Nothing but a signature, or only framing clauses like
                # `modifies s`: there is nothing to informalize and nothing to
                # compare, so the row cannot be scored either way.
                dropped.append((cls, p, "no requires/ensures"))
                continue
            if "ensures" not in kinds:
                # A precondition-only contract guarantees nothing about the
                # result. The row is still scoreable -- "the spec says nothing
                # about what is returned" is exactly the vacuity finding this
                # benchmark asks for -- but it is worth knowing it is in there.
                precondition_only.append((cls, p))

            is_gt = cls == "ground_truth"
            name = p if is_gt else f"{p}__{cls}"
            rows.append(row(
                id=f"clover/dafny/{name}", name=name,
                informal=doc.strip(), formal=formal, context="",
                source="clover", language="dafny",
                expected=EXPECTED[cls], variant=VARIANT[cls],
                mutation_rule=MUTATION_RULE[cls],
                mutation_of=None if is_gt else f"clover/dafny/{p}",
                category=None,
                # Only the classes that move the formal have a previous version
                # of it; C1's formal is the ground truth's, so a diff would be
                # empty and misleading. Its original docstring is one hop away
                # through mutation_of.
                meta_orig=None if cls in ("ground_truth", "C1") else gt_formal,
                clover_class=cls,
                clover_category=cats.get((cls, p)),
            ))

    return rows, dropped, missing, precondition_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clover-dir", type=Path, default=DEFAULT_CLOVER)
    ap.add_argument("--out", type=Path, default=HERE / "benchmark" / "clover.jsonl")
    ap.add_argument("--include-c3", action="store_true",
                    help="also emit the 62 unlabelled C3 rows (expected=null)")
    ap.add_argument("--controls", type=Path, default=HERE / "benchmark" / "clover-control.jsonl",
                    help="meaning-preserving control mutations to merge in, from "
                         "clover-mutate.py. They are built separately because "
                         "generating them needs Dafny to discharge an equivalence "
                         "proof per row, which this script does not require. Pass "
                         "an absent path to build the slice without them.")
    ap.add_argument("--csv", action="store_true", help="also write a .csv view")
    args = ap.parse_args()

    if not (args.clover_dir / "textbook_algo").is_dir():
        raise SystemExit(f"CloverBench not found at {args.clover_dir}\n"
                         f"clone https://github.com/ChuyueSun/Clover and point --clover-dir at "
                         f"dataset/CloverBench")

    rows, dropped, missing, precondition_only = build(args.clover_dir, args.include_c3)

    # Controls are appended rather than regenerated here, and every one must
    # descend from a ground truth this build actually emitted. A control whose
    # `mutation_of` does not resolve would be a row built against a different
    # version of the slice, which is the one way a stale file could enter
    # silently.
    controls = []
    if args.controls and args.controls.exists():
        emitted = {r["id"] for r in rows}
        for line in args.controls.read_text(encoding="utf8").splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            if c["mutation_of"] not in emitted:
                raise SystemExit(f"control {c['id']} descends from {c['mutation_of']}, "
                                 f"which this build did not emit; regenerate with "
                                 f"construction/clover-mutate.py")
            controls.append(c)
        rows += controls

    ids = [r["id"] for r in rows]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate ids: {sorted(dupes)[:5]}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.csv:
        with args.out.with_suffix(".csv").open("w", encoding="utf8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), quoting=csv.QUOTE_ALL)
            w.writeheader()
            w.writerows(rows)

    # ------------------------------------------------------------------ report
    print(f"wrote {len(rows)} rows -> {args.out}")
    if args.csv:
        print(f"      and a csv view -> {args.out.with_suffix('.csv')}")
    print()
    print(f"{'class':<16}{'rows':>6}{'confirmed':>11}{'disputed':>10}{'unlabelled':>12}")
    for cls in RUNNABLE + ["control", "C3"]:
        sub = [r for r in rows if r["clover_class"] == cls]
        if not sub:
            continue
        c = sum(1 for r in sub if r["expected"] == "confirmed")
        d = sum(1 for r in sub if r["expected"] == "disputed")
        print(f"{cls:<16}{len(sub):>6}{c:>11}{d:>10}{len(sub) - c - d:>12}")
    runnable = [r for r in rows if r["expected"]]
    c = sum(1 for r in runnable if r["expected"] == "confirmed")
    print(f"{'RUNNABLE':<16}{len(runnable):>6}{c:>11}{len(runnable) - c:>10}{'':>12}")

    if dropped:
        print(f"\ndropped, nothing to compare ({len(dropped)}):")
        for cls, p, why in dropped:
            print(f"  {cls}/{p}: {why}")
    if precondition_only:
        print(f"\nkept but precondition-only, no ensures ({len(precondition_only)}):")
        for cls, p in precondition_only:
            print(f"  {cls}/{p}")
    if missing:
        print(f"\nmissing files ({len(missing)}):")
        for cls, p, why in missing:
            print(f"  {cls}/{p}: {why}")

    print("\nclover_category (Clover's own hand annotation):")
    by = {}
    for r in rows:
        k = (r["clover_class"], r["clover_category"])
        by[k] = by.get(k, 0) + 1
    for k in sorted(by, key=lambda x: (x[0], str(x[1]))):
        if k[0] == "ground_truth":
            continue
        print(f"  {k[0]:<4}{str(k[1]):<34}{by[k]:>4}")


if __name__ == "__main__":
    main()
