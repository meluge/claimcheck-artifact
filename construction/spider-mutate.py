#!/usr/bin/env python3
"""
spider-mutate.py — Build a text-to-SQL mutation benchmark for claimcheck.

Pipeline:
  1. Sample base (question, gold SQL) pairs from Spider dev.json.
  2. Apply rule-based mutations, two families:
       - control  : intended to PRESERVE meaning  -> should stay `confirmed`
       - semantic : intended to CHANGE meaning     -> should be flagged `disputed`
  3. Execute original + mutated on the real Spider SQLite DB and compare result
     sets. The EXECUTION decides the ground-truth label, not the rule's intent:
       result sets equal    -> expected = confirmed
       result sets differ   -> expected = disputed
       mutated query errors -> candidate dropped
  4. Balance to ~N rows and emit the claimcheck dataset.

Output (in --out dir):
  dataset.jsonl   one rich row per line (requirement, dafnyCode=SQL, domain=db_id,
                  expected, family, rule, orig_sql, question)
  summary.txt     intended-vs-actual behavior of each rule

Usage:
  python3 construction/spider-mutate.py --spider <spider_data dir> --n 50 --out construction/spider-bench
"""

import argparse
import json
import os
import random
import sqlite3
import sys
from collections import defaultdict

import sqlglot
from sqlglot import expressions as exp

DIALECT = "sqlite"


# ----------------------------------------------------------------------------
# Mutation rules. Each takes a parsed AST (copy) and returns a mutated AST or
# None if the rule does not apply. Rules must not mutate the input in place.
# ----------------------------------------------------------------------------

def _copy(tree):
    return tree.copy()


# --- control family (intended to preserve semantics) ---

def m_identity(tree):
    # Re-render only; tests invariance to formatting/normalization.
    return _copy(tree)


def m_commute_eq(tree):
    t = _copy(tree)
    for eq in t.find_all(exp.EQ):
        left, right = eq.this, eq.expression
        eq.set("this", right)
        eq.set("expression", left)
        return t
    return None


def m_reorder_and(tree):
    t = _copy(tree)
    for a in t.find_all(exp.And):
        left, right = a.this, a.expression
        a.set("this", right)
        a.set("expression", left)
        return t
    return None


def m_count_star_to_one(tree):
    t = _copy(tree)
    for c in t.find_all(exp.Count):
        if isinstance(c.this, exp.Star):
            c.set("this", exp.Literal.number(1))
            return t
    return None


# --- semantic family (intended to change semantics) ---

_AGG_SWAP = {exp.Max: exp.Min, exp.Min: exp.Max, exp.Sum: exp.Avg, exp.Avg: exp.Sum}


def m_swap_aggregate(tree):
    t = _copy(tree)
    for node in t.walk():
        n = node[0] if isinstance(node, tuple) else node
        cls = type(n)
        if cls in _AGG_SWAP:
            new = _AGG_SWAP[cls](this=n.this)
            n.replace(new)
            return t
    return None


def m_flip_order(tree):
    t = _copy(tree)
    for o in t.find_all(exp.Ordered):
        o.set("desc", not o.args.get("desc"))
        return t
    return None


_CMP_SWAP = {exp.GT: exp.LT, exp.LT: exp.GT, exp.GTE: exp.LTE, exp.LTE: exp.GTE}


def m_flip_comparison(tree):
    t = _copy(tree)
    for node in t.walk():
        n = node[0] if isinstance(node, tuple) else node
        cls = type(n)
        if cls in _CMP_SWAP:
            new = _CMP_SWAP[cls](this=n.this, expression=n.expression)
            n.replace(new)
            return t
    return None


def m_drop_where_conjunct(tree):
    t = _copy(tree)
    sel = t if isinstance(t, exp.Select) else t.find(exp.Select)
    if sel is None:
        return None
    where = sel.args.get("where")
    if where is None:
        return None
    cond = where.this
    if isinstance(cond, exp.And):
        sel.set("where", exp.Where(this=cond.this))  # keep one predicate
    else:
        sel.set("where", None)  # drop the filter entirely
    return t


def m_toggle_distinct(tree):
    t = _copy(tree)
    sel = t if isinstance(t, exp.Select) else t.find(exp.Select)
    if sel is None:
        return None
    if sel.args.get("distinct"):
        sel.set("distinct", None)
    else:
        sel.set("distinct", exp.Distinct())
    return t


def m_bump_limit(tree):
    t = _copy(tree)
    for lim in t.find_all(exp.Limit):
        val = lim.expression
        if isinstance(val, exp.Literal) and val.is_int:
            lim.set("expression", exp.Literal.number(int(val.this) + 1))
            return t
    return None



# --- harder families -------------------------------------------------------
#
# The three rules below were added after the first round of results showed the
# original controls saturating: 246 of 418 were `commute_eq` (a=b -> b=a) and
# 141 `count_star_to_one`, which are no-ops anyone can see. A control is only
# doing its job if it LOOKS like a meaning change and is not one.
#
# Two constraints shaped these. A control must be equivalent by algebra, on any
# data, because execution can prove difference but never equivalence -- "same
# result on this database" is not a proof. And a semantic mutation has to be
# detectable from the question, the query and the schema alone, since that is
# all the judge ever sees.


def _first_inner_join(sel):
    """The first join with no side (LEFT/RIGHT/FULL), i.e. an inner join."""
    for j in sel.args.get("joins") or []:
        if not (j.args.get("side") or ""):
            return j
    return None


def _inner_join_for(sel, pred):
    """The earliest inner join at which every table `pred` mentions is in scope.

    Moving a predicate into a join's ON clause is only well formed if the tables
    it references have already been introduced by that point. The first version
    of this rule always used the FIRST inner join, and in Spider the filter
    almost always targets the LAST-joined table -- so 93 of 97 rows came out
    with a forward reference. SQLite accepts those and returns the same rows, so
    the execution check passed them, but they are not valid standard SQL and a
    judge that flags them is right. The rule was measuring "can you spot broken
    SQL", not "can you spot a meaning-preserving rewrite".
    """
    want = {c.table for c in pred.find_all(exp.Column) if c.table}
    scope = set()
    frm = sel.args.get("from")
    if frm:
        for tb in frm.find_all(exp.Table):
            scope.add(tb.alias_or_name)
    for j in sel.args.get("joins") or []:
        for tb in j.find_all(exp.Table):
            scope.add(tb.alias_or_name)
        if not (j.args.get("side") or "") and want <= scope:
            return j
    return None


def _pop_where_conjunct(sel):
    """Remove one conjunct from WHERE and return it, or None."""
    where = sel.args.get("where")
    if where is None:
        return None
    cond = where.this
    if isinstance(cond, exp.And):
        sel.set("where", exp.Where(this=cond.this))
        return cond.expression
    sel.set("where", None)
    return cond


def m_where_to_on_inner(tree):
    """CONTROL: move a WHERE predicate into an INNER JOIN's ON clause.

    For an inner join, ON and WHERE predicates are interchangeable -- both filter
    the joined result -- so this is equivalent on every database. It is the
    control half of a pair: the semantic half performs the same visible edit on a
    LEFT join, where it is NOT equivalent. A judge cannot tell them apart by
    noticing that a predicate moved; it has to know what the join does.
    """
    t = _copy(tree)
    sel = t if isinstance(t, exp.Select) else t.find(exp.Select)
    if sel is None:
        return None
    where = sel.args.get("where")
    if where is None:
        return None
    cond = where.this
    cand = cond.expression if isinstance(cond, exp.And) else cond
    j = _inner_join_for(sel, cand)
    if j is None:
        return None
    moved = _pop_where_conjunct(sel)
    on = j.args.get("on")
    j.set("on", exp.and_(on, moved) if on is not None else moved)
    return t


def m_left_join_where_to_on(tree):
    """SEMANTIC: the same edit, but the join becomes LEFT.

    In a LEFT join a predicate in WHERE runs after the join and discards the
    unmatched left rows, because their NULLs fail it; the same predicate in ON
    runs during the join and those rows survive, padded with NULLs. Textually
    this looks almost identical to the control above -- one keyword differs.
    """
    t = _copy(tree)
    sel = t if isinstance(t, exp.Select) else t.find(exp.Select)
    if sel is None:
        return None
    where = sel.args.get("where")
    if where is None:
        return None
    cond = where.this
    cand = cond.expression if isinstance(cond, exp.And) else cond
    j = _inner_join_for(sel, cand)
    if j is None:
        return None
    moved = _pop_where_conjunct(sel)
    j.set("side", "LEFT")
    on = j.args.get("on")
    j.set("on", exp.and_(on, moved) if on is not None else moved)
    return t


def m_inner_to_left(tree):
    """SEMANTIC: turn an inner join into a left join.

    The result gains every left row that has no match, padded with NULLs. In
    English the difference is plain -- "customers who placed an order" versus
    "all customers, with their orders if any" -- so the question determines which
    one is meant.
    """
    t = _copy(tree)
    sel = t if isinstance(t, exp.Select) else t.find(exp.Select)
    if sel is None:
        return None
    j = _first_inner_join(sel)
    if j is None:
        return None
    j.set("side", "LEFT")
    return t


def m_swap_correlation(tree):
    """SEMANTIC: correlate a subquery on the wrong column.

    Rewrites `WHERE s.k = t.k` inside a subquery to `WHERE s.k = t.j`, where j is
    another column of the same outer table used elsewhere in the query. The query
    still parses, still runs, and still looks like a correlated subquery; only
    reading the schema reveals that it now links the wrong two things.
    """
    t = _copy(tree)
    root = t if isinstance(t, exp.Select) else t.find(exp.Select)
    if root is None:
        return None
    by_table = defaultdict(set)
    for col in t.find_all(exp.Column):
        if col.table:
            by_table[col.table].add(col.name)
    for sub in t.find_all(exp.Select):
        if sub is root:
            continue
        for e in sub.find_all(exp.EQ):
            l, r = e.this, e.expression
            if not (isinstance(l, exp.Column) and isinstance(r, exp.Column)):
                continue
            if not (l.table and r.table and l.table != r.table):
                continue
            # Prefer another column already used on the same alias; fall back to
            # any column name in the query. A name that does not exist on that
            # table makes the query fail to run, and apply_rules drops anything
            # that does not execute -- so generating liberally is safe here.
            all_names = {c.name for c in t.find_all(exp.Column)}
            for side in (r, l):
                alts = sorted(by_table[side.table] - {side.name}) or \
                       sorted(all_names - {side.name})
                if alts:
                    side.set("this", exp.to_identifier(alts[0]))
                    return t
    return None


RULES = [
    ("control", "identity", m_identity),
    ("control", "commute_eq", m_commute_eq),
    ("control", "reorder_and", m_reorder_and),
    ("control", "count_star_to_one", m_count_star_to_one),
    ("control", "where_to_on_inner", m_where_to_on_inner),
    ("semantic", "swap_aggregate", m_swap_aggregate),
    ("semantic", "flip_order", m_flip_order),
    ("semantic", "flip_comparison", m_flip_comparison),
    ("semantic", "drop_where_conjunct", m_drop_where_conjunct),
    ("semantic", "toggle_distinct", m_toggle_distinct),
    ("semantic", "bump_limit", m_bump_limit),
    ("semantic", "left_join_where_to_on", m_left_join_where_to_on),
    ("semantic", "inner_to_left", m_inner_to_left),
    ("semantic", "swap_correlation", m_swap_correlation),
]


# ----------------------------------------------------------------------------
# Execution + result-set comparison
# ----------------------------------------------------------------------------

def db_path(spider_dir, db_id):
    return os.path.join(spider_dir, "database", db_id, f"{db_id}.sqlite")


def run_query(path, sql, timeout_s=10):
    """Return (rows, error). rows is a list of tuples or None on error."""
    try:
        conn = sqlite3.connect(path, timeout=timeout_s)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        cur = conn.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return rows, None
    except Exception as e:  # noqa: BLE001 - any DB error means "unusable candidate"
        return None, str(e)


def get_schema(path):
    """Full DDL of the database: the CREATE TABLE statements, with types + keys."""
    conn = sqlite3.connect(path)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    rows = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql NOT NULL ORDER BY name"
    ).fetchall()
    conn.close()
    return "\n\n".join(r[0].strip().rstrip(";") + ";" for r in rows)


def has_top_order(tree):
    sel = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    return sel is not None and sel.args.get("order") is not None


def result_key(rows, ordered):
    if ordered:
        return [tuple(map(repr, r)) for r in rows]
    return sorted(tuple(map(repr, r)) for r in rows)


# Which rules feed the "control" pool (real rewrites that must preserve results).
# `identity` is excluded — the untouched originals already cover the no-change case.
CONTROL_RULES = [(fam, rule, fn) for (fam, rule, fn) in RULES
                 if fam == "control" and rule != "identity"]
SEMANTIC_RULES = [(fam, rule, fn) for (fam, rule, fn) in RULES if fam == "semantic"]


# ----------------------------------------------------------------------------
# Collection: for each executable base query, gather one original candidate,
# plus control mutations verified UNCHANGED and semantic mutations verified
# CHANGED. Execution is the ground-truth label, applied here as a hard filter.
# ----------------------------------------------------------------------------

def apply_rules(rules, base_tree, gold, path, base_key, ordered):
    """Run a family of rules; return list of (rule, mutated_sql, changed) for
    every candidate that parses, differs from gold, and executes cleanly."""
    out = []
    for _fam, rule, fn in rules:
        try:
            mut_tree = fn(base_tree)
        except Exception:
            mut_tree = None
        if mut_tree is None:
            continue
        mut_sql = mut_tree.sql(dialect=DIALECT)
        if mut_sql.strip() == gold.strip():
            continue  # no-op on this query
        mut_rows, mut_err = run_query(path, mut_sql)
        if mut_err is not None:
            continue  # unexecutable mutation — cannot be labeled
        changed = result_key(mut_rows, ordered) != base_key
        out.append((rule, mut_sql, changed))
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spider", required=True, help="Spider data dir (dev.json, database/)")
    ap.add_argument("--split", default="dev.json")
    ap.add_argument("--n-original", type=int, default=30, help="untouched gold queries")
    ap.add_argument("--n-control", type=int, default=15, help="mutations verified UNCHANGED")
    ap.add_argument("--n-semantic", type=int, default=25, help="mutations verified CHANGED")
    ap.add_argument("--out", default="construction/spider-bench")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-base", type=int, default=800, help="max base queries to scan")
    ap.add_argument("--dev-hardness", default="construction/spider-bench/dev_hardness.json",
                    help="dev examples annotated with official Spider hardness (from spider_hardness.py)")
    ap.add_argument("--levels", default="hard,extra",
                    help="comma-separated difficulty levels to sample from")
    ap.add_argument("--only-rules", default=None,
                    help="comma-separated rule names; restrict generation to these. "
                         "Use it to build a dataset of only the newer, harder rules "
                         "and measure them on their own before mixing them in.")
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    ctrl_rules, sem_rules = CONTROL_RULES, SEMANTIC_RULES
    if args.only_rules:
        keep = {r.strip() for r in args.only_rules.split(",") if r.strip()}
        known = {r for _f, r, _fn in RULES}
        unknown = keep - known
        if unknown:
            raise SystemExit(f"unknown rule(s): {sorted(unknown)}\nknown: {sorted(known)}")
        ctrl_rules = [r for r in CONTROL_RULES if r[1] in keep]
        sem_rules = [r for r in SEMANTIC_RULES if r[1] in keep]
        print(f"restricted to: control={[r[1] for r in ctrl_rules]} "
              f"semantic={[r[1] for r in sem_rules]}", file=sys.stderr)

    # Load the hardness-annotated dev set and keep only the requested levels.
    levels = {s.strip() for s in args.levels.split(",") if s.strip()}
    examples = json.load(open(args.dev_hardness))
    examples = [ex for ex in examples if ex.get("hardness") in levels]
    random.shuffle(examples)

    schema_cache = {}          # db_id -> DDL string
    originals, controls, semantics = [], [], []
    intent_stats = defaultdict(lambda: {"kept": 0, "unchanged": 0, "changed": 0, "err": 0})
    scanned = 0

    for ex in examples:
        if scanned >= args.max_base:
            break
        db_id, question, gold = ex["db_id"], ex["question"], ex["query"]
        hardness = ex.get("hardness", "unknown")
        path = db_path(args.spider, db_id)
        if not os.path.exists(path):
            continue
        try:
            base_tree = sqlglot.parse_one(gold, read=DIALECT)
        except Exception:
            continue
        base_rows, base_err = run_query(path, gold)
        if base_err is not None:
            continue  # gold must run to serve as reference
        scanned += 1

        if db_id not in schema_cache:
            schema_cache[db_id] = get_schema(path)
        schema = schema_cache[db_id]
        ordered = has_top_order(base_tree)
        base_key = result_key(base_rows, ordered)

        # (a) the untouched original
        originals.append({
            "db_id": db_id, "question": question, "schema": schema, "hardness": hardness,
            "orig_sql": gold, "sql": gold, "family": "original", "rule": "none",
            "expected": "confirmed",
        })

        # (b) control mutations: keep only those verified UNCHANGED
        for rule, mut_sql, changed in apply_rules(ctrl_rules, base_tree, gold, path, base_key, ordered):
            st = intent_stats[("control", rule)]
            st["kept"] += 1
            st["changed" if changed else "unchanged"] += 1
            if not changed:
                controls.append({
                    "db_id": db_id, "question": question, "schema": schema, "hardness": hardness,
                    "orig_sql": gold, "sql": mut_sql, "family": "control", "rule": rule,
                    "expected": "confirmed",
                })

        # (c) semantic mutations: keep only those verified CHANGED
        for rule, mut_sql, changed in apply_rules(sem_rules, base_tree, gold, path, base_key, ordered):
            st = intent_stats[("semantic", rule)]
            st["kept"] += 1
            st["changed" if changed else "unchanged"] += 1
            if changed:
                semantics.append({
                    "db_id": db_id, "question": question, "schema": schema, "hardness": hardness,
                    "orig_sql": gold, "sql": mut_sql, "family": "semantic", "rule": rule,
                    "expected": "disputed",
                })

    # --- Fill quotas, spread for variety ---

    def take_spread(pool, k, keyfn):
        buckets = defaultdict(list)
        for c in pool:
            buckets[keyfn(c)].append(c)
        for b in buckets.values():
            random.shuffle(b)
        order = list(buckets.keys())
        out, i = [], 0
        while len(out) < k and any(buckets[o] for o in order):
            b = buckets[order[i % len(order)]]
            if b:
                out.append(b.pop())
            i += 1
        return out

    random.shuffle(originals)
    chosen_orig = take_spread(originals, args.n_original, lambda c: c["db_id"])
    chosen_ctrl = take_spread(controls, args.n_control, lambda c: c["rule"])
    chosen_sem = take_spread(semantics, args.n_semantic, lambda c: c["rule"])
    chosen = chosen_orig + chosen_ctrl + chosen_sem
    random.shuffle(chosen)

    # --- Emit dataset.jsonl ---

    out_path = os.path.join(args.out, "dataset.jsonl")
    with open(out_path, "w") as f:
        for i, c in enumerate(chosen):
            row = {
                "lemmaName": f'{c["db_id"]}__q{i:03d}__{c["family"]}__{c["rule"]}',
                "requirement": c["question"],
                "dafnyCode": c["sql"],      # the SQL query (the "formal artifact")
                "domain": c["db_id"],
                "schema": c["schema"],      # full DDL context for the informalizer
                "expected": c["expected"],
                "family": c["family"],
                "rule": c["rule"],
                "hardness": c["hardness"],  # official Spider difficulty (hard/extra)
                "orig_sql": c["orig_sql"],
            }
            f.write(json.dumps(row) + "\n")

    # --- Summary ---

    got = {"original": len(chosen_orig), "control": len(chosen_ctrl), "semantic": len(chosen_sem)}
    want = {"original": args.n_original, "control": args.n_control, "semantic": args.n_semantic}
    n_conf = sum(1 for c in chosen if c["expected"] == "confirmed")
    lines = []
    lines.append(f"Scanned base queries : {scanned}")
    lines.append(f"Pools available      : original={len(originals)} control={len(controls)} semantic={len(semantics)}")
    for fam in ("original", "control", "semantic"):
        flag = "" if got[fam] >= want[fam] else "  <-- SHORT"
        lines.append(f"  {fam:9}: {got[fam]}/{want[fam]}{flag}")
    lines.append(f"Rows written         : {len(chosen)}  (confirmed={n_conf}, disputed={len(chosen)-n_conf})")
    lines.append(f"Output               : {out_path}")
    lines.append("")
    lines.append("Rule behavior (how each mutation actually executed):")
    lines.append(f"  {'family/rule':32} {'kept':>5} {'unchg':>6} {'chg':>5} {'err':>5}")
    for (fam, rule), st in sorted(intent_stats.items()):
        lines.append(f"  {fam + '/' + rule:32} {st['kept']:>5} {st['unchanged']:>6} {st['changed']:>5} {st['err']:>5}")
    summary = "\n".join(lines)
    open(os.path.join(args.out, "summary.txt"), "w").write(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
