#!/usr/bin/env python3
"""Assemble the Spider/SQL faithfulness benchmark as a standalone dataset.

Emits the same schema as build_clover.py, so the two can be run by the same
harness and reported side by side.

Composition (all 1284 rows by default):

  original   340   confirmed   the upstream Spider question and its gold query
  control    418   confirmed   MEANING-PRESERVING rewrites of the gold query
  semantic   526   disputed    meaning-changing rewrites

The control family is what makes this benchmark able to separate a judge that
reads from one that pattern-matches. Everywhere else in this project
`expected == confirmed` holds exactly when the row is unmutated, so answering
"unmutated, therefore faithful" scores perfectly on the confirmed half. Here 418
of the 758 confirmed rows ARE mutated, so that strategy fails on more than half
of them.

Labels are execution-verified upstream: a control was kept only if the mutated
query returns the same result set as the gold query on the real database, and a
semantic mutation only if it returns a different one. That makes this the most
trustworthy label source in the project -- MOB and PutnamBench originals are
human-written and taken on faith, and generated mutations are true by
construction rather than checked.

By default every row is emitted. `--subsample` reproduces the 890-row
composition used when this was one slice of the combined benchmark (340 + 200
controls + 350 semantic), which exists only for continuity with older runs.

Usage:
    python3 construction/build_spider.py
    python3 construction/build_spider.py --subsample --seed 0 --csv
"""

import argparse
import csv
import json
import random
from pathlib import Path

import sqlglot
from sqlglot import expressions as exp

HERE = Path(__file__).resolve().parent
SPIDER = HERE / "spider-bench-full" / "dataset.jsonl"

SUBSAMPLE_TARGET = {"control": 200, "semantic": 350}

VARIANT = {"original": "original", "control": "control_mutation",
           "semantic": "semantic_mutation"}


def jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf8").splitlines() if l.strip()]


# --------------------------------------------------------------- well-formedness

def _from_clause(select):
    """sqlglot renamed this arg; accept either spelling."""
    return select.args.get("from_") or select.args.get("from")


def has_forward_reference(sql):
    """True if a JOIN's ON clause references a table introduced by a LATER join.

    SQLite resolves the whole FROM/JOIN chain as one scope, so such a query runs
    and returns the expected rows -- which is exactly why the execution check let
    these through. Postgres, MySQL and SQL Server all reject them, and a judge
    that calls them broken is right, so they cannot serve as `confirmed` rows.

    This bit the `where_to_on_inner` rule: it moved a WHERE predicate into the
    FIRST inner join, but in Spider the filter usually targets the last-joined
    table. Rows that trip this check are dropped rather than shipped.
    """
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return False
    lower = lambda v: (v or "").lower()
    for select in tree.find_all(exp.Select):
        scope = set()
        frm = _from_clause(select)
        if frm:
            scope |= {lower(t.alias_or_name) for t in frm.find_all(exp.Table)}
        for join in select.args.get("joins") or []:
            joined = {lower(t.alias_or_name) for t in join.find_all(exp.Table)}
            on = join.args.get("on")
            if on is not None:
                refs = {lower(c.table) for c in on.find_all(exp.Column) if c.table}
                if refs - scope - joined:
                    return True
            scope |= joined
    return False


def row(**kw):
    base = dict(id=None, name=None, informal="", formal="", context="", source=None,
                language=None, expected=None, variant="original",
                mutation_rule=None, mutation_of=None, category=None, meta_orig=None,
                clover_class=None, clover_category=None)
    base.update(kw)
    return base


def build(recs, rng, subsample, dropped=None):
    groups = {"original": [], "control": [], "semantic": []}
    for r in recs:
        groups[r["family"]].append(r)

    picked = list(groups["original"])
    for fam in ("control", "semantic"):
        pool = list(groups[fam])
        if subsample:
            rng.shuffle(pool)
            pool = pool[:SUBSAMPLE_TARGET[fam]]
        picked += pool

    # The q-index in a row's name is a position in the upstream file, not a
    # problem id, so a mutation's name cannot be turned into its original's.
    # (domain, requirement) does identify a unique original -- verified: no key
    # holds two originals, all 944 mutations resolve, and each one's orig_sql
    # equals the resolved original's query.
    orig_by_question = {}
    for r in recs:
        if r["family"] == "original":
            orig_by_question[(r["domain"], r["requirement"].strip())] = r["lemmaName"]

    rows = []
    for r in picked:
        fam = r["family"]
        name = r["lemmaName"]
        if has_forward_reference(r["dafnyCode"]):
            if dropped is not None:
                dropped.append((name, r.get("rule")))
            continue
        orig_name = orig_by_question.get((r["domain"], r["requirement"].strip()))
        rows.append(row(
            id=f"spider/sql/{name}", name=name,
            informal=r["requirement"],
            # `dafnyCode` is the harness's generic field name for "the formal
            # artifact"; on these rows it holds the SQL query.
            formal=r["dafnyCode"],
            # The DDL is not optional context here: without the schema, column
            # and alias references in the query cannot be resolved, so a judge
            # cannot say what the query computes.
            context=r.get("schema", ""),
            source="spider", language="sql",
            expected=r["expected"], variant=VARIANT[fam],
            mutation_rule=None if fam == "original" else r.get("rule"),
            mutation_of=None if fam == "original" or orig_name is None else f"spider/sql/{orig_name}",
            category=r.get("domain"),
            # only a mutation has a previous version; Spider ships orig_sql on
            # originals too, where it means something else.
            meta_orig=None if fam == "original" else r.get("orig_sql"),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", type=Path, nargs="+", default=[SPIDER],
                    help="one or more spider-mutate outputs to read and merge. Merging at "
                         "this level rather than concatenating built files matters: "
                         "mutation_of is resolved against the originals in the same batch, "
                         "so a rules-only run has to be combined with the run that holds "
                         "the originals or its rows lose that link.")
    ap.add_argument("--out", type=Path, default=HERE / "benchmark" / "spider.jsonl")
    ap.add_argument("--subsample", action="store_true",
                    help="emit the older 890-row composition instead of all 1284")
    ap.add_argument("--csv", action="store_true", help="also write a .csv view alongside")
    args = ap.parse_args()

    recs = []
    for d in args.data:
        if not d.exists():
            raise SystemExit(f"Spider dataset not found at {d}")
        recs += jsonl(d)
    dropped = []
    rows = build(recs, random.Random(args.seed), args.subsample, dropped)

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
    if dropped:
        by_rule = {}
        for _n, rule in dropped:
            by_rule[rule] = by_rule.get(rule, 0) + 1
        print(f"dropped {len(dropped)} malformed rows (ON clause references a "
              f"later-joined table):")
        for rule, n in sorted(by_rule.items()):
            print(f"  {rule:<24}{n:>4}")
    if args.csv:
        print(f"      and a csv view -> {args.out.with_suffix('.csv')}")
    print()
    print(f"{'variant':<20}{'rows':>6}{'confirmed':>11}{'disputed':>10}")
    for v in ("original", "control_mutation", "semantic_mutation"):
        sub = [r for r in rows if r["variant"] == v]
        if not sub:
            continue
        c = sum(1 for r in sub if r["expected"] == "confirmed")
        print(f"{v:<20}{len(sub):>6}{c:>11}{len(sub) - c:>10}")
    c = sum(1 for r in rows if r["expected"] == "confirmed")
    print(f"{'TOTAL':<20}{len(rows):>6}{c:>11}{len(rows) - c:>10}")

    mutated_confirmed = sum(1 for r in rows
                            if r["expected"] == "confirmed" and r["variant"] != "original")
    print(f"\nmutated but confirmed (the control slice): {mutated_confirmed}"
          f" of {c} confirmed rows")

    by = {}
    for r in rows:
        if r["mutation_rule"] and r["mutation_rule"] != "none":
            by[(r["variant"], r["mutation_rule"])] = by.get((r["variant"], r["mutation_rule"]), 0) + 1
    print("\nmutations by rule:")
    for k in sorted(by):
        print(f"  {k[0]:<20}{k[1]:<24}{by[k]:>4}")


if __name__ == "__main__":
    main()
