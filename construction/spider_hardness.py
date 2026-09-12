#!/usr/bin/env python3
"""
spider_hardness.py — Annotate every Spider dev example with its OFFICIAL
difficulty label (easy / medium / hard / extra), using Spider's own
`eval_hardness` from evaluation.py (no reimplementation).

Reads:  <spider>/dev.json, <spider>/tables.json
Writes: construction/spider-bench/dev_hardness.json
          [ {db_id, question, query, hardness}, ... ]  aligned with dev.json order
        (rows whose SQL fails Spider's parser get hardness="unparsable")

Usage:
  python3 construction/spider_hardness.py --spider <spider_data dir>
"""

import argparse
import json
import os
import sys
from collections import Counter

# Spider's official scripts live next to this file.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "spider_eval"))
from process_sql import Schema, get_sql       # noqa: E402
from evaluation import Evaluator              # noqa: E402


def build_schemas(tables_json):
    """tables.json -> { db_id: {table_lower: [col_lower, ...]} } for Schema()."""
    schemas = {}
    for entry in json.load(open(tables_json)):
        tables = entry["table_names_original"]
        schema = {t.lower(): [] for t in tables}
        for tidx, col in entry["column_names_original"]:
            if tidx >= 0:
                schema[tables[tidx].lower()].append(col.lower())
        schemas[entry["db_id"]] = schema
    return schemas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spider", required=True)
    ap.add_argument("--split", default="dev.json")
    ap.add_argument("--out", default="construction/spider-bench/dev_hardness.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    schemas = build_schemas(os.path.join(args.spider, "tables.json"))
    examples = json.load(open(os.path.join(args.spider, args.split)))
    evaluator = Evaluator()

    out, hist = [], Counter()
    for ex in examples:
        db_id, question, query = ex["db_id"], ex["question"], ex["query"]
        try:
            schema = Schema(schemas[db_id])
            parsed = get_sql(schema, query)
            hardness = evaluator.eval_hardness(parsed)
        except Exception:
            hardness = "unparsable"
        hist[hardness] += 1
        out.append({"db_id": db_id, "question": question, "query": query, "hardness": hardness})

    json.dump(out, open(args.out, "w"), indent=0)
    print(f"Annotated {len(out)} dev examples -> {args.out}")
    for level in ("easy", "medium", "hard", "extra", "unparsable"):
        print(f"  {level:11}: {hist.get(level, 0)}")


if __name__ == "__main__":
    main()
