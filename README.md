# Specification-faithfulness benchmark and evaluation harness

Data, construction scripts and evaluation code for the paper
*claimcheck: Measuring and Improving Specification Faithfulness*.

Every row pairs a natural-language requirement with a formal artifact (a Dafny
contract or a SQL query) and a label saying whether the artifact means what
the requirement asks for: `confirmed` or `disputed`. The harness scores a
single-call baseline and the two-step `claimcheck` pipeline on those rows.

## Layout

```
bench/data/clover_v2.jsonl   Dafny slice, 362 rows (Table 2 of the paper)
bench/data/spider_v4.jsonl   SQL slice, 1,458 rows (Table 1)
bench/data/spider_v3.jsonl   1,485-row superset the SQL runs were made on; see below
bench/run_baseline.js        the single-call baseline
bench/run_claimcheck.js      informalize, then compare
bench/lib/                   dataset loading, chunking, model calls, scoring
src/prompts.js               every prompt the harness sends, verbatim
src/schemas.js               the tool schemas the models answer through
construction/                the scripts that built the two slices
```

## Row schema

| field | meaning |
|---|---|
| `id`, `name` | row identifier; mutations carry the original's name plus a suffix |
| `informal` | the natural-language requirement |
| `formal` | the artifact: a Dafny contract with the body stripped, or a SQL query |
| `context` | what is needed to read the artifact; the database DDL for SQL, empty for Dafny |
| `expected` | `confirmed` or `disputed` |
| `variant` | `original`, `control_mutation`, `semantic_mutation`, or `informal_mutation` |
| `mutation_rule` | the rewrite rule that produced the row, `null` for originals |
| `mutation_of` | the id of the original this row descends from |
| `clover_class`, `clover_category` | Dafny only: CloverBench's variant class and its hand annotation |

### Dafny slice (`clover_v2.jsonl`, 362 rows)

Built from CloverBench's 62 textbook programs and its released variant
classes. Originals are `confirmed`; `C1` (docstring mutated, 62 rows), `C2`
(contract mutated, 60) and `C6` (contract and body mutated, 60) are
`disputed`; 118 control mutations rewrite the contract while preserving its
meaning and stay `confirmed`. Every control carries an equivalence lemma that
Dafny discharged before the row was kept.

### SQL slice (`spider_v4.jsonl`, 1,458 rows)

Built from the Spider 1.0 development questions rated `hard` or `extra` by
Spider's own evaluator: 340 originals, 488 controls and 630 semantic
mutations over 20 databases. Labels come from executing both queries on the
Spider SQLite databases: a control is kept only if its result set equals the
gold query's, a semantic mutation only if it differs.

`spider_v3.jsonl` is the 1,485-row set the API runs were made on, before 27
`where_to_on_inner` controls were cut. `run_claimcheck.js` calls the API on
v3 and scores on the v4 id set, so nothing needed re-running.

## Running the evaluation

Requires Node 20 or later and an Anthropic API key.

```
npm install
export ANTHROPIC_API_KEY=...

node bench/run_baseline.js   --dataset clover
node bench/run_baseline.js   --dataset spider --chunk domain
node bench/run_claimcheck.js --dataset clover
node bench/run_claimcheck.js --dataset spider --chunk domain
```

Results land in `bench/results/<dataset>_<system>.json` with per-row verdicts,
the two class recalls, macro recall, per-variant recall and token usage.
Runs checkpoint per chunk; add `--resume` to continue an interrupted run.

Settings match the paper's Experimental Setup: informalizer
`claude-haiku-4-5-20251001`, judge `claude-sonnet-4-6`, 8192 max tokens,
temperature 0, tool-call output. Dafny is chunked at ten rows per call and
SQL at one call per database. Rows descending from the same original never
share a call.

The paper's main table restricts both systems to rows on which both returned
a usable verdict. Run standalone, each script reports over its own
denominator; on these datasets the difference is at most 0.1 points of macro
recall.

The model sweep in the paper called other vendors through OpenRouter. That
code path is not included here; `bench/lib/models.js` talks to the Anthropic
API only, which is all the main result needs.

## Rebuilding the benchmark

The shipped JSONL files are the benchmark. The scripts below regenerate them
from the upstream datasets and are included so the construction can be
audited. Rebuilding the SQL slice needs `sqlglot` and the Spider download
(`dev.json`, `tables.json`, `database/`, and Spider's `evaluation.py` and
`process_sql.py` placed in `construction/spider_eval/`). Rebuilding the Dafny
controls needs a Dafny 4 binary.

```
# SQL: annotate difficulty, mutate and execution-label, then assemble
python3 construction/spider_hardness.py --spider <spider dir>
python3 construction/spider-mutate.py --spider <spider dir> --out construction/spider-bench-full \
        --n-original 340 --n-control 100000 --n-semantic 100000   # large n keeps every candidate
python3 construction/build_spider.py

# Dafny: assemble the Clover variants, prove the controls, assemble again with them
python3 construction/build_clover.py --clover-dir <CloverBench dir>
python3 construction/clover-mutate.py --dafny <path to dafny>
python3 construction/build_clover.py --clover-dir <CloverBench dir> \
        --controls construction/benchmark/clover-control.jsonl
```

The sampling flags used for the shipped files were not recorded alongside
them, so a rebuild reproduces the construction rather than the exact row set;
the shipped JSONL is the reference. `spider-mutate.py` writes a `summary.txt` giving, per rule, how many
candidates its intent predicted and how many execution actually confirmed.
`clover-mutate.py` writes each equivalence lemma to a work directory; pass
`--keep-work` to keep the `.dfy` files for inspection.

## License

MIT, see `LICENSE`. Spider and CloverBench remain under their own licenses.
