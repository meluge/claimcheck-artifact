# Results

The per-run outputs behind every Dafny and SQL number in the paper: the main
result, both model sweeps, the strongest pairing, and the cost columns. The
Lean slice's runs are not included here.

Each run is a pair of files:

- `<run>.json`: the run's configuration, summary metrics, and a per-row verdict
  in `perRow` (`id`, `variant`, `expectedMatch`, `predictedMatch`). A
  `predictedMatch` of `null` means the system returned no usable verdict for
  that row.
- `<run>.json.progress`: the per-call checkpoint the run wrote as it went:
  verdicts, latency, and `cost`, the billed cost in USD as reported by
  OpenRouter. `cost` is `null` for calls made through the Anthropic API, which
  does not report it.

## File names

`<dataset>_<step>_<model>_n<rows>[_seed42_c<chunk>][_<judge>].json`

- `clover2` is `bench/data/clover_v2.jsonl` (Dafny, 362 rows). `spider2` is
  `bench/data/spider_v3.jsonl` (SQL, 1,485 rows); the paper scores the
  1,458-row `spider_v4.jsonl` subset of each SQL run by id.
- `bare` is the baseline, one row per call. `compare` is `claimcheck` with the
  informalizer fixed at `claude-haiku-4.5` and the named model as judge.
  `informalize` is `claimcheck` with the named model as informalizer and
  `gemini-3.7-flash` as judge.
- `c10` is ten rows per call; `cdomain` is one call per database.

## Which runs back which table

| Paper table | Runs |
|---|---|
| Main result, Dafny and SQL rows | `clover2_bare_anthropic_claude-sonnet-4.6_n362`, `clover2_compare_anthropic_claude-sonnet-4.6_n362_seed42_c10`, `spider2_bare_anthropic_claude-sonnet-4.6_n1485`, `spider2_compare_anthropic_claude-sonnet-4.6_n1485_seed42_cdomain` |
| Judge sweep, Dafny | every `clover2_bare_*` with the matching `clover2_compare_*` |
| Judge sweep, SQL | every `spider2_bare_*` with the matching `spider2_compare_*` |
| Informalizer sweep, Dafny | every `clover2_informalize_*` |
| Informalizer sweep, SQL | every `spider2_informalize_*` |
| Strongest pairing | `*_bare_google_gemini-3.7-flash_*` with `*_informalize_google_gemini-3.7-flash_*` |
| Billed costs | the `cost` fields in the `.progress` files |
| Estimated costs (main result) | `cost_estimate/` |

In the paper, a row counts toward a baseline-versus-`claimcheck` comparison
only if both systems returned a usable verdict on it, and SQL is scored on the
`spider_v4` ids. Intervals are 95% Wilson intervals.

## Estimated costs

The main-result configuration ran through the Anthropic API, which reports no
billed cost, so its cost is estimated at list price:

- `cost_estimate/cost_estimate_counts.json`: input tokens for every prompt
  those runs sent, rebuilt from the same data, sampling, chunking and prompt
  code and counted with Anthropic's token-counting endpoint; and the
  informalizer's output tokens, counted from its saved back-translations.
- `cost_estimate/cost_estimate_sample.json`: 38 calls re-issued with the same
  settings, with the usage the API billed. They give the judge's and the
  baseline's output length, which the original runs did not save. On every one
  of them the counted input matched the billed input exactly.

Prices used (USD per million input / output tokens, list price on
2026-09-10): `claude-haiku-4.5` $1 / $5, `claude-sonnet-4.6` $3 / $15.

## Provenance

Each run records the commit of the `claimcheck` code that produced it. Those
fields read `withheld-for-review` in this anonymous release.
