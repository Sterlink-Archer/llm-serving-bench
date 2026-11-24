# results/

- **[CLAIMS.md](CLAIMS.md) / [claims.json](claims.json)**: every claim (group × column), the rule that made it, and what blocks it.
- **[PROTOCOL_TABLE.md](PROTOCOL_TABLE.md) / [protocol_rows.json](protocol_rows.json)**: every protocol cell (rate × seed).
- **[COLUMN_C.md](COLUMN_C.md)**: cached traffic (supplementary). **[QUALITY_TABLE.md](QUALITY_TABLE.md)**: quality vs the BF16 original.
- **[COST_TABLE.md](COST_TABLE.md)**, **[HEADLINE.md](HEADLINE.md)**, **[RUNS.md](RUNS.md)**: costs, per-config headlines, one line per run.
- **[experiments.jsonl](experiments.jsonl)**: the journal's records ([docs/JOURNAL.md](../docs/JOURNAL.md) renders them).
- **`<label>/<UTC stamp>/`**: one directory per run: `config.json`, `run.json`, `protocol_*/` (summary + cells), gate/eval JSONs.
- Raw artifacts (server logs, GPU monitor CSVs, profiler traces) sit beside each run's JSON, kept for anyone who wants to
  check a number at its source.

All tables here are generated (`python3 -I analysis/report.py`); edit the code, not the tables.
