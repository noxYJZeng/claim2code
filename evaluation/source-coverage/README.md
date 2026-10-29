# Source-index coverage

`pytorch-source-index.json` records an actual source-only preparation run of
commit `4ec860a`. All 15 pinned repositories were parsed into an isolated,
migrated PostgreSQL/pgvector database: 12 use PyTorch, with 27,981 total code
symbols and all 178 annotated symbol references resolved. Method targets
include their parent-class qualification. Annotated source quotations were
checked at their exact lines, and PyTorch framework evidence was checked
against real imports in the pinned code.

The run made no embedding or chat-model calls; all embedding counts are zero.
It establishes source/index coverage, not retrieval or verification accuracy.
The stored source digest identifies the measured code; absolute clone paths
and temporary database row identifiers were removed from this public copy.

To repeat in a separate migrated database, set `DATABASE_URL` and run:

```bash
python evaluation/prepare_benchmark.py --parse-only --output /tmp/source-coverage.json
```

For model measurements and resume-target checks, use
`python evaluation/run_local.py --prepare` as described in the evaluation guide.
