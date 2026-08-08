# Claim2Code API

Python 3.11 + FastAPI + Pydantic. Wires up the 9 subpackages from
`docs/specs/spec-summary.md §10` of the Claim2Code spec:

```
claim2code_api/
├── app/                  # FastAPI app, routers, main entrypoint
├── paper/                # PDF parser, sectionizer, claim extractor  (Phase 2)
├── repository/           # GitHub client, AST parser, symbol graph    (Phase 3)
├── retrieval/            # semantic + lexical + structural ranker     (Phase 4–5)
├── verification/         # verifier + discrepancy classifier           (Phase 6)
├── literature/           # scholarly API + structured comparison       (Phase 7)
├── workflow/             # explicit stateful executor                  (Phase 1+)
├── llm/                  # LLMClient interface + Pydantic schemas     (Phase 2+)
├── observability/        # OTel spans + metrics + logging              (Phase 9)
├── safety/               # secret redaction + prompt-injection defense (Phase 8)
└── db/                   # SQLAlchemy / pgvector config                (Phase 1+)
```

## Phase 0 status

Package skeleton only. Each subdirectory will get a `__init__.py` and real
modules starting in Phase 1.