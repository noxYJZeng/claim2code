# Claim2Code

Check whether a paper's claims match its code. Give Claim2Code a paper PDF and
the GitHub repository that implements it; for each claim it finds the code
that implements it and returns `VERIFIED`, `PARTIALLY_VERIFIED`,
`CONTRADICTED` or `UNVERIFIED`, with the paper passage and the code lines
it relied on, plus any mismatches (hyperparameters, objectives, datasets,
metrics).

![Verification of a LoRA claim: partially verified, with the discrepancy explained](docs/screenshots/verification.jpg)

| The code it cited | A reference test of that code |
|---|---|
| ![Cited code: lora_A is initialised with kaiming_uniform_, not a Gaussian as the paper says](docs/screenshots/code-evidence.jpg) | ![Reference test: all 7 inputs gave the same output from Linear.forward and from a reference written from the paper](docs/screenshots/reference-test.jpg) |

Above: [microsoft/LoRA](https://github.com/microsoft/LoRA) checked with a local
27B model. The forward pass matches the paper, and a reference test confirms
it on 7 inputs. The verdict is still partial: the paper initialises A from a
Gaussian, while the code uses `kaiming_uniform_`, and a comment in the code
says so.

**No model or API key is included.** Claim2Code calls a chat model and an
embedding model that you configure: a local OpenAI-compatible server
(LM Studio, Ollama) or a hosted provider with your own key.

## Requirements

- Python 3.11+
- PostgreSQL 16 with pgvector
- Node 20+ (for the web UI)
- A chat model and an embedding model behind an OpenAI-compatible endpoint
- Docker + Compose v2 (optional, for `make dev`)

## Setup

```bash
git clone https://github.com/noxYJZeng/claim2code.git
cd claim2code
cp .env.example .env
```

Edit `.env`:

```bash
DATABASE_URL=postgresql+psycopg://claim2code:change_me_local_dev@localhost:5432/claim2code

OPENAI_BASE_URL=http://localhost:1234/v1      # your endpoint
OPENAI_API_KEY=...                            # your key (any value for a local server)
OPENAI_MODEL_CHAT=<chat model id>
OPENAI_MODEL_EMBED=<embedding model id>
OPENAI_EMBED_DIM=768                          # must match the embedding model
```

Changing the embedding model or dimension later requires a fresh database.

### With Docker

```bash
make dev        # postgres + pgvector, API on :8000, web on :5173
```

### Without Docker

```bash
# database
createdb claim2code
psql claim2code -c "CREATE EXTENSION IF NOT EXISTS vector;"

# backend
python3.11 -m venv .venv
.venv/bin/pip install -e "apps/api[dev]"
cd apps/api && ../../.venv/bin/alembic upgrade head
../../.venv/bin/uvicorn claim2code_api.app.main:app --port 8000

# frontend (second terminal)
cd apps/web && npm install && npm run dev
```

## Usage

Open http://localhost:5173:

1. Upload a paper PDF; its claims are extracted.
2. Enter the paper's GitHub repository URL; it is cloned and indexed.
3. In **Verification**, verify one claim or all of them.
4. **Show details** on a verdict shows the cited paper passage and the cited
   code, with a GitHub link to those lines.

The **Chat model** panel switches the chat model (LM Studio, Ollama,
Anthropic, OpenAI, Gemini, DeepSeek, OpenRouter, or any OpenAI-compatible
server). Changing it from the UI requires `LLM_MANAGEMENT_ALLOW_LOCAL=true`
or `LLM_MANAGEMENT_TOKEN` in `.env`.

The same steps through the API (docs at http://localhost:8000/docs):

```bash
curl -X POST http://localhost:8000/papers/ -F "file=@paper.pdf"            # → analysis_id
curl -X POST http://localhost:8000/repositories/ \
  -H "Content-Type: application/json" -d '{"url": "https://github.com/owner/repo"}'  # → repository_id
curl -X POST http://localhost:8000/analyses/<analysis_id>/verifications \
  -H "Content-Type: application/json" -d '{"repository_id": "<repository_id>"}'
```

### Reference tests

A verdict is the model reading the code. A **reference test** runs it: for
one claim, the model writes test inputs for the cited function (with its
code in view) and a separate reference implementation from the paper
passage and the function's signature only. Both run on the same inputs and
the outputs are compared: same output, different output, could not run, or
not testable. Open a claim in **Verification** and use **Run reference test**.

A claim that two computations agree, such as "the low-rank update can be
merged into the weights with no change in output", is tested as a
**property** instead: the model writes a `check()` that runs both paths
through the repository's own code (before and after merging), and the two
results are compared. No reference implementation is involved.

This executes repository code and model-written code on your machine, so it
is off by default. To turn it on, set in `.env`:

```bash
REFERENCE_CHECK_ENABLED=true
REFERENCE_CHECK_PYTHON=/path/to/python   # an interpreter with the repo's dependencies (torch, ...)
REFERENCE_CHECK_TIMEOUT_SECONDS=60
```

Packages the module imports but the interpreter lacks are replaced by
stubs, so a loss in a file that also imports `wandb` or `transformers` can
still be tested; on a machine without CUDA, `.cuda()` and `device="cuda"`
run on the CPU. The result lists both, since they can change behaviour.

The run gets an empty environment (no keys reach it), CPU and file-size
limits and a timeout; on macOS it also has no network and can write only to
a temporary directory, and on Linux it has no network when `unshare` is
available. This limits accidents; it is not a boundary against hostile
code, so only enable it for repositories you would run anyway.

### Demo without a model

`npm run dev` in `apps/web`, then open http://localhost:5173/?demo=1 for a
read-only sample analysis. It needs no API, database or model.

## Tests and evaluation

```bash
make test       # backend + frontend unit tests
make lint
make eval-all   # retrieval, verification and safety benchmarks (needs a model)
```

Benchmark data, methods and recorded results are in
[`evaluation/`](evaluation/README.md).

## License

MIT — see [LICENSE](LICENSE).
