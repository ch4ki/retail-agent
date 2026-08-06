# Retail Analysis Agent

A CLI chat agent that answers questions about the theLook e-commerce dataset in
BigQuery. It writes SQL, runs it behind a static safety guard, masks personal
data before the model ever sees it, and explains the results.

- **[Design document](docs/design.md)** — architecture, services, and how each
  requirement is handled, with what is built vs designed marked per section
- **[Example run](docs/example-run.md)** — an annotated transcript of a real
  session against live BigQuery

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker (for Postgres)
- A Google Cloud project you own — **theLook is a public dataset, but queries
  are billed to your project**
- An LLM API key (a free Gemini key from [AI Studio](https://aistudio.google.com/apikey) works)

## Setup

```bash
uv sync

cp .env.example .env
# Fill in GOOGLE_API_KEY

gcloud auth application-default login

docker compose up -d postgres
uv run retail-agent migrate
```

`GOOGLE_CLOUD_PROJECT` is optional. If you leave it blank the BigQuery client
uses the default project from your application-default credentials. Set it
explicitly when you have several projects and want to control which one is
billed.

## Run

```bash
uv run retail-agent chat
```

Try:

```
› what data do you have?
› who are our top 10 customers by spend?
› why does brand X outperform brand Y?
› save that as a report with action items for Q2
› delete all reports mentioning revenue
› /reports        list what you have saved
› /undo           reverse the last deletion
› /trace          explain the last turn: nodes, timings, every SQL attempt
› /metrics        first-pass SQL validity, self-correction, latency per node
› /persona list   change the agent's tone without a restart
› /prefs          your answer format, depth and table size
```

See [docs/example-run.md](docs/example-run.md) for an annotated transcript of a
real session, including the confirmation flow.

## Using a different LLM

Set `LLM_PROVIDER` in `.env` to `gemini`, `openai`, `openrouter` or `ollama`,
and supply the matching key. No code change is needed.

To pin a model, use the per-provider variable — `GEMINI_MODEL`, `OPENAI_MODEL`,
`OPENROUTER_MODEL`, `OLLAMA_MODEL`. A name pinned there is only ever sent to
that provider, so switching `LLM_PROVIDER` stays safe. The generic `LLM_MODEL`
applies to whichever provider is active, and is never sent to a fallback.

For downtime rather than preference, set a chain: `LLM_FALLBACKS=openai,ollama`.
Transient failures are retried on the current provider with jittered backoff, a
rejected key moves on immediately, and a provider that keeps failing is skipped
until a cooldown elapses.

## How it works

A LangGraph state machine owns each turn. The model decides *what* to ask; the
graph decides *what is allowed*. Every safety property is an edge in
`src/retail_agent/agent/graph.py`, not an instruction in a prompt.

```
start_turn → route ─┬─ schema      answered from cached metadata, no SQL
                    ├─ chat        follow-ups, answered from history
                    ├─ report_ops  save / list / stage a delete
                    │                └─ await_confirmation ─→ apply_delete
                    │                   (a breakpoint: nothing writes before you answer)
                    └─ plan → draft_sql → guard → dry_run → execute → mask
                                  ↑                │              → synthesize → egress
                                  ├──── repair ────┤  (budget: 3, held by the graph)
                                  └─── diagnose ───┘  (budget: 1, for empty results)
```

The two budgets are separate on purpose. An empty result is not a broken query —
sometimes "no orders matched" is the true answer — so diagnosing one must not
consume the retries that exist for SQL that genuinely failed.

## Viewing the pipeline in LangGraph Studio

```bash
uv run langgraph dev
```

Then open:

```
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

Studio renders the graph, lets you run a thread against the real BigQuery
connection, and shows the state after every node — including which SQL the
guard rejected and how the repair budget was spent.

The graph it loads is `src/retail_agent/agent/studio.py`, which builds the same
deps the CLI does. It passes no checkpointer, because the Studio server owns
thread persistence.

Studio does not replace the CLI: the confirmation flow for destructive actions
is a terminal interaction. Studio does show it paused, though — `pending_action`
in the state panel is the exact manifest awaiting your answer.

## Tracing (optional)

Set both `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in `.env` to send
per-node traces to [LangSmith](https://smith.langchain.com). You get a span per
graph node — `route`, `plan`, `draft_sql`, `execute`, `synthesize` — plus every
model call, which is enough to see exactly where a turn went wrong.

The banner tells you when it is on. Enabling it sends prompts and query results
off the machine; results are PII-masked before the model sees them, so what
leaves is masked, but it is still your data going to a third party.

Setting only one of the two variables leaves tracing off rather than warning on
every call.

## Safety

- **SQL guard** — every query is parsed to an AST before execution. Anything
  that is not a single read-only `SELECT` over the allowed tables is rejected,
  and a `LIMIT` is enforced. Covered by tests for DML, stacked statements,
  `EXPORT DATA`, dynamic SQL, and PII smuggled through subqueries.
- **PII masking** — result rows are masked the moment they return from
  BigQuery, *before* entering model context. The model never receives an email
  address, so no prompt can make it emit one. Rules live in
  `src/retail_agent/safety/policies/thelook.yaml`.
- **Egress scan** — the final answer is swept for anything resembling contact
  data. This is the second line of defence, not the first.
- **Cost ceiling** — every query is dry-run first and capped by
  `BQ_MAX_BYTES_BILLED` (2 GB by default).
- **Confirmed deletes** — removing saved reports shows the exact list first and
  requires typing the token it asks for (`y` for one, `DELETE <n>` for several).
  Deletes are soft, audited and reversible with `/undo`. Ownership is a SQL
  predicate on every statement, so it holds even if the model is compromised.
- **Bounded self-correction** — a failed query is retried at most twice. The
  counter lives in graph state, so the bound holds regardless of what the model
  decides. When it runs out the agent explains what it tried instead of looping.

## Semantic search over the analyst corpus (optional)

The agent answers from a corpus of analyst "trios" — question, SQL, report, and
the metric definitions that connect them. Matching a question to the right trio
is lexical by default, which needs nothing installed and no API key.

Set `DENSE_RETRIEVAL=true` to add a semantic ranker fused with it. None of
these questions share a distinctive word with the trio they find:

```
› how many shoppers have gone quiet?
  lexical: nothing        hybrid: churn-90
› which labels sell best?
  lexical: nothing        hybrid: brand-performance
› what is the capital of France?
  lexical: nothing        hybrid: nothing        ← the floor, doing its job
```

The index is [Milvus Lite](https://milvus.io/docs/milvus_lite.md) — embedded, a
single file, no server. Embeddings come from `text-embedding-3-small` when
`OPENAI_API_KEY` is set, and otherwise from a local ONNX model that needs no key
and sends nothing off the machine.

Those two are not equivalent, and the choice between them was measured rather
than assumed — run `scripts/calibrate_dense.py` to reproduce it:

| backend | right trio first | weakest true match | loudest nonsense |
|---|---|---|---|
| `text-embedding-3-small` | 4/5 | 0.296 | 0.102 |
| local ONNX | 2/5 | 0.138 | 0.222 |

The local model's ranges **overlap**: a question it should match scores lower
than nonsense it should reject, so no relevance floor is both sensitive and
precise. It is kept as the no-key fallback with the floor set to favour
precision, and `EMBEDDING_BACKEND=local` forces it. That number is a property of
a 45 MB model, not of the retrieval code — which is why the floor lives next to
the measurement that produced it, in `knowledge/dense.py`.

## Does it get the numbers right?

Every test below asserts a *path* — that a syntax error routes to repair, that
PII never reaches the warehouse. All of them pass while the agent returns a
confidently wrong number, which is the failure that actually reaches a user.

```bash
uv run retail-agent eval                  # all cases against live BigQuery
uv run retail-agent eval --case loyal-count
uv run retail-agent eval --json run.json  # then --baseline run.json next time
```

Each case pairs a question with a hand-written reference query. Both run, and
the agent's number is compared to the reference's. Ground truth is the query,
not a frozen number: theLook is appended to continuously — its newest order is
dated today — so a literal expected value would start rotting the day it was
written, and the suite would fail for reasons that have nothing to do with the
agent.

The exit code is the point: `0` ships, `1` does not. A PII leak fails the run
outright however high the accuracy, because the alternative is trading a
customer's email address against a percentage point.

The first live run scored **48.9%** — 23 correct, 15 wrong, 9 unanswered, no
PII leaks — and found a defect none of the 785 path-based tests could see: on
multi-step plans the agent inlines the previous step's truncated results into
the next query as literals, then answers from them. See §5.7 of the design doc.

## Tests

```bash
uv run pytest              # 785 tests, no credentials or database needed
uv run pytest -m db        # 72 tests, needs `docker compose up -d postgres`
uv run pytest -m live      # 7 tests, needs real BigQuery access and an LLM key
uv run pytest -m vector    # 14 tests, needs DENSE_RETRIEVAL deps; 9 need an OpenAI key
```

The safety modules are pure functions and are tested first, against an
adversarial corpus. Graph behaviour is tested with a scripted fake LLM and a
fake warehouse, asserting *paths* rather than output text — for example that an
exhausted repair budget degrades instead of looping, and that a query selecting
PII never reaches the warehouse at all.

## Troubleshooting

**"Could not connect to BigQuery"** — run `gcloud auth application-default
login`, and set `GOOGLE_CLOUD_PROJECT` if you have more than one project.

**"Could not reach the database"** — run `docker compose up -d postgres` then
`uv run retail-agent migrate`. It binds host port **5433**, not 5432, to avoid
clashing with a local install. The agent still runs without it; you lose
conversation history across restarts and saved reports, and it says so.

**Rate limits on the Gemini free tier** — switch `LLM_PROVIDER` to `openrouter`
or `ollama`, or set `LLM_FALLBACKS` so the agent moves on by itself.

## Status

Built: BigQuery access, SQL guard, PII masking, egress scan, the turn graph and
the CLI; the saved-reports library with its delete-confirmation gate, audit
trail and `/undo`; turn traces with `/trace`, `/trace <id>` and `/metrics`; and
the full resilience story — bounded self-correction, the `diagnose` edge for
empty results, and a provider fallback chain with a circuit breaker.

Also built: personas, so a non-developer can change the agent's tone without a
deploy — versioned, attributed, and provably unable to reach the safety rules —
and per-user answer preferences.

The agent also learns preferences from how you phrase questions, and proposes
them rather than applying them — it will ask before changing anything.

Also built: the Golden Bucket of analyst Trios — question, SQL, report and the
metric definitions that connect them — with hybrid lexical/dense retrieval, a
measured relevance floor, a clarifying question when a term is undefined that is
remembered per user, and promotion of an answered definition into the corpus.

Not yet built: the LLM judge for narrative quality, and system-level learning.
Both are designed in
[docs/design.md](docs/design.md), which marks each requirement Built, Partial or
Designed and names the command or test that demonstrates each Built claim.
