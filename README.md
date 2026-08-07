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
  `BQ_MAX_BYTES_BILLED` (2 GB by default). Note that a SQL `LIMIT` is *not* a
  cost control: BigQuery bills bytes scanned, and adding `LIMIT 500` to a real
  query here was measured to save 0%. The guard still injects a `LIMIT` as a
  ceiling against an unbounded result, but how many rows are shown is decided
  when the result is read — so `row_count` is the true size of the result rather
  than the cap, and a question like "how many customers are loyal" has a correct
  answer even when the agent returns rows instead of a `COUNT`.
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

Vectors are stored with [pgvector](https://github.com/pgvector/pgvector) in the
same Postgres that already holds the trios — one store to run, migrate and back
up rather than two, and a trio and its embedding are written in the same
transaction so they cannot drift apart. Embeddings come from
`text-embedding-3-small`, so this needs `OPENAI_API_KEY` as well as the
database. Without either, retrieval falls back to lexical.

The relevance floor was measured, not guessed. Scoring paraphrased questions
against the trio each should find, and unrelated questions against the whole
corpus:

```
text-embedding-3-small   top-1 4/5   weakest true match 0.296   loudest nonsense 0.102
0.194 of daylight between them, so the floor sits at 0.20
```

That gap is the whole argument for the floor: a vector index always returns its
nearest neighbour however far away it is, and a wrong trio is worse than no trio
because it supplies a confident wrong definition the agent cannot tell is wrong.
An earlier local ONNX model was dropped for failing exactly this test — its
scores for relevant questions ran as low as 0.138 while nonsense reached 0.222,
so no floor could be both sensitive and precise.

## Tests

```bash
uv run pytest              # 836 tests, no credentials or database needed
uv run pytest -m db        # 94 tests, needs `docker compose up -d postgres`
uv run pytest -m live      # 9 tests, needs real BigQuery access and an LLM key
uv run pytest -m vector    # 14 tests, needs DENSE_RETRIEVAL deps; 9 need an OpenAI key
```

The safety modules are pure functions and are tested first, against an
adversarial corpus. Graph behaviour is tested with a scripted fake LLM and a
fake warehouse, asserting *paths* rather than output text — for example that an
exhausted repair budget degrades instead of looping, and that a query which
would disclose PII is rejected before execution. Selecting a PII column *bare*
is deliberately allowed — an unaliased column keeps its name, which is what lets
the masking policy find it on the way back. What the guard rejects is anything
that would defeat that: an alias, or the column buried inside an expression.

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
them rather than applying them — it will ask before changing anything. It reads
the phrasing with the model, in the router call it already makes, and refuses to
quote you on anything you did not literally type.

Also built: the Golden Bucket of analyst Trios — question, SQL, report and the
metric definitions that connect them — with hybrid lexical/dense retrieval, a
measured relevance floor, a clarifying question when a term is undefined that is
remembered per user, and promotion of an answered definition into the corpus.

Not yet built: the LLM judge for narrative quality, and system-level learning.
Both are designed in
[docs/design.md](docs/design.md), which marks each requirement Built, Partial or
Designed and names the command or test that demonstrates each Built claim.
