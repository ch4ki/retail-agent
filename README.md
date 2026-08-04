# Retail Analysis Agent

A CLI chat agent that answers questions about the theLook e-commerce dataset in
BigQuery. It writes SQL, runs it behind a static safety guard, masks personal
data before the model ever sees it, and explains the results.

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
```

## Using a different LLM

Set `LLM_PROVIDER` in `.env` to `gemini`, `openai`, `openrouter` or `ollama`,
and supply the matching key. No code change is needed.

To pin a model, use the per-provider variable — `GEMINI_MODEL`, `OPENAI_MODEL`,
`OPENROUTER_MODEL`, `OLLAMA_MODEL`. A name pinned there is only ever sent to
that provider, so switching `LLM_PROVIDER` stays safe. The generic `LLM_MODEL`
applies to whichever provider is active.

## How it works

A LangGraph state machine owns each turn. The model decides *what* to ask; the
graph decides *what is allowed*. Every safety property is an edge in
`src/retail_agent/agent/graph.py`, not an instruction in a prompt.

```
route ─┬─ schema      structural questions, answered from cached metadata
       ├─ chat        follow-ups, answered from conversation history
       └─ plan → draft_sql → guard → dry_run → execute → mask → synthesize → egress
                     ↑                  │
                     └──── repair ──────┘   (budget: 2, held by the graph)
```

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
(phase 2) is a terminal interaction.

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
- **Bounded self-correction** — a failed query is retried at most twice. The
  counter lives in graph state, so the bound holds regardless of what the model
  decides. When it runs out the agent explains what it tried instead of looping.

## Tests

```bash
uv run pytest              # 171 tests, no credentials needed
uv run pytest -m live      # 4 tests, needs real BigQuery access
```

The safety modules are pure functions and are tested first, against an
adversarial corpus. Graph behaviour is tested with a scripted fake LLM and a
fake warehouse, asserting *paths* rather than output text — for example that an
exhausted repair budget degrades instead of looping, and that a query selecting
PII never reaches the warehouse at all.

## Troubleshooting

**"Could not connect to BigQuery"** — run `gcloud auth application-default
login`, and set `GOOGLE_CLOUD_PROJECT` if you have more than one project.

**"Could not reach the database"** — run `docker compose up -d postgres`. It
binds host port **5433**, not 5432, to avoid clashing with a local install.
The agent still runs without it; you just lose conversation history.

**Rate limits on the Gemini free tier** — switch `LLM_PROVIDER` to `openrouter`
or `ollama`.

## Status

Phase 1 of 4. Built: BigQuery access, SQL guard, PII masking, egress scan,
bounded self-correction, the turn graph, and the CLI.

LangSmith tracing is wired up and verified. Not yet built: the local
`traces` table and `/trace` replay, saved reports with the delete-confirmation
gate, personas and user preferences (phase 2); the Golden
Bucket of analyst Trios (phase 3); the eval suite (phase 4). Design for all of
these is in `docs/superpowers/specs/`.
