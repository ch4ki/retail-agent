# Retail Analysis Agent

A CLI chat agent that answers questions about the theLook e-commerce dataset in
BigQuery. It writes SQL, runs it behind a static safety guard, masks personal
data before the model ever sees it, and explains the results.

- **[Documentation](docs_to_submit/README.md)** — architecture, data flow, how
  each requirement is handled, and an annotated example run

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

The same agent also runs behind the LangGraph server —
`uv run langgraph dev` — see
[docs/04-setup-and-run.md](docs/04-setup-and-run.md#running-as-a-server).

Try:

```
› what data do you have?
› who are our top 10 customers by spend?
› why does brand X outperform brand Y?
› save that as a report with action items for Q2
› delete all reports mentioning revenue
› /reports        list what you have saved
› /undo           reverse the last deletion
› /trace          explain the last turn: every tool call, timing, SQL attempt,
                  the definitions used and any term the agent settled itself
› /metrics        first-pass SQL validity, self-correction, latency per step
› /persona list   change the agent's tone without a restart
› /prefs          your answer format, depth and table size
› /definitions    what you have told it terms mean; `forget <term>` to re-ask
› /trios          the analyst corpus it answers definitions from
› /help           all of them
```

## Using a different LLM

Set `LLM_PROVIDER` in `.env` to `gemini`, `openai`, `openrouter` or `ollama`,
and supply the matching key. No code change is needed.

To pin a model, use the per-provider variable — `GEMINI_MODEL`, `OPENAI_MODEL`,
`OPENROUTER_MODEL`, `OLLAMA_MODEL`. A name pinned there is only ever sent to
that provider, so switching `LLM_PROVIDER` stays safe. The generic `LLM_MODEL`
applies to whichever provider is active, and is never sent to a fallback.

For downtime rather than preference, set a chain: `LLM_FALLBACKS=openai,ollama`.
Transient failures (429s, timeouts, 5xx) are retried on the current provider
with jittered backoff, then the next provider in the chain takes over. A
rejected key is not retried at all: it is a permanent failure, and retrying it
only adds latency before the identical rejection.

There is no circuit breaker. `ModelFallbackMiddleware` always tries the
configured provider first, so during an outage every turn pays that provider's
retry budget before falling through.

## How it works

One ReAct supervisor with ten tools, three of which are subagents. The model
decides *what* to ask; middleware and tool preconditions decide *what is
allowed*. No safety property is an instruction in a prompt.

```
your question
  └─ supervisor           persona + your preferences, read per model call
     ├─ analyst           ── a subagent with its own loop ──────────────┐
     │                       resolves what terms mean first;            │
     │                       returns without querying if one is unsettled│
     │                       run_sql → guard → dry_run → execute → mask │
     │                          ↑                    │                  │
     │                          └──── error back to the model ──────────┘
     │                               (budget: 14 queries per turn)
     ├─ report_writer      a subagent with no data tools, so it cannot
     │                     invent a figure the analyst did not find.
     │                     Saves what it wrote, scanned for PII first,
     │                     and returns only a receipt
     ├─ ask_about_report   answers from a saved report without loading
     │                     its body back into the conversation
     ├─ describe_schema    cached metadata, no SQL
     ├─ list_reports
     ├─ delete_reports     ⟵ interrupts for approval BEFORE it runs
     ├─ ask_for_definitions ⟵ interrupts to ask, in the CLI
     └─ remember_definition · note_preference · forget_preference
     └─ the trace
```

Two properties are worth being precise about, because they are what the earlier
hand-built graph was built to guarantee:

**PII cannot leak.** `run_sql` is the only code in the system that returns a
warehouse row, and masking is inside it — before a single row is rendered. This
does not depend on what order anything runs in, and a test reads the source of
every other tool to make sure none of them reaches the warehouse.

**A business term is settled, or the answer says who settled it.** The agent
asks by calling `ask_for_definitions`, and that tool looks the word up in the
definitions your analytics team agreed first, then in your own. Only a word
neither covers reaches you. Lookup is on whole words, so "loyal customers"
finds the agreed meaning of `loyal` while "disloyal customers" does not.

In the CLI the tool interrupts before its body runs, so the turn *pauses*: you
are offered a few candidate meanings, can type your own, or can hand the
decision back and be told what was assumed. What you pick is remembered, so the
question is asked once. Headless callers — the eval harness, Studio — cannot
answer a pause, so for them the tool records the assumption and the answer is
required to disclose it.

This one is weaker than the PII guarantee, and deliberately so. Deciding which
words need defining used to be a regex over nineteen hardcoded words, which
could not be skipped and could not recognise the twentieth — asked about "10
LGB customers" it found nothing and the agent invented a meaning. The judgement
moved to the model, which is better placed to make it, and the cost is that a
tool can be declined where a precondition could not. So it is measured rather
than assumed: `uv run retail-agent eval` reports how often the agent asked
before it spent a query.

## Viewing the pipeline in LangGraph Studio

```bash
uv run langgraph dev
```

Then open:

```
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

Studio renders the compiled agent — the model node, the tool node, and every
middleware hook around them — lets you run a thread against the real BigQuery
connection, and shows the messages after each step, including which SQL the
guard rejected.

What it loads is `src/retail_agent/agent/studio.py`, which builds the same deps
the CLI does. It passes no checkpointer, because the Studio server owns thread
persistence.

Studio does not replace the CLI: the confirmation flow for destructive actions
is a terminal interaction. Studio does show the run paused at the interrupt.

## Tracing (optional)

Set both `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in `.env` to send
traces to [LangSmith](https://smith.langchain.com). You get a span per model
call and per tool call — including the analyst subagent's own loop nested inside
the supervisor's — which is enough to see exactly where a turn went wrong.

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
- **Egress scan** — a report body is swept for anything resembling contact data
  *before* it is saved, and every eval answer is swept too, where one finding
  fails the release gate outright. This is the second line of defence, not the
  first: masking is what makes a leak impossible, and a system that relies on
  scanning output for PII is one clever prompt away from failing.
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
- **Scope** — held by the tool set, not by an input filter. Every tool reads
  retail data or your own saved reports, so a question about the weather has
  nothing to answer it with. There is no lexical filter on your input: the properties one would appear to protect are already held at the SQL
  and column boundaries, which cannot be talked out of them, and the phrasings
  it would match overlap with real work — "delete all reports mentioning Client
  X" is a feature and "how many distinct email addresses" is legitimate
  analysis.
- **Bounded self-correction** — a failed query comes back to the model as an
  error it can act on, and `run_sql` can be called at most 14 times in a turn.
  The counter is middleware, so the bound holds regardless of what the model
  decides. When it runs out the agent says what it could not retrieve instead of
  looping.

## Editing the analyst corpus

The agent answers from a corpus of analyst "trios" — question, SQL, report, and
the metric definitions that connect them. `src/retail_agent/knowledge/seeds.py`
is the hand-authored version, copied into Postgres the first time it runs.

**Editing `seeds.py` does not change a database that has already been seeded.**
Seeding inserts what is absent and leaves what is there, so that an edit made
through the store survives a restart. The cost is that a `seeds.py` change is
silently ignored, and the agent keeps answering from the corpus it was first
given. To see and apply the difference:

```bash
uv run retail-agent trios           # what differs; changes nothing
uv run retail-agent trios --force   # overwrite those from seeds.py
```

Without a database the corpus is read from `seeds.py` directly, so edits take
effect immediately and this does not arise.

## Semantic search over the corpus (optional)

Matching a question to the right trio is lexical by default, which needs
nothing installed and no API key.

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
uv run pytest              # 823 tests, no credentials or database needed
uv run pytest -m db        # 86 tests, needs `docker compose up -d postgres`
uv run pytest -m live      # 16 tests, needs real BigQuery access and an LLM key
uv run pytest -m vector    # 9 tests, needs DENSE_RETRIEVAL deps and an OpenAI key
uv run retail-agent eval   # 47 cases against live BigQuery; exit 0 ships
```

The safety modules are pure functions and are tested first, against an
adversarial corpus. Agent behaviour is tested with a scripted chat model and a
fake warehouse, asserting *what happened* rather than output text — for example
that a rejected query never reaches the warehouse, and that a delete does not
reach the store while you are still being asked about it. Selecting a PII column *bare*
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

Built: BigQuery access, the SQL guard, PII masking, the egress scan over saved
reports and eval answers, the agent and the CLI; the saved-reports library with
its delete-confirmation gate, audit trail and `/undo`; turn traces with
`/trace`, `/trace <id>` and `/metrics`; the eval suite and its release gate; and
the full resilience story — bounded self-correction, an empty result that says
what it probably means, and a provider fallback chain with classified retries.

Also built: personas, so a non-developer can change the agent's tone without a
deploy — versioned, attributed, read per model call, and provably unable to
reach the safety rules — and per-user answer preferences.

Also built: the Golden Bucket of analyst Trios — question, SQL, report and the
metric definitions that connect them — with hybrid lexical/dense retrieval, a
measured relevance floor, a clarifying question when a term is undefined that is
remembered per user.

The agent also picks up how you want answers laid out. Say "keep it brief" and
it saves that as your default and tells you it did — by the CLI, not by the
model, so you are told whether or not it mentions it. It will only do that on
words you actually typed: the evidence is checked against your message, so it
cannot decide on your behalf that you prefer brevity. Noticing is a tool it may
decline to call, which is the one weakness.

Not yet built: the LLM judge for narrative quality, a numeric-provenance check,
and system-level learning. All three are designed in
[the documentation](docs_to_submit/03-requirements.md), which marks each
requirement Built, Partial or Designed and names the command or test that
demonstrates each Built claim.
