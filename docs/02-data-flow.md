# 2. Data flow between components

Deliverable 2b, plus how the system extends to new capabilities and new data
sources.

## One analysis question, end to end

Walking *"who are our loyal customers by spend?"*

1. **supervisor** is called with a system prompt assembled *now* — the active
   persona row, the safety rules, this user's saved preference notes — and
   decides the question needs data.

2. **`ask_for_definitions`**, if the model judges a word needs settling. The tool
   looks it up in the agreed corpus first, then in this executive's own
   definitions. Matching is on whole words, so *"loyal customers"* finds the
   agreed meaning of `loyal` while *"disloyal customers"* does not. Only a word
   neither source covers reaches the executive. In the CLI this is an interrupt
   *before* the tool body runs, so the turn pauses; headless callers record the
   assumption instead and the answer is required to disclose it.

3. **analyst** resolves meaning before building anything: hybrid retrieval over
   the Golden Bucket, then this user's own remembered definitions, then the
   column conventions for whichever columns are in scope.

4. **analyst's model** writes SQL, given the schema DDL, the values each
   enumerable column actually holds, and the definitions from step 3.

5. **sql_guard** parses it to an AST. Rejects anything that is not a single
   read-only `SELECT` over the allowed tables, rejects `SELECT *`, rejects PII
   columns used in a way masking could not survive, qualifies bare table names
   with the dataset, and enforces a `LIMIT`. No model involved. A violation is
   returned to the model as a tool error, and it tries again on its budget.

6. **dry_run** asks BigQuery for a byte estimate and refuses anything over
   `BQ_MAX_BYTES_BILLED`. Dry runs are not billed, so this gate is free.

7. **execute** runs it with `maximum_bytes_billed` as a hard ceiling and a
   60-second timeout, fetching at most `display_row_limit` rows while
   `total_rows` still reports the true size of the result.

8. **mask_dataframe** applies the declarative policy the moment rows return.
   **This is the trust boundary**: raw rows exist only as a local variable
   inside `run_sql`.

9. The masked frame is rendered to markdown and returned to the analyst's model,
   carrying a sample warning if the read was capped and the empty-result hint if
   nothing matched. `PIIMiddleware` sweeps that tool result before the model
   reads it. The model queries again or answers.

10. **analyst** appends the assumption note if it had to decide a term alone, and
    returns its findings to the supervisor as a tool result.

11. **supervisor** writes the answer, calling `report_writer` if a report was
    asked for. That subagent has no data tools, so it cannot invent a figure the
    analyst did not find; it saves what it wrote — scanned for contact data
    first — and returns only a receipt.

12. **after_agent** persists the trace and records what the thread will cost
    every later turn.

13. **render** prints it, with a footnote reporting redactions and attempts.

A failure at step 5, 6 or 7 returns the error to the model rather than raising,
and costs one of `run_sql`'s call budget. When that budget is exhausted the tool
stops being callable and the model answers with what it has, saying what it could
not retrieve.

## What crosses each boundary

| Boundary | What crosses it | What cannot |
|---|---|---|
| model → warehouse | one validated read-only `SELECT` | DML, multiple statements, unqualified tables, unmaskable projections |
| warehouse → model | masked rows only | any raw personal value — masking is inside the only function that reads a row |
| tool → conversation state | a rendered markdown table plus row counts | the `MaskedFrame` itself is not put in the trace, so a trace cannot become a second disclosure path |
| model → report library | a title and a body | the set of reports a delete touches — that is resolved in SQL, from the model's search term |
| persona → behaviour | tone and format | the safety rules, which are appended after it, and the guards, which never read prompt text |

## What one turn records

There is no `TurnState`. An agent has a message list, and the rows it saw were
rendered into a `ToolMessage` string; re-parsing that string to score or trace it
would measure how the tool formatted its output. So the tools write what they did
into a `TurnCapture` on the way past, and the eval and the trace read it back
untouched.

```python
@dataclass
class TurnCapture:
    user_id: str; session_id: str; question: str; turn_id: str

    frame: MaskedFrame | None        # the last successful result. Only ever masked.
    executed_sql: str                # the query that ran, not the first draft
    attempts: list[dict]             # every try, with the guard's verdict and outcome
    events: list[tuple[str, int, str]]   # (step, ms, what it decided) — for /trace
    trio_ids: list[str]              # what the Golden Bucket settled
    assumed_terms: list[str]         # what it had to decide alone
    redactions: int
    calls: int
    preference_changes: list[tuple[str, str]]   # the CLI announces these itself
    reports_written: list[WrittenReport]        # bodies included, so the CLI can print them
    context_tokens: int              # what this thread costs every later turn
    status: str                      # "ok" or "failed"
    pending: PendingDelete | None    # resolved by the approval gate, read by the tool
    pending_definition: PendingDefinition | None
```

One capture per turn, created by the caller and closed over by the tools.
Deliberately not global: eval cases run sequentially today, but a shared capture
would attribute case 4's rows to case 3 the moment that stopped being true.

`intent` is not stored — it is derived from which tools ran. The graph asked a
model to classify the turn before doing any of it; which tools were actually
called is the same answer, arrived at afterwards and for free.

## Extending it

Two protocols and one pattern carry all of it.

**A new data source is one adapter and no change to the agent.**

```python
class DataSource(Protocol):        # implemented: BigQuerySource
    dialect: str
    def list_tables(self) -> list[str]: ...
    def describe(self, table: str) -> TableSchema: ...
    def describe_all(self) -> list[TableSchema]: ...
    def dry_run(self, sql: str) -> DryRunResult: ...
    def assert_within_budget(self, sql: str) -> DryRunResult: ...
    def execute(self, sql: str) -> QueryResult: ...
```

The split is drawn at what is dialect-specific: `column_values` holds the policy
every warehouse shares — which columns may be enumerated at all, and how values
are rendered into a schema — while the `APPROX_TOP_COUNT` query that fetches them
lives in the BigQuery adapter with the dialect that answers it.

**A new capability is a `@tool` in `build_tools`.** "Email this report", "search
the web for trends", "generate a chart" — each is one function. When the
capability needs its own loop — its own tools to call and decide between — it
is a compiled `create_agent` behind that function, the way `analyst` is. When
it only needs a model, one call through `resilient_call` gives it the retry and
fallback a loop would otherwise get from middleware, without paying for a
graph it has no use for. `report_writer` and `ask_about_report` are that
second shape:

```python
def build_subagents(deps, capture):
    @tool
    def analyst(question: str) -> str:
        """Query the retail data and report what it found."""
        ...
        agent = create_agent(model=deps.llm, tools=..., middleware=...)
        return final_text(agent.invoke(...))

    @tool
    def report_writer(brief: str, title: str) -> str:
        """Write a report from findings, save it, and show it to the executive."""
        ...
        return resilient_call(deps, lambda model: model.invoke(...))

    return [analyst, report_writer, ask_about_report]
```

Both shapes end up the same `@tool` on the supervisor's tool list, so the
supervisor never has to know which one it is calling. That is the whole
subagent pattern, and it needs no dependency beyond the one already in use.

Two rules keep it safe as tools are added:

- **Anything destructive goes on the supervisor**, not inside a subagent, so its
  approval interrupt fires at the top-level tool boundary where the interface can
  render a manifest and resume. An interrupt raised one `.invoke()` down is not
  reachable by the CLI.
- **Nothing but `run_sql` reads the warehouse.** A tool that queried BigQuery
  outside it would return unmasked rows, and a test reads the source of every
  other tool to make sure none does.
