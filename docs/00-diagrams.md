# Diagrams — the High-Level Design at a glance

Deliverable 1, in one place. Every diagram in this submission, with a line
saying what it shows and where the reasoning behind it lives. The HLD's prose —
services, components, layers and why each was chosen — is
[01-architecture.md](01-architecture.md).

---

## 1. System view

The whole target system. Colour marks what runs today, so the diagram doubles as
a statement of what the prototype implements rather than what it aspires to.
Services and the reasoning for each are in
[01-architecture.md](01-architecture.md#why-these-choices).

```mermaid
flowchart TB
    subgraph client["Client"]
        CLI["CLI REPL<br/><i>rich</i>"]
        ADMIN["Persona admin page<br/><i>non-developers edit tone</i>"]
        SLACK["Slack / web surface"]
    end

    subgraph run["Cloud Run — stateless, autoscaled"]
        API["Agent service<br/><i>ReAct supervisor + subagents</i>"]
        GUARD["Safety layer<br/><i>SQL guard · PII policy · masking · egress on save</i>"]
        RESIL["Provider chain<br/><i>retry · fallback</i>"]
        CAP["Further subagents<br/><i>charts · email · web search</i>"]
        RETR["Trio retrieval<br/><i>hybrid + RRF</i>"]
    end

    subgraph data["Data plane"]
        BQ[("BigQuery<br/><i>thelook_ecommerce</i>")]
        SQL[("Cloud SQL — Postgres<br/><i>checkpoints · reports · audit<br/>traces · personas · prefs · trios</i>")]
        GCS[("Cloud Storage<br/><i>trio corpus · report bodies</i>")]
        VEC[("Vertex AI Vector Search<br/><i>trio embeddings</i>")]
    end

    subgraph ext["External"]
        LLM["LLM providers<br/><i>gemini → openai → ollama</i>"]
        LS["LangSmith"]
    end

    subgraph ops["Platform"]
        SM["Secret Manager"]
        LOG["Cloud Logging<br/>+ BigQuery sink"]
        SCHED["Cloud Scheduler + Pub/Sub<br/><i>nightly aggregation · re-embed</i>"]
        EVAL["Eval harness<br/><i>release gate</i>"]
    end

    CLI --> API
    SLACK -.-> API
    ADMIN -.-> SQL
    API --> GUARD
    API --> RESIL
    API -.-> CAP
    API --> RETR
    GUARD -->|"validated SQL only"| BQ
    BQ -->|"rows"| GUARD
    GUARD -->|"masked rows"| API
    API <--> SQL
    RESIL --> LLM
    CAP -.-> SQL
    CAP -.-> GCS
    RETR --> VEC
    VEC -.->|"embeddings of"| GCS
    API -.-> LS
    API --> LOG
    SM -.->|"credentials"| API
    SCHED --> API
    LOG --> SCHED
    EVAL -.->|"gates deploys of"| API

    classDef built fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef progress fill:#fef9c3,stroke:#ca8a04,color:#713f12
    classDef planned fill:#f1f5f9,stroke:#94a3b8,color:#334155,stroke-dasharray:4 3
    class CLI,API,GUARD,RESIL,RETR,BQ,SQL,LLM,LS,EVAL built
    class ADMIN progress
    class SLACK,CAP,GCS,VEC,SM,LOG,SCHED planned
```

<sub>**Green** runs today · **amber** partly built · **dashed grey** designed, not built.</sub>

---

## 2. Agent hierarchy

Who delegates to whom, and — the part that matters — what each one can reach.
The system view above draws the agent as a single service because that is what
it is at the compute level: one process, one deployment. Inside it there are
three tiers, and the boundaries between them are where the safety properties
live.

```mermaid
flowchart TB
    IFACE["CLI / any surface"]

    subgraph t1["Tier 1 — supervisor"]
        SUP["supervisor<br/><i>own model · persona + safety rules<br/>+ preference notes, read per model call</i>"]
    end

    subgraph t2["Tier 2 — subagents: a compiled agent behind a callable"]
        AN["<b>analyst</b><br/><i>own model, own loop,<br/>own query budget</i>"]
        RW["<b>report_writer</b><br/><i>own model · NO tools</i>"]
        AR["<b>ask_about_report</b><br/><i>own model · NO tools</i>"]
    end

    subgraph t1tools["Tier 1 tools — plain functions, no model"]
        SCH["describe_schema"]
        LIB["list_reports"]
        DEL["delete_reports"]
        DEF["ask_for_definitions"]
        MEM["remember_definition<br/>note_preference<br/>forget_preference"]
    end

    subgraph t3["Tier 3 — the analyst's own tools"]
        SQLT["<b>run_sql</b><br/><i>guard → dry_run → execute → mask</i>"]
        LOOK["lookup_definitions"]
    end

    BQ[("BigQuery")]
    PG[("Postgres<br/><i>reports · definitions · prefs</i>")]

    IFACE --> SUP
    SUP --> AN
    SUP --> RW
    SUP --> AR
    SUP --> SCH
    SUP --> LIB
    SUP --> DEL
    SUP --> DEF
    SUP --> MEM

    AN --> SQLT
    AN --> LOOK

    SQLT ==>|"rows — masked inside run_sql"| BQ
    SCH -.->|"schema metadata only,<br/>cached, no query"| BQ
    LOOK --> PG
    DEF --> PG
    MEM --> PG
    LIB --> PG
    DEL --> PG
    AR --> PG

    classDef only fill:#dcfce7,stroke:#16a34a,color:#14532d,stroke-width:2px
    classDef none fill:#f1f5f9,stroke:#94a3b8,color:#334155
    class SQLT only
    class RW,AR none
```

Four properties are visible here and nowhere else:

**Only `run_sql` returns a warehouse row.** It is a tier-3 tool held by one
subagent, and masking is inside it, so there is exactly one path from BigQuery
into model context. `describe_schema` also touches BigQuery — the dotted edge —
but only for cached schema metadata; it executes no query and returns no rows,
which is why it can sit at tier 1 safely. The distinction is the one the test
enforces: it reads the source of every other tool and fails if any of them calls
`deps.source.execute`. Checked, rather than drawn.

**`report_writer` has no tools at all.** That is the whole mechanism preventing
an invented figure in a report: it can only write from the findings the analyst
passed it, because it has no way to look anything up.

**Destructive tools stay at tier 1.** `delete_reports` sits on the supervisor so
its approval interrupt fires at the top-level tool boundary, where the interface
can render a manifest and resume. An interrupt raised one `.invoke()` down is
not reachable by the CLI. The same is true of `ask_for_definitions`.

**Each tier bounds itself.** The analyst carries the query budget because it is
the loop that spends it; the supervisor carries the approval gate because it is
the boundary the interface can resume. Both are capped on model calls
independently, so a runaway subagent cannot spend the supervisor's allowance.

Adding a capability means adding a node at tier 1 — a plain function, or another
compiled agent behind a callable if it needs its own loop. `analyst`,
`report_writer` and `ask_about_report` are three instances of that one pattern,
so extensibility is demonstrated rather than promised.

---

## 3. One turn

The supervisor, its ten tools, the two points where a turn stops for a person,
and the analyst subagent's inner loop. Everything that constrains the model is a
middleware, a tool precondition or a SQL predicate — never an instruction in a
prompt. The middleware stack is tabulated in
[01-architecture.md](01-architecture.md#the-turn); the step-by-step
walkthrough is [02-data-flow.md](02-data-flow.md).

```mermaid
flowchart TD
    IN([User turn]) --> SUP[supervisor model<br/><i>persona + safety rules<br/>+ preference notes, per call</i>]

    SUP -->|no tool needed| REC
    SUP --> ANALYST[[analyst subagent]]
    SUP --> WRITER[[report_writer subagent]]
    SUP --> ASKREPORT[[ask_about_report subagent]]
    SUP --> SCHEMA[describe_schema<br/><i>cached metadata, no SQL</i>]
    SUP --> ASKDEF[ask_for_definitions]
    SUP --> MEM[remember_definition<br/>note_preference · forget_preference]
    SUP --> LIB[list_reports]
    SUP --> DEL[delete_reports]

    DEL --> GATE(((approval interrupt<br/>manifest resolved read-only)))
    GATE -->|typed token| APPLY[soft delete + audit]
    GATE -->|anything else| ABORT[nothing changed]

    ASKDEF --> DEFGATE(((definition interrupt<br/>terms the corpus or the user<br/>already settles are filtered out<br/><i>CLI only</i>)))
    DEFGATE -->|picked or typed| REMEMBER[remembered, then the tool reads it back]
    DEFGATE -->|decide for me| ASSUME[reject: choose and disclose]
    DEFGATE -->|cancelled| ABORT2[nothing queried]

    subgraph analystloop["analyst subagent — its own loop"]
        RECALL[resolve definitions<br/><i>trios, then the user's own</i>]
        RECALL --> LOOP[model]
        LOOP --> LOOKUP[lookup_definitions<br/><i>re-reads the corpus mid-loop</i>]
        LOOKUP --> LOOP
        LOOP --> SQLT[run_sql]
        SQLT --> G{sql_guard<br/><i>AST, no model</i>}
        G -->|violation| BACK[error to the model]
        G -->|pass| DRY[dry_run + execute]
        DRY -->|error| BACK
        DRY -->|rows| MASK[mask_dataframe<br/><b>PII stripped here</b>]
        BACK --> LOOP
        MASK -->|empty| HINT[rows + 'this is usually<br/>an exact-match filter']
        MASK -->|rows| LOOP
        HINT --> LOOP
    end

    ANALYST -.-> RECALL
    APPLY --> REC
    ABORT --> REC
    SCHEMA --> REC
    MEM --> REC
    LIB --> REC
    WRITER --> REC
    ASKREPORT --> REC

    REC[after_agent<br/><i>persist the trace</i>] --> OUT[render, with the footnote]
```

---

## 4. The Golden Bucket

Left: how analyst knowledge reaches a query at run time. Right: how the corpus
is kept current. Nothing flows from the agent into the corpus without passing
`human review`, and that edge is the whole safety argument for the learning
loop. Detail in
[03-requirements.md](03-requirements.md#1-hybrid-intelligence--the-golden-bucket).

```mermaid
flowchart LR
    subgraph query["At query time"]
        Q([User question]) --> EMB[embed]
        Q --> TERMS[extract business terms]
        EMB --> DENSE[dense search<br/><i>Vector Search</i>]
        TERMS --> BM25[BM25 + tag filter]
        DENSE --> RRF{{"RRF fusion, top 5"}}
        BM25 --> RRF
        RRF --> REL{clears relevance<br/>threshold?}
        REL -->|yes| INJECT[inject metric_definitions<br/>+ report style]
        REL -->|no| UNDEF{undefined term<br/>in question?}
        UNDEF -->|yes| STATE["ask, or state the<br/>assumed definition"]
        UNDEF -->|no| PROCEED[proceed unaided]
    end

    subgraph update["Keeping it current"]
        A1[analyst-authored trios] --> GATE
        A2["promotion<br/><i>user approves a report</i>"] --> CAND[(candidate queue)]
        A3["correction capture<br/><i>'no, churn means X'</i>"] --> CAND
        CAND --> GATE{{human review}}
        GATE -->|merge| CORPUS[("trio corpus<br/><i>versioned, superseded_by</i>")]
        GATE -->|reject| DROP([discarded])
        CORPUS --> REEMBED[re-embed]
    end

    REEMBED -.->|serves| DENSE
```

---

## 5. The destructive-action gate

Deleting saved reports is the one destructive capability, and the ordering is
the safety property, so it is drawn as a sequence rather than a flow. The model
names the search term; it never decides which reports match. Detail in
[03-requirements.md](03-requirements.md#3-high-stakes-oversight-destructive-ops).

```mermaid
sequenceDiagram
    actor U as Executive
    participant G as Agent (approval gate)
    participant M as Model
    participant P as Postgres

    U->>G: "delete all reports mentioning Client X"
    G->>M: which tool, and with what arguments?
    M-->>G: delete_reports(term="Client X")
    Note over G,M: The model names the term.<br/>It never decides which reports match.
    G->>P: SELECT id, title FROM reports<br/>WHERE owner_id = :me AND deleted_at IS NULL<br/>AND search @@ to_tsquery('Client X')
    P-->>G: 7 rows
    G->>U: manifest — the exact 7 titles, and the count
    G-->>G: interrupt before the tool runs<br/>manifest held as pending(action_id)

    alt user types DELETE 7
        U->>G: DELETE 7
        G->>P: UPDATE reports SET deleted_at = now()<br/>WHERE id = ANY(:ids) AND owner_id = :me
        G->>P: INSERT INTO report_audit<br/>(who, ids, when, token, action_id)
        G->>U: 7 reports deleted. /undo to reverse.
    else anything else
        U->>G: "no" / a new question / Ctrl-C
        G->>U: aborted, nothing changed
    end

    Note over G,P: Replaying a consumed action_id is a no-op,<br/>so a double-resume cannot delete twice.
```
