# Retail Analysis Agent — submission

A conversational data analysis agent for retail executives, over
`bigquery-public-data.thelook_ecommerce`.

This folder is self-contained: everything the brief asks for is here, and
nothing in it depends on reading another document.

## The two artifacts the task asks for

> *"Design a production-ready full High-Level Design (HLD), accompanied by a
> detailed technical explanation, and build a data analysis chat-agent
> prototype."*

| Artifact | Where |
|---|---|
| **High-Level Design** | [00-diagrams.md](00-diagrams.md) — the five diagrams: system view, agent hierarchy, one turn, the Golden Bucket, the destructive-action gate<br>[01-architecture.md](01-architecture.md) — building blocks, services, compute, layers, and the reasoning for every service, model and framework chosen |
| **Detailed technical explanation** | [02-data-flow.md](02-data-flow.md) — how a question travels, what crosses each boundary, how to extend it<br>[03-requirements.md](03-requirements.md) — each of the eight requirements answered, including error handling and fallback<br>[04-setup-and-run.md](04-setup-and-run.md) — setup, and an annotated example run |
| **Prototype** | `src/retail_agent/`, run per [04-setup-and-run.md](04-setup-and-run.md) |

## Deliverables, one by one

| # | Deliverable | Where |
|---|---|---|
| 1 | Architecture diagram | [00-diagrams.md](00-diagrams.md) — all five, or [01-architecture.md](01-architecture.md) for the reasoning |
| 2a | Reasoning for cloud services, models and frameworks | [01-architecture.md](01-architecture.md#why-these-choices) |
| 2b | Data flow between components | [02-data-flow.md](02-data-flow.md) |
| 2c | Error handling and fallback strategies | [03-requirements.md](03-requirements.md#5-resilience-and-graceful-error-handling) |
| 2d | Setup instructions and example run | [04-setup-and-run.md](04-setup-and-run.md) |
| 2e | How each requirement is handled | [03-requirements.md](03-requirements.md) |
| 3 | Working prototype | source in `src/retail_agent/`, run per [04-setup-and-run.md](04-setup-and-run.md) |
| 4 | CLI interface | `uv run retail-agent chat` |
| 5 | Runnable on another machine | [04-setup-and-run.md](04-setup-and-run.md) |
| 6 | Framework | LangChain `create_agent` on LangGraph — [01-architecture.md](01-architecture.md#why-these-choices) |

## What is built, and what is designed

The brief asks for a production design *and* a prototype demonstrating at least
two of five named requirements. They are different things, so every section is
labelled:

| Label | Meaning |
|---|---|
| **Built** | Runs today. The section names the command, file or test that shows it. |
| **Partial** | The core mechanism runs; specific named pieces are deferred. |
| **Designed** | Not in the prototype. The production design is given in full. |

| Brief requirement | Status |
|---|---|
| 1. Hybrid Intelligence — the Golden Bucket | **Built** |
| 2. Safety & PII Masking | **Built** |
| 3. High-Stakes Oversight (destructive ops) | **Built** |
| 4a. Continuous Improvement — user level | **Partial** |
| 4b. Continuous Improvement — system level | Designed |
| 5. Resilience & Graceful Error Handling | **Built** |
| 6. Quality Assurance | **Built** |
| 7. Observability | **Built** |
| 8. Agility — Persona Management | **Built** |
| Extensibility (new capabilities, new data sources) | **Built** |

The prototype implements five of the five requirements deliverable 3 offers,
rather than the two it asks for: PII masking, high-stakes oversight, resilience,
quality assurance and observability.

## Measured, not asserted

| | |
|---|---|
| Offline tests | **823** — no credentials, no database, no API key |
| Against Postgres | 86 (`-m db`) |
| Against live BigQuery and a live model | 16 (`-m live`) |
| Eval suite | **47 cases, 40 correct, 85%** execution accuracy |
| PII leaks in the eval | **0** — a single finding fails the release gate outright |

Queries are real BigQuery queries billed to a real project. Nothing is stubbed
at runtime.

## The one-paragraph version

The obvious reading of the brief is "build a text-to-SQL chatbot", and the
brief's own example questions show why that fails. *"Why did our churn rate
spike?"* — theLook has no subscriptions and no cancellations, so churn cannot be
read off the schema at all; a human decided what it means. Without those
business definitions the agent still answers: it picks a definition, writes
clean SQL, and returns a confident number that does not match what Finance
reports, with nothing in the output revealing a guess was made. That silent
failure is what the Golden Bucket exists to prevent, and it is the requirement
the rest of the architecture is shaped around — along with the fact that a
language model is writing SQL against a billed warehouse of personal data on
behalf of people who cannot read SQL.
