-- Phase 1 needs only a migration ledger. LangGraph's PostgresSaver creates its
-- own checkpoint tables via setup(). Reports, traces, personas and preferences
-- arrive in phase 2 as 002_*.sql.

CREATE TABLE IF NOT EXISTS schema_migrations (
    filename    TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
