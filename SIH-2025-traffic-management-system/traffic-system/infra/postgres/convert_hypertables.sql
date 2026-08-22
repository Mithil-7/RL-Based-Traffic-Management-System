-- One-time setup: convert the time-series tables SQLAlchemy created at app
-- startup into TimescaleDB hypertables, and set a retention policy on raw
-- telemetry. Idempotent (safe to re-run). Invoke via `make db-hypertables`
-- (see Makefile), which runs this against the running `postgres` container
-- after the api/brain/ingestion services have started at least once.

SELECT create_hypertable('telemetry_records', 'timestamp', if_not_exists => TRUE, migrate_data => TRUE);
SELECT create_hypertable('signal_commands', 'issued_at', if_not_exists => TRUE, migrate_data => TRUE);
SELECT create_hypertable('intersection_snapshots', 'timestamp', if_not_exists => TRUE, migrate_data => TRUE);

-- Raw telemetry at multi-second granularity is only useful for a few weeks
-- of dashboards/KPIs; keep it bounded so disk use is predictable at city
-- scale. Remove this (or export to cold storage first) for a deployment
-- that wants to keep raw data longer for offline retraining.
SELECT add_retention_policy('telemetry_records', INTERVAL '30 days', if_not_exists => TRUE);
