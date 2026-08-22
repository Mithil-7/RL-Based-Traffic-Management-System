-- Runs automatically on first Postgres container start (mounted into
-- /docker-entrypoint-initdb.d/ via docker-compose.yml, using the official
-- `timescale/timescaledb` image which bundles the extension).
--
-- Only the extension is enabled here. Hypertable conversion happens in
-- `convert_hypertables.sql` instead of here, because SQLAlchemy creates the
-- actual tables at *application* startup (after this init script has
-- already run against a fresh volume) -- trying to convert tables that
-- don't exist yet would fail. Run `make db-hypertables` once after the
-- stack's first `docker compose up` (see Makefile).

CREATE EXTENSION IF NOT EXISTS timescaledb;
