"""Persistent storage for the information system.

SQLite is the default (a single file, enough for development and tests). Set DATABASE_URL to a
PostgreSQL URL (for example the one Render gives to a managed database) to keep the data across
deployments. All SQL in the project uses "?" placeholders; they are translated for PostgreSQL here.
"""
import os
import sqlite3
from datetime import datetime, timedelta, timezone

from flask import current_app, g

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQLITE_DEFAULT = os.path.join(ROOT, "data", "app.db")

ROLES = ("admin", "operator", "viewer")


def now_iso():
    """Local time of Chía (UTC-5) as 'YYYY-MM-DD HH:MM:SS'. Text timestamps sort correctly in any engine."""
    return datetime.now(timezone(timedelta(hours=-5))).strftime("%Y-%m-%d %H:%M:%S")


SCHEMA = [
    """CREATE TABLE IF NOT EXISTS users (
        id {PK},
        username TEXT NOT NULL UNIQUE,
        full_name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'viewer' CHECK (role IN ('admin', 'operator', 'viewer')),
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        last_login TEXT)""",
    """CREATE TABLE IF NOT EXISTS intersections (
        id {PK},
        code TEXT NOT NULL UNIQUE,
        name TEXT NOT NULL,
        junction INTEGER UNIQUE,
        phases INTEGER NOT NULL DEFAULT 2 CHECK (phases BETWEEN 2 AND 8),
        latitude DOUBLE PRECISION,
        longitude DOUBLE PRECISION,
        speed_limit_kmh INTEGER CHECK (speed_limit_kmh IS NULL OR speed_limit_kmh BETWEEN 10 AND 120),
        notes TEXT,
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS model_versions (
        id {PK},
        version TEXT NOT NULL UNIQUE,
        model_name TEXT NOT NULL,
        trained_on TEXT,
        test_r2 DOUBLE PRECISION,
        test_mae DOUBLE PRECISION,
        test_rmse DOUBLE PRECISION,
        is_current INTEGER NOT NULL DEFAULT 0,
        registered_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS inferences (
        id {PK},
        user_id INTEGER NOT NULL REFERENCES users(id),
        intersection_id INTEGER NOT NULL REFERENCES intersections(id),
        target_datetime TEXT NOT NULL,
        predicted_vehicles DOUBLE PRECISION NOT NULL,
        group_label TEXT,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        latency_ms DOUBLE PRECISION,
        source TEXT NOT NULL DEFAULT 'web',
        curve_json TEXT,
        notes_json TEXT,
        observed_vehicles DOUBLE PRECISION CHECK (observed_vehicles IS NULL OR observed_vehicles >= 0),
        observed_by INTEGER REFERENCES users(id),
        observed_at TEXT,
        abs_error DOUBLE PRECISION,
        created_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS timing_plans (
        id {PK},
        intersection_id INTEGER NOT NULL REFERENCES intersections(id),
        inference_id INTEGER NOT NULL REFERENCES inferences(id),
        created_by INTEGER NOT NULL REFERENCES users(id),
        decided_by INTEGER REFERENCES users(id),
        status TEXT NOT NULL DEFAULT 'draft'
            CHECK (status IN ('draft', 'pending', 'approved', 'rejected', 'applied')),
        model_green_s INTEGER NOT NULL,
        final_green_s INTEGER NOT NULL CHECK (final_green_s BETWEEN 8 AND 60),
        rationale TEXT,
        decision_note TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        decided_at TEXT,
        applied_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS incidents (
        id {PK},
        intersection_id INTEGER NOT NULL REFERENCES intersections(id),
        inference_id INTEGER REFERENCES inferences(id),
        plan_id INTEGER REFERENCES timing_plans(id),
        reported_by INTEGER NOT NULL REFERENCES users(id),
        assigned_to INTEGER REFERENCES users(id),
        kind TEXT NOT NULL DEFAULT 'incident' CHECK (kind IN ('feedback', 'incident', 'model_issue')),
        severity TEXT NOT NULL DEFAULT 'low' CHECK (severity IN ('low', 'medium', 'high')),
        description TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'in_review', 'closed')),
        resolution TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        closed_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS app_logs (
        id {PK},
        level TEXT NOT NULL,
        event TEXT NOT NULL,
        user_id INTEGER,
        endpoint TEXT,
        status_code INTEGER,
        latency_ms DOUBLE PRECISION,
        detail TEXT,
        created_at TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS ix_inferences_user ON inferences (user_id, created_at)",
    "CREATE INDEX IF NOT EXISTS ix_inferences_inter ON inferences (intersection_id, target_datetime)",
    "CREATE INDEX IF NOT EXISTS ix_plans_status ON timing_plans (status)",
    "CREATE INDEX IF NOT EXISTS ix_incidents_status ON incidents (status)",
    "CREATE INDEX IF NOT EXISTS ix_logs_event ON app_logs (event, created_at)",
]


class Session:
    """One database connection with small helpers. Rows are returned as plain dictionaries."""

    def __init__(self, kind, conn):
        self.kind, self.conn = kind, conn

    def _prep(self, sql):
        return sql.replace("?", "%s") if self.kind == "postgres" else sql

    def _cursor(self):
        if self.kind == "postgres":
            import psycopg2.extras
            return self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        return self.conn.cursor()

    def _execute(self, cur, sql, params):
        try:
            cur.execute(sql, tuple(params))
        except Exception:
            self.conn.rollback()   # PostgreSQL refuses new statements after an error until the transaction ends
            raise

    def all(self, sql, params=()):
        cur = self._cursor()
        self._execute(cur, self._prep(sql), params)
        rows = cur.fetchall()
        cur.close()
        return [dict(r) for r in rows]

    def one(self, sql, params=()):
        rows = self.all(sql, params)
        return rows[0] if rows else None

    def run(self, sql, params=()):
        cur = self._cursor()
        self._execute(cur, self._prep(sql), params)
        n = cur.rowcount
        cur.close()
        self.conn.commit()
        return n

    def insert(self, table, values):
        cols = list(values)
        marks = ", ".join("?" for _ in cols)
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({marks})"
        cur = self._cursor()
        args = tuple(values[c] for c in cols)
        if self.kind == "postgres":
            self._execute(cur, self._prep(sql + " RETURNING id"), args)
            new_id = cur.fetchone()["id"]
        else:
            self._execute(cur, sql, args)
            new_id = cur.lastrowid
        cur.close()
        self.conn.commit()
        return new_id

    def rollback(self):
        self.conn.rollback()

    def close(self):
        self.conn.close()


class Database:
    def __init__(self, url=None, path=None):
        url = os.environ.get("DATABASE_URL", "") if url is None else url
        if url.startswith(("postgres://", "postgresql://")):
            self.kind = "postgres"
            self.dsn = url.replace("postgres://", "postgresql://", 1)
            self.path = None
        else:
            self.kind = "sqlite"
            self.path = path or os.environ.get("DATABASE_PATH") or SQLITE_DEFAULT
            self.dsn = None

    def connect(self):
        if self.kind == "postgres":
            import psycopg2
            return Session("postgres", psycopg2.connect(self.dsn))
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return Session("sqlite", conn)

    def create_schema(self):
        pk = "SERIAL PRIMARY KEY" if self.kind == "postgres" else "INTEGER PRIMARY KEY AUTOINCREMENT"
        s = self.connect()
        try:
            for stmt in SCHEMA:
                cur = s._cursor()
                cur.execute(stmt.replace("{PK}", pk))
                cur.close()
            s.conn.commit()
        finally:
            s.close()


def get_db():
    """Per-request session, closed automatically when the request ends."""
    if "db_session" not in g:
        g.db_session = current_app.extensions["db"].connect()
    return g.db_session


def close_db(_exc=None):
    s = g.pop("db_session", None)
    if s is not None:
        s.close()


def init_app(app, database=None):
    app.extensions["db"] = database or Database()
    app.extensions["db"].create_schema()
    app.teardown_appcontext(close_db)


# Shortcuts used by the services
def q(sql, params=()):
    return get_db().all(sql, params)


def q1(sql, params=()):
    return get_db().one(sql, params)


def run(sql, params=()):
    return get_db().run(sql, params)


def insert(table, values):
    return get_db().insert(table, values)


def log_event(level, event, user_id=None, endpoint=None, status_code=None, latency_ms=None, detail=None):
    """Errors, inference requests and response times are stored here (table app_logs)."""
    try:
        insert("app_logs", {"level": level, "event": event, "user_id": user_id, "endpoint": endpoint,
                            "status_code": status_code, "latency_ms": latency_ms,
                            "detail": (str(detail)[:500] if detail is not None else None),
                            "created_at": now_iso()})
    except Exception:  # logging must never break a request
        pass
