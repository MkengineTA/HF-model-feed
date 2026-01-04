# database.py
from __future__ import annotations

import sqlite3
import json
import logging
from datetime import datetime, timedelta, timezone
import dateutil.parser

logger = logging.getLogger("EdgeAIScout")

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        return self.conn

    def init_db(self) -> None:
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                name TEXT,
                author TEXT,
                created_at TIMESTAMP,
                last_modified TIMESTAMP,
                params_est REAL,
                hf_tags TEXT,
                llm_analysis TEXT,
                user_label INTEGER DEFAULT NULL,
                status TEXT
            )
            """
        )

        # Migrations
        self._ensure_column(cursor, "models", "last_modified", "TIMESTAMP")
        self._ensure_column(cursor, "models", "namespace", "TEXT")
        self._ensure_column(cursor, "models", "author_kind", "TEXT")
        self._ensure_column(cursor, "models", "trust_tier", "INTEGER")
        self._ensure_column(cursor, "models", "pipeline_tag", "TEXT")
        self._ensure_column(cursor, "models", "filter_trace", "TEXT")
        self._ensure_column(cursor, "models", "report_notes", "TEXT")

        # New: separate total/active params
        self._ensure_column(cursor, "models", "params_total_b", "REAL")
        self._ensure_column(cursor, "models", "params_active_b", "REAL")
        self._ensure_column(cursor, "models", "params_source", "TEXT")

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS dynamic_blacklist (
                namespace TEXT PRIMARY KEY,
                added_at TIMESTAMP NOT NULL,
                reason TEXT,
                count INTEGER DEFAULT 0,
                last_seen TIMESTAMP NOT NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS authors (
                namespace TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                num_followers INTEGER,
                is_pro INTEGER,
                created_at TIMESTAMP,
                last_checked TIMESTAMP NOT NULL,
                raw_json TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS dynamic_whitelist (
                namespace TEXT PRIMARY KEY,
                added_at TIMESTAMP NOT NULL,
                reason TEXT,
                count INTEGER DEFAULT 0,
                last_seen TIMESTAMP NOT NULL
            )
            """
        )

        conn.commit()

    def _ensure_column(self, cursor, table: str, column: str, col_type: str) -> None:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [row["name"] for row in cursor.fetchall()]
        if column not in columns:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

    def get_existing_ids(self) -> set[str]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM models")
        rows = cursor.fetchall()
        return {row["id"] for row in rows}

    def get_model_last_modified(self, model_id: str):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT last_modified FROM models WHERE id = ?", (model_id,))
        row = cursor.fetchone()
        if row and row["last_modified"]:
            return row["last_modified"]
        return None

    def get_last_run_timestamp(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM metadata WHERE key = 'last_run'")
        row = cursor.fetchone()
        if row and row["value"]:
            try:
                return dateutil.parser.parse(row["value"])
            except Exception:
                pass
        return datetime.now(timezone.utc) - timedelta(hours=24)

    def set_last_run_timestamp(self, timestamp):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('last_run', ?)",
            (timestamp.isoformat(),),
        )
        conn.commit()

    def _migrate_dynamic_blacklist_metadata(self) -> set[str]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM metadata WHERE key = 'dynamic_blacklist'")
        row = cursor.fetchone()
        if row and row["value"]:
            try:
                data = json.loads(row["value"])
                if not isinstance(data, (list, tuple, set)):
                    logger.warning("Dynamic blacklist metadata is not a list; resetting.")
                    return set()
                namespaces = {str(x) for x in data if str(x).strip()}
                if namespaces:
                    now = datetime.now(timezone.utc).isoformat()
                    for ns in namespaces:
                        cursor.execute(
                            """
                            INSERT OR IGNORE INTO dynamic_blacklist (namespace, added_at, reason, count, last_seen)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (ns, now, "legacy_metadata", 0, now),
                        )
                    conn.commit()
                return namespaces
            except Exception as e:
                logger.warning(
                    "Failed to parse dynamic blacklist from metadata; resetting. Error: %s",
                    e,
                    exc_info=True,
                )
        return set()

    def get_dynamic_blacklist(self) -> set[str]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT namespace FROM dynamic_blacklist")
        rows = cursor.fetchall()
        if rows:
            return {row["namespace"] for row in rows}
        return self._migrate_dynamic_blacklist_metadata()

    def upsert_dynamic_blacklist(self, additions: dict[str, int], reason: str) -> None:
        if not additions:
            return
        conn = self.get_connection()
        cursor = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        for ns, count in additions.items():
            if not ns:
                continue
            cursor.execute(
                """
                INSERT INTO dynamic_blacklist (namespace, added_at, reason, count, last_seen)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace) DO UPDATE SET
                    reason=excluded.reason,
                    count=MAX(dynamic_blacklist.count, excluded.count),
                    last_seen=excluded.last_seen,
                    added_at=COALESCE(dynamic_blacklist.added_at, excluded.added_at)
                """,
                (ns, now, reason, int(count or 0), now),
            )
        conn.commit()

    def prune_dynamic_blacklist(self, cutoff_dt: datetime) -> set[str]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT namespace FROM dynamic_blacklist WHERE last_seen < ?",
            (cutoff_dt.isoformat(),),
        )
        rows = cursor.fetchall()
        to_remove = {row["namespace"] for row in rows}
        if to_remove:
            cursor.execute(
                "DELETE FROM dynamic_blacklist WHERE last_seen < ?",
                (cutoff_dt.isoformat(),),
            )
            conn.commit()
        return to_remove

    def remove_dynamic_blacklist(self, namespaces: set[str]) -> set[str]:
        if not namespaces:
            return set()
        ns_list = sorted({n for n in namespaces if n})
        if not ns_list:
            return set()
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"DELETE FROM dynamic_blacklist WHERE namespace IN ({','.join('?' for _ in ns_list)})",
            tuple(ns_list),
        )
        conn.commit()
        return set(ns_list)

    def get_dynamic_whitelist(self) -> set[str]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT namespace FROM dynamic_whitelist")
        rows = cursor.fetchall()
        return {row["namespace"] for row in rows}

    def upsert_dynamic_whitelist(self, additions: dict[str, int], reason: str) -> None:
        if not additions:
            return
        conn = self.get_connection()
        cursor = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        for ns, count in additions.items():
            if not ns:
                continue
            cursor.execute(
                """
                INSERT INTO dynamic_whitelist (namespace, added_at, reason, count, last_seen)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace) DO UPDATE SET
                    reason=excluded.reason,
                    count=MAX(dynamic_whitelist.count, excluded.count),
                    last_seen=excluded.last_seen,
                    added_at=COALESCE(dynamic_whitelist.added_at, excluded.added_at)
                """,
                (ns, now, reason, int(count or 0), now),
            )
        conn.commit()

    def prune_dynamic_whitelist(self, cutoff_dt: datetime) -> set[str]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT namespace FROM dynamic_whitelist WHERE last_seen < ?",
            (cutoff_dt.isoformat(),),
        )
        rows = cursor.fetchall()
        to_remove = {row["namespace"] for row in rows}
        if to_remove:
            cursor.execute(
                "DELETE FROM dynamic_whitelist WHERE last_seen < ?",
                (cutoff_dt.isoformat(),),
            )
            conn.commit()
        return to_remove

    def remove_dynamic_whitelist(self, namespaces: set[str]) -> set[str]:
        if not namespaces:
            return set()
        ns_list = sorted({n for n in namespaces if n})
        if not ns_list:
            return set()
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"DELETE FROM dynamic_whitelist WHERE namespace IN ({','.join('?' for _ in ns_list)})",
            tuple(ns_list),
        )
        conn.commit()
        return set(ns_list)

    def get_author(self, namespace: str):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM authors WHERE namespace = ?", (namespace,))
        return cursor.fetchone()

    def upsert_author(self, data: dict) -> None:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO authors
              (namespace, kind, num_followers, is_pro, created_at, last_checked, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["namespace"],
                data["kind"],
                data.get("num_followers"),
                data.get("is_pro"),
                data.get("created_at"),
                datetime.now(timezone.utc).isoformat(),
                json.dumps(data.get("raw_json")),
            ),
        )
        conn.commit()

    def save_model(self, model_data: dict) -> None:
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO models (
                    id, name, author, created_at, last_modified,
                    params_est, params_total_b, params_active_b, params_source,
                    hf_tags, llm_analysis, status,
                    namespace, author_kind, trust_tier, pipeline_tag, filter_trace, report_notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_data["id"],
                    model_data["name"],
                    model_data["author"],
                    model_data["created_at"],
                    model_data.get("last_modified"),
                    model_data.get("params_est"),
                    model_data.get("params_total_b"),
                    model_data.get("params_active_b"),
                    model_data.get("params_source"),
                    json.dumps(model_data.get("hf_tags", [])),
                    json.dumps(model_data.get("llm_analysis", {})),
                    model_data.get("status", "processed"),
                    model_data.get("namespace"),
                    model_data.get("author_kind"),
                    model_data.get("trust_tier"),
                    model_data.get("pipeline_tag"),
                    json.dumps(model_data.get("filter_trace", [])),
                    model_data.get("report_notes"),
                ),
            )
            conn.commit()
        except Exception as e:
            logger.error(f"Error saving model {model_data.get('id')}: {e}")

    def close(self) -> None:
        if self.conn:
            self.conn.close()
