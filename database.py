import sqlite3
import json
import logging
from datetime import datetime, timedelta, timezone
import dateutil.parser

class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.init_db()

    def get_connection(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        return self.conn

    def init_db(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        # 1. Models Table
        cursor.execute('''
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
        ''')

        # Migrations for 'models'
        self._ensure_column(cursor, 'models', 'last_modified', 'TIMESTAMP')
        self._ensure_column(cursor, 'models', 'namespace', 'TEXT')
        self._ensure_column(cursor, 'models', 'author_kind', 'TEXT')
        self._ensure_column(cursor, 'models', 'trust_tier', 'INTEGER')
        self._ensure_column(cursor, 'models', 'pipeline_tag', 'TEXT')
        self._ensure_column(cursor, 'models', 'filter_trace', 'TEXT')
        self._ensure_column(cursor, 'models', 'report_notes', 'TEXT')

        # 2. Metadata Table (App State)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')

        # 3. Authors Table (Cache)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS authors (
                namespace TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                num_followers INTEGER,
                is_pro INTEGER,
                created_at TIMESTAMP,
                last_checked TIMESTAMP NOT NULL,
                raw_json TEXT
            )
        ''')

        conn.commit()

    def _ensure_column(self, cursor, table, column, col_type):
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [row['name'] for row in cursor.fetchall()]
        if column not in columns:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

    def get_existing_ids(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM models")
        rows = cursor.fetchall()
        return {row['id'] for row in rows}

    def get_model_last_modified(self, model_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT last_modified FROM models WHERE id = ?", (model_id,))
        row = cursor.fetchone()
        if row and row['last_modified']:
            return row['last_modified']
        return None

    def get_last_run_timestamp(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM metadata WHERE key = 'last_run'")
        row = cursor.fetchone()

        if row and row['value']:
            try:
                return dateutil.parser.parse(row['value'])
            except Exception:
                pass
        return datetime.now(timezone.utc) - timedelta(hours=24)

    def set_last_run_timestamp(self, timestamp):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('last_run', ?)", (timestamp.isoformat(),))
        conn.commit()

    def get_author(self, namespace):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM authors WHERE namespace = ?", (namespace,))
        return cursor.fetchone()

    def upsert_author(self, data):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO authors (namespace, kind, num_followers, is_pro, created_at, last_checked, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['namespace'],
            data['kind'],
            data.get('num_followers'),
            data.get('is_pro'),
            data.get('created_at'),
            datetime.now(timezone.utc),
            json.dumps(data.get('raw_json'))
        ))
        conn.commit()

    def save_model(self, model_data):
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO models (
                    id, name, author, created_at, last_modified, params_est, hf_tags, llm_analysis, status,
                    namespace, author_kind, trust_tier, pipeline_tag, filter_trace, report_notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_data['id'],
                model_data['name'],
                model_data['author'],
                model_data['created_at'],
                model_data.get('last_modified'),
                model_data.get('params_est'),
                json.dumps(model_data.get('hf_tags', [])),
                json.dumps(model_data.get('llm_analysis', {})),
                model_data.get('status', 'processed'),
                model_data.get('namespace'),
                model_data.get('author_kind'),
                model_data.get('trust_tier'),
                model_data.get('pipeline_tag'),
                json.dumps(model_data.get('filter_trace', [])),
                model_data.get('report_notes')
            ))
            conn.commit()
        except Exception as e:
            logging.error(f"Error saving model {model_data.get('id')}: {e}")

    def close(self):
        if self.conn:
            self.conn.close()
