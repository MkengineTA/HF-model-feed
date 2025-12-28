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

        # Check if last_modified exists (migration hack for dev)
        cursor.execute("PRAGMA table_info(models)")
        columns = [row['name'] for row in cursor.fetchall()]
        if 'last_modified' not in columns:
            cursor.execute("ALTER TABLE models ADD COLUMN last_modified TIMESTAMP")

        # Metadata table for app state (last_run)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')

        conn.commit()

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
        """
        Returns the last run timestamp.
        If not set, returns 24 hours ago (default for first run).
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM metadata WHERE key = 'last_run'")
        row = cursor.fetchone()

        if row and row['value']:
            try:
                return dateutil.parser.parse(row['value'])
            except Exception:
                pass

        # Default: 24h ago
        return datetime.now(timezone.utc) - timedelta(hours=24)

    def set_last_run_timestamp(self, timestamp):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('last_run', ?)", (timestamp.isoformat(),))
        conn.commit()

    def save_model(self, model_data):
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO models (id, name, author, created_at, last_modified, params_est, hf_tags, llm_analysis, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_data['id'],
                model_data['name'],
                model_data['author'],
                model_data['created_at'],
                model_data.get('last_modified'),
                model_data.get('params_est'),
                json.dumps(model_data.get('hf_tags', [])),
                json.dumps(model_data.get('llm_analysis', {})),
                model_data.get('status', 'processed')
            ))
            conn.commit()
        except Exception as e:
            logging.error(f"Error saving model {model_data.get('id')}: {e}")

    def update_model_status(self, model_id, status):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE models SET status = ? WHERE id = ?", (status, model_id))
        conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()
