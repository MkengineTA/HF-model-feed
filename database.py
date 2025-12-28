import sqlite3
import json
import logging
from datetime import datetime

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

        # Added last_modified column
        # Note: If DB exists, migration is needed. Since this is a dev environment/new tool,
        # we can assume recreation or user deletes old db.
        # To be safe for existing users (in this session), I'll add logic to check columns or just create if not exists.
        # But for 'create if not exists', it won't add column if table exists.
        # Let's try to add column via ALTER TABLE if it's missing (simple migration).

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

        conn.commit()

    def get_existing_ids(self):
        """
        Returns a set of all model IDs in the database.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM models")
        rows = cursor.fetchall()
        return {row['id'] for row in rows}

    def get_model_last_modified(self, model_id):
        """
        Returns the last_modified timestamp for a model if it exists, else None.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT last_modified FROM models WHERE id = ?", (model_id,))
        row = cursor.fetchone()
        if row and row['last_modified']:
            # Return as string or datetime?
            # Storing as text usually (SQLite default for TIMESTAMP often).
            # Let's return the raw value, calling code can parse.
            return row['last_modified']
        return None

    def save_model(self, model_data):
        """
        model_data should be a dict matching the columns.
        hf_tags and llm_analysis should be passed as dicts/lists and will be JSON serialized here.
        """
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
