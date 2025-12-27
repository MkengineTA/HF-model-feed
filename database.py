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

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                name TEXT,
                author TEXT,
                created_at TIMESTAMP,
                params_est REAL,
                hf_tags TEXT,
                llm_analysis TEXT,
                user_label INTEGER DEFAULT NULL,
                status TEXT
            )
        ''')
        conn.commit()

    def get_existing_ids(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM models")
        rows = cursor.fetchall()
        return {row['id'] for row in rows}

    def save_model(self, model_data):
        """
        model_data should be a dict matching the columns.
        hf_tags and llm_analysis should be passed as dicts/lists and will be JSON serialized here.
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO models (id, name, author, created_at, params_est, hf_tags, llm_analysis, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_data['id'],
                model_data['name'],
                model_data['author'],
                model_data['created_at'],
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
