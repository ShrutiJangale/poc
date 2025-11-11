import datetime
import sqlite3
import json
import os
from pathlib import Path

# Get the project root directory (supplychain_ai/)
# This file is in: supplychain_ai/analysis_supplychain_ai/utils/db_utils.py
# So we go up 3 levels to get to supplychain_ai/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = str(PROJECT_ROOT / "supplyChain.db")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_files_table(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS files (
            file_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            status TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )
    cur.execute(
        '''
        CREATE TRIGGER IF NOT EXISTS trg_files_updated
        AFTER UPDATE ON files
        FOR EACH ROW
        BEGIN
            UPDATE files SET updated_at = CURRENT_TIMESTAMP WHERE file_id = NEW.file_id;
        END;
        '''
    )
    # Minimal migration: if legacy 'requests' table exists and 'files' is empty, copy rows
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='requests'")
    if cur.fetchone():
        cur.execute("SELECT COUNT(1) AS c FROM files")
        if cur.fetchone()[0] == 0:
            # Add file_name to legacy table if missing
            try:
                cur.execute('ALTER TABLE requests ADD COLUMN file_name TEXT')
            except sqlite3.OperationalError:
                pass
            # Copy data: request_id -> file_id, status, file_name, timestamps
            try:
                cur.execute(
                    "INSERT INTO files (file_id, file_name, status, created_at, updated_at) "
                    "SELECT request_id, file_name, status, created_at, updated_at FROM requests"
                )
            except Exception:
                # Fallback if timestamps missing
                cur.execute(
                    "INSERT INTO files (file_id, file_name, status) SELECT request_id, file_name, status FROM requests"
                )
    conn.commit()


def create_schema():
    """Create database schema if it doesn't exist."""
    # Ensure the directory exists
    db_dir = Path(DB_PATH).parent
    db_dir.mkdir(parents=True, exist_ok=True)
    
    conn = get_db_connection()
    cur = conn.cursor()

    # Recreate dependent tables (non-destructive create)
    _ensure_files_table(conn)

    cur.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id INTEGER NOT NULL,
            category_name TEXT,
            log_text TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (request_id) REFERENCES files(file_id)
        )
    ''')

    cur.execute('''
        CREATE TABLE IF NOT EXISTS summaries (
            summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id INTEGER NOT NULL,
            category_name TEXT,
            summary_type TEXT CHECK(summary_type IN ('overall','questionwise')),
            summary_text TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (request_id) REFERENCES files(file_id)
        )
    ''')

    cur.execute('''
        CREATE TRIGGER IF NOT EXISTS trg_summaries_updated
        AFTER UPDATE ON summaries
        FOR EACH ROW
        BEGIN
            UPDATE summaries SET updated_at = CURRENT_TIMESTAMP WHERE summary_id = NEW.summary_id;
        END;
    ''')

    conn.commit()
    conn.close()


def create_file(status: str = "uploaded", file_name: str | None = None) -> int:
    conn = get_db_connection()
    _ensure_files_table(conn)
    cur = conn.cursor()
    cur.execute('INSERT INTO files (status, file_name) VALUES (?, ?)', (status, file_name))
    file_id = cur.lastrowid
    conn.commit()
    conn.close()
    return file_id


def get_all_files():
    conn = get_db_connection()
    _ensure_files_table(conn)
    rows = conn.execute(
        'SELECT file_id, file_name, status, created_at FROM files ORDER BY created_at DESC'
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_file_status(file_id: int, status: str):
    conn = get_db_connection()
    _ensure_files_table(conn)
    conn.execute('UPDATE files SET status = ? WHERE file_id = ?', (status, file_id))
    conn.commit()
    conn.close()


def insert_logs(file_id: int, logs):
    if not logs:
        return
    conn = get_db_connection()
    _ensure_files_table(conn)
    cur = conn.cursor()
    cur.executemany(
        'INSERT INTO logs (request_id, category_name, log_text) VALUES (?, ?, ?)',
        [(file_id, log.get('category_name'), log['log_text']) for log in logs]
    )
    conn.commit()
    conn.close()


def upsert_summary(file_id: int, summary_type: str, summary_text: str, category_name: str | None = None):
    conn = get_db_connection()
    _ensure_files_table(conn)
    cur = conn.cursor()
    if category_name is None:
        cur.execute('SELECT summary_id FROM summaries WHERE request_id=? AND summary_type=? AND category_name IS NULL',
                    (file_id, summary_type))
    else:
        cur.execute('SELECT summary_id FROM summaries WHERE request_id=? AND summary_type=? AND category_name=?',
                    (file_id, summary_type, category_name))
    row = cur.fetchone()
    if row:
        cur.execute('UPDATE summaries SET summary_text=? WHERE summary_id=?', (summary_text, row['summary_id']))
    else:
        cur.execute('INSERT INTO summaries (request_id, category_name, summary_type, summary_text) VALUES (?, ?, ?, ?)',
                    (file_id, category_name, summary_type, summary_text))
    conn.commit()
    conn.close()
