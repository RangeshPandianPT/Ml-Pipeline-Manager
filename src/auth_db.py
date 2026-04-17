"""
Database handler for user authentication and security.
"""

import sqlite3
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class AuthDatabase:
    """Database handler for authentication."""

    def __init__(self, db_path: str = "auth_metadata.db"):
        self.db_path = Path(db_path)
        self._connection = None
        self._initialize_database()

    def _get_connection(self):
        if self._connection is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def _initialize_database(self):
        """Create users table and seed initial admin user."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                full_name TEXT,
                email TEXT,
                hashed_password TEXT NOT NULL,
                disabled INTEGER DEFAULT 0,
                created_at TEXT
            )
        ''')
        conn.commit()
        logger.info("Auth Database initialized successfully")

    def seed_admin_user(self, hashed_password: str):
        """Seeds an admin user if no users exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as count FROM users")
        if cursor.fetchone()["count"] == 0:
            cursor.execute('''
                INSERT INTO users (username, full_name, email, hashed_password, disabled, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                "admin",
                "System Admin",
                "admin@example.com",
                hashed_password,
                0,
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()
            logger.info("Default admin user seeded successfully")

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Retrieve a user by username."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        
        if row:
            # Convert disabled integer to boolean
            user_data = dict(row)
            user_data["disabled"] = bool(user_data["disabled"])
            return user_data
        return None

    def close(self):
        if self._connection:
            self._connection.close()
            self._connection = None
