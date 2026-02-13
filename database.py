"""
Database module for metadata logging and tracking.
Supports SQLite and PostgreSQL for storing pipeline metadata.
"""

import sqlite3
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from config import DatabaseConfig, default_config

logger = logging.getLogger(__name__)


@dataclass
class IngestionLog:
    """Log entry for data ingestion."""
    ingestion_id: str
    source_path: str
    rows_ingested: int
    columns: List[str]
    data_hash: str
    timestamp: str
    status: str
    error_message: Optional[str] = None


@dataclass
class FeatureLog:
    """Log entry for feature engineering."""
    feature_id: str
    ingestion_id: str
    transformations_applied: List[str]
    features_created: int
    features_dropped: int
    timestamp: str
    status: str
    error_message: Optional[str] = None


@dataclass
class DriftLog:
    """Log entry for drift detection."""
    drift_id: str
    reference_ingestion_id: str
    current_ingestion_id: str
    columns_tested: List[str]
    drift_detected: bool
    drift_columns: List[str]
    ks_statistics: Dict[str, float]
    p_values: Dict[str, float]
    timestamp: str
    status: str


@dataclass
class ModelLog:
    """Log entry for model training."""
    model_id: str
    feature_id: str
    model_type: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    model_path: str
    timestamp: str
    is_active: bool
    status: str


class MetadataDatabase:
    """Database handler for pipeline metadata logging."""
    
    def __init__(self, config: DatabaseConfig = None):
        self.config = config or default_config.database
        self._connection = None
        self._initialize_database()
    
    def _get_connection(self):
        """Get database connection."""
        if self._connection is None:
            if self.config.db_type == "sqlite":
                db_path = Path(self.config.db_name)
                db_path.parent.mkdir(parents=True, exist_ok=True)
                self._connection = sqlite3.connect(
                    str(db_path), 
                    check_same_thread=False
                )
                self._connection.row_factory = sqlite3.Row
            else:
                # PostgreSQL connection
                import psycopg2
                self._connection = psycopg2.connect(
                    host=self.config.db_host,
                    port=self.config.db_port,
                    database=self.config.db_name,
                    user=self.config.db_user,
                    password=self.config.db_password
                )
        return self._connection
    
    def _initialize_database(self):
        """Create tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Ingestion logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ingestion_logs (
                ingestion_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                rows_ingested INTEGER,
                columns TEXT,
                data_hash TEXT,
                timestamp TEXT,
                status TEXT,
                error_message TEXT
            )
        ''')
        
        # Feature engineering logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_logs (
                feature_id TEXT PRIMARY KEY,
                ingestion_id TEXT,
                transformations_applied TEXT,
                features_created INTEGER,
                features_dropped INTEGER,
                timestamp TEXT,
                status TEXT,
                error_message TEXT,
                FOREIGN KEY (ingestion_id) REFERENCES ingestion_logs(ingestion_id)
            )
        ''')
        
        # Drift detection logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drift_logs (
                drift_id TEXT PRIMARY KEY,
                reference_ingestion_id TEXT,
                current_ingestion_id TEXT,
                columns_tested TEXT,
                drift_detected INTEGER,
                drift_columns TEXT,
                ks_statistics TEXT,
                p_values TEXT,
                timestamp TEXT,
                status TEXT,
                FOREIGN KEY (reference_ingestion_id) REFERENCES ingestion_logs(ingestion_id),
                FOREIGN KEY (current_ingestion_id) REFERENCES ingestion_logs(ingestion_id)
            )
        ''')
        
        # Model logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_logs (
                model_id TEXT PRIMARY KEY,
                feature_id TEXT,
                model_type TEXT,
                hyperparameters TEXT,
                metrics TEXT,
                model_path TEXT,
                timestamp TEXT,
                is_active INTEGER,
                status TEXT,
                FOREIGN KEY (feature_id) REFERENCES feature_logs(feature_id)
            )
        ''')
        
        # Reference data statistics table (for drift comparison)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reference_stats (
                stat_id TEXT PRIMARY KEY,
                ingestion_id TEXT,
                column_name TEXT,
                mean REAL,
                std REAL,
                min_val REAL,
                max_val REAL,
                sample_data TEXT,
                timestamp TEXT,
                FOREIGN KEY (ingestion_id) REFERENCES ingestion_logs(ingestion_id)
            )
        ''')
        
        conn.commit()
        logger.info("Database initialized successfully")
    
    def log_ingestion(self, log: IngestionLog) -> None:
        """Log data ingestion event."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO ingestion_logs 
            (ingestion_id, source_path, rows_ingested, columns, data_hash, timestamp, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            log.ingestion_id,
            log.source_path,
            log.rows_ingested,
            json.dumps(log.columns),
            log.data_hash,
            log.timestamp,
            log.status,
            log.error_message
        ))
        conn.commit()
        logger.info(f"Logged ingestion: {log.ingestion_id}")
    
    def log_feature_engineering(self, log: FeatureLog) -> None:
        """Log feature engineering event."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO feature_logs 
            (feature_id, ingestion_id, transformations_applied, features_created, 
             features_dropped, timestamp, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            log.feature_id,
            log.ingestion_id,
            json.dumps(log.transformations_applied),
            log.features_created,
            log.features_dropped,
            log.timestamp,
            log.status,
            log.error_message
        ))
        conn.commit()
        logger.info(f"Logged feature engineering: {log.feature_id}")
    
    def log_drift(self, log: DriftLog) -> None:
        """Log drift detection event."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO drift_logs 
            (drift_id, reference_ingestion_id, current_ingestion_id, columns_tested,
             drift_detected, drift_columns, ks_statistics, p_values, timestamp, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            log.drift_id,
            log.reference_ingestion_id,
            log.current_ingestion_id,
            json.dumps(log.columns_tested),
            1 if log.drift_detected else 0,
            json.dumps(log.drift_columns),
            json.dumps(log.ks_statistics),
            json.dumps(log.p_values),
            log.timestamp,
            log.status
        ))
        conn.commit()
        logger.info(f"Logged drift detection: {log.drift_id}")
    
    def log_model(self, log: ModelLog) -> None:
        """Log model training event."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # If this model is active, deactivate other models
        if log.is_active:
            cursor.execute('UPDATE model_logs SET is_active = 0')
        
        cursor.execute('''
            INSERT OR REPLACE INTO model_logs 
            (model_id, feature_id, model_type, hyperparameters, metrics, 
             model_path, timestamp, is_active, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            log.model_id,
            log.feature_id,
            log.model_type,
            json.dumps(log.hyperparameters),
            json.dumps(log.metrics),
            log.model_path,
            log.timestamp,
            1 if log.is_active else 0,
            log.status
        ))
        conn.commit()
        logger.info(f"Logged model: {log.model_id}")
    
    def store_reference_stats(self, ingestion_id: str, column_name: str, 
                              mean: float, std: float, min_val: float, 
                              max_val: float, sample_data: List[float]) -> None:
        """Store reference statistics for drift comparison."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        stat_id = hashlib.md5(f"{ingestion_id}_{column_name}".encode()).hexdigest()[:16]
        
        cursor.execute('''
            INSERT OR REPLACE INTO reference_stats 
            (stat_id, ingestion_id, column_name, mean, std, min_val, max_val, sample_data, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            stat_id,
            ingestion_id,
            column_name,
            mean,
            std,
            min_val,
            max_val,
            json.dumps(sample_data[:1000]),  # Store max 1000 samples
            datetime.utcnow().isoformat()
        ))
        conn.commit()
    
    def get_reference_stats(self, ingestion_id: str) -> Dict[str, Dict[str, Any]]:
        """Get reference statistics for a given ingestion."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT column_name, mean, std, min_val, max_val, sample_data 
            FROM reference_stats WHERE ingestion_id = ?
        ''', (ingestion_id,))
        
        results = {}
        for row in cursor.fetchall():
            results[row['column_name']] = {
                'mean': row['mean'],
                'std': row['std'],
                'min_val': row['min_val'],
                'max_val': row['max_val'],
                'sample_data': json.loads(row['sample_data'])
            }
        return results
    
    def get_latest_ingestion_id(self) -> Optional[str]:
        """Get the most recent ingestion ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT ingestion_id FROM ingestion_logs 
            WHERE status = 'success' 
            ORDER BY timestamp DESC LIMIT 1
        ''')
        
        row = cursor.fetchone()
        return row['ingestion_id'] if row else None
    
    def get_active_model(self) -> Optional[Dict[str, Any]]:
        """Get the currently active model."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM model_logs WHERE is_active = 1
        ''')
        
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def get_drift_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent drift detection history."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM drift_logs ORDER BY timestamp DESC LIMIT ?
        ''', (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with timestamp."""
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    hash_part = hashlib.md5(str(datetime.utcnow().timestamp()).encode()).hexdigest()[:8]
    return f"{prefix}{timestamp}_{hash_part}"
