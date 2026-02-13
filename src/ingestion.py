"""
Data Ingestion Module for ML Pipeline.
Handles data loading from various sources with validation and logging.
Supports mock weather API (JSON), local CSV, and multiple formats.
Domain-agnostic implementation suitable for any dataset.
"""

import pandas as pd
import numpy as np
import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Callable, Type
from dataclasses import dataclass

from .config import PipelineConfig, default_config
from .database import MetadataDatabase, IngestionLog, generate_id

logger = logging.getLogger(__name__)


# =============================================================================
# Schema Definition for Type-Hinted Validation
# =============================================================================

@dataclass
class ColumnSchema:
    """Schema definition for a single column with Python type hints."""
    name: str
    dtype: Type          # Expected Python/numpy type (int, float, str, etc.)
    nullable: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None

    def validate(self, series: pd.Series) -> List[str]:
        """Validate a pandas Series against this column schema."""
        errors: List[str] = []
        if not self.nullable and series.isnull().any():
            errors.append(f"Column '{self.name}' has {series.isnull().sum()} null values but is non-nullable")
        if self.min_value is not None:
            below = (series.dropna() < self.min_value).sum()
            if below > 0:
                errors.append(f"Column '{self.name}' has {below} values below minimum {self.min_value}")
        if self.max_value is not None:
            above = (series.dropna() > self.max_value).sum()
            if above > 0:
                errors.append(f"Column '{self.name}' has {above} values above maximum {self.max_value}")
        if self.allowed_values is not None:
            invalid = ~series.dropna().isin(self.allowed_values)
            if invalid.any():
                errors.append(f"Column '{self.name}' has {invalid.sum()} values outside allowed set")
        return errors


@dataclass
class DatasetSchema:
    """Full schema definition for a dataset."""
    columns: List[ColumnSchema]

    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]


# =============================================================================
# Mock Weather API
# =============================================================================

class MockWeatherAPI:
    """
    Simulates pulling data from a weather API in JSON format.
    Generates realistic weather observations with temperature, humidity,
    wind speed, pressure, and precipitation.
    """

    def __init__(self, n_records: int = 500, seed: int = 42):
        self.n_records: int = n_records
        self.seed: int = seed

    def fetch(self) -> pd.DataFrame:
        """
        Fetch weather data from the mock API.

        Returns:
            DataFrame with columns: timestamp, temperature_c, humidity_pct,
            wind_speed_kmh, pressure_hpa, precipitation_mm, weather_condition
        """
        np.random.seed(self.seed)
        base_time = datetime.utcnow() - timedelta(days=self.n_records)
        timestamps = [base_time + timedelta(hours=i) for i in range(self.n_records)]

        data = {
            "timestamp": timestamps,
            "temperature_c": np.round(np.random.normal(22, 8, self.n_records), 1),
            "humidity_pct": np.clip(np.random.normal(60, 15, self.n_records), 5, 100).round(1),
            "wind_speed_kmh": np.abs(np.random.normal(15, 7, self.n_records)).round(1),
            "pressure_hpa": np.random.normal(1013, 10, self.n_records).round(1),
            "precipitation_mm": np.abs(np.random.exponential(2, self.n_records)).round(2),
            "weather_condition": np.random.choice(
                ["sunny", "cloudy", "rainy", "stormy", "foggy"],
                size=self.n_records,
                p=[0.35, 0.30, 0.20, 0.05, 0.10],
            ),
        }

        # Inject ~3 % missing values for realism
        df = pd.DataFrame(data)
        for col in ["temperature_c", "humidity_pct", "wind_speed_kmh"]:
            mask = np.random.random(self.n_records) < 0.03
            df.loc[mask, col] = np.nan

        logger.info(f"MockWeatherAPI: fetched {self.n_records} records")
        return df

    def fetch_json(self) -> str:
        """Return the mock data as a JSON string (simulating an API response)."""
        df = self.fetch()
        df["timestamp"] = df["timestamp"].astype(str)
        return df.to_json(orient="records", indent=2)


# =============================================================================
# Data Quality Report
# =============================================================================

@dataclass
class DataQualityReport:
    """Report on data quality after ingestion."""
    total_rows: int
    total_columns: int
    missing_values: Dict[str, int]
    missing_percentages: Dict[str, float]
    duplicate_rows: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    data_types: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_rows': self.total_rows,
            'total_columns': self.total_columns,
            'missing_values': self.missing_values,
            'missing_percentages': self.missing_percentages,
            'duplicate_rows': self.duplicate_rows,
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns,
            'datetime_columns': self.datetime_columns,
            'data_types': self.data_types,
        }


# =============================================================================
# DataValidator — full schema-consistency & missing-value checks
# =============================================================================

class DataValidator:
    """
    Validates ingested data for quality, missing values, and schema consistency.
    Uses Python type hints via ColumnSchema / DatasetSchema for strict checks.
    """

    @staticmethod
    def validate_schema(df: pd.DataFrame, expected_columns: List[str] = None) -> bool:
        """Validate that DataFrame has expected columns."""
        if expected_columns is None:
            return True
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            logger.warning(f"Missing expected columns: {missing_cols}")
            return False
        return True

    @staticmethod
    def validate_schema_strict(df: pd.DataFrame, schema: DatasetSchema) -> Dict[str, Any]:
        """
        Validate a DataFrame against a full DatasetSchema.

        Returns:
            dict with keys: 'valid' (bool), 'missing_columns', 'extra_columns',
            'column_errors' (per-column error list).
        """
        expected_names = set(schema.column_names())
        actual_names = set(df.columns)
        missing = expected_names - actual_names
        extra = actual_names - expected_names
        column_errors: Dict[str, List[str]] = {}

        for col_schema in schema.columns:
            if col_schema.name in df.columns:
                errs = col_schema.validate(df[col_schema.name])
                if errs:
                    column_errors[col_schema.name] = errs

        is_valid = len(missing) == 0 and len(column_errors) == 0
        result = {
            "valid": is_valid,
            "missing_columns": list(missing),
            "extra_columns": list(extra),
            "column_errors": column_errors,
        }
        if not is_valid:
            logger.warning(f"Schema validation failed: {result}")
        else:
            logger.info("Schema validation passed")
        return result

    @staticmethod
    def check_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detailed missing-value analysis.

        Returns:
            dict with total_missing, per-column counts, and percentages.
        """
        counts = df.isnull().sum()
        pcts = (counts / len(df) * 100).round(2)
        return {
            "total_missing": int(counts.sum()),
            "columns_with_missing": {
                col: {"count": int(counts[col]), "pct": float(pcts[col])}
                for col in counts.index
                if counts[col] > 0
            },
        }

    @staticmethod
    def validate_not_empty(df: pd.DataFrame) -> bool:
        """Validate that DataFrame is not empty."""
        if df.empty:
            logger.warning("DataFrame is empty")
            return False
        return True

    @staticmethod
    def validate_no_all_null_columns(df: pd.DataFrame) -> bool:
        """Validate that no columns are entirely null."""
        all_null_cols = df.columns[df.isnull().all()].tolist()
        if all_null_cols:
            logger.warning(f"Columns with all null values: {all_null_cols}")
            return False
        return True

    @staticmethod
    def generate_quality_report(df: pd.DataFrame) -> DataQualityReport:
        """Generate a comprehensive data quality report."""
        missing_values = df.isnull().sum().to_dict()
        missing_percentages = (df.isnull().sum() / len(df) * 100).to_dict()

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        return DataQualityReport(
            total_rows=len(df),
            total_columns=len(df.columns),
            missing_values=missing_values,
            missing_percentages=missing_percentages,
            duplicate_rows=df.duplicated().sum(),
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            datetime_columns=datetime_cols,
            data_types={col: str(dtype) for col, dtype in df.dtypes.items()},
        )


class DataIngestion:
    """Handles data ingestion from various sources including mock APIs and local files."""

    def __init__(self, config: PipelineConfig = None, db: MetadataDatabase = None):
        self.config = config or default_config
        self.db = db or MetadataDatabase(self.config.database)
        self.validator = DataValidator()
        self._current_ingestion_id: Optional[str] = None
        self._current_data: Optional[pd.DataFrame] = None
        self._quality_report: Optional[DataQualityReport] = None

        # Create landing folder for raw data storage
        self.landing_path: Path = Path(self.config.data_dir) / "landing"
        self.landing_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 2 — pull from mock weather API + local CSV, validate, save
    # ------------------------------------------------------------------

    def ingest_from_sources(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        api: Optional[MockWeatherAPI] = None,
        schema: Optional[DatasetSchema] = None,
    ) -> pd.DataFrame:
        """
        Pull data from a mock weather API (JSON) and/or a local CSV,
        validate with DataValidator (including schema consistency),
        and save the raw data to the *landing* folder.

        Args:
            csv_path: Path to a local CSV file (optional).
            api: A MockWeatherAPI instance (optional; one is created if None).
            schema: Optional DatasetSchema for strict validation.

        Returns:
            Combined DataFrame after validation.
        """
        frames: List[pd.DataFrame] = []

        # 1. Pull from mock weather API
        weather_api = api or MockWeatherAPI()
        api_df = weather_api.fetch()
        logger.info(f"Pulled {len(api_df)} rows from mock weather API")
        frames.append(api_df)

        # 2. Pull from local CSV if provided
        if csv_path is not None:
            csv_path = Path(csv_path)
            if csv_path.exists():
                csv_df = pd.read_csv(csv_path)
                logger.info(f"Loaded {len(csv_df)} rows from CSV: {csv_path}")
                frames.append(csv_df)
            else:
                logger.warning(f"CSV file not found: {csv_path}")

        # 3. Combine
        combined_df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

        # 4. Validate — missing values & schema consistency
        if not self.validator.validate_not_empty(combined_df):
            raise ValueError("Ingested data is empty after combining sources")

        self.validator.validate_no_all_null_columns(combined_df)
        missing_report = self.validator.check_missing_values(combined_df)
        logger.info(f"Missing-value check: {missing_report['total_missing']} total missing")

        if schema is not None:
            schema_result = self.validator.validate_schema_strict(combined_df, schema)
            if not schema_result["valid"]:
                logger.warning(f"Schema issues detected — continuing with available data")

        # 5. Save raw data to landing folder
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        landing_file = self.landing_path / f"raw_data_{ts}.csv"
        combined_df.to_csv(landing_file, index=False)
        logger.info(f"Raw data saved to landing folder: {landing_file}")

        # Also save the API JSON snapshot
        json_file = self.landing_path / f"api_snapshot_{ts}.json"
        with open(json_file, "w") as f:
            f.write(weather_api.fetch_json())

        # 6. Standard ingestion bookkeeping
        return self.ingest_dataframe(combined_df, source_name=f"multi_source_{ts}")
    
    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute a hash of the dataframe for change detection."""
        return hashlib.md5(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()
        ).hexdigest()
    
    def ingest_csv(self, 
                   file_path: Union[str, Path],
                   **pandas_kwargs) -> pd.DataFrame:
        """
        Ingest data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            **pandas_kwargs: Additional arguments for pd.read_csv
        
        Returns:
            DataFrame with ingested data
        """
        file_path = Path(file_path)
        ingestion_id = generate_id("ING_")
        
        try:
            logger.info(f"Starting ingestion from: {file_path}")
            
            # Read the CSV file
            df = pd.read_csv(file_path, **pandas_kwargs)
            
            # Validate data
            if not self.validator.validate_not_empty(df):
                raise ValueError("Ingested data is empty")
            
            self.validator.validate_no_all_null_columns(df)
            
            # Generate quality report
            self._quality_report = self.validator.generate_quality_report(df)
            
            # Compute data hash
            data_hash = self._compute_data_hash(df)
            
            # Log ingestion
            log = IngestionLog(
                ingestion_id=ingestion_id,
                source_path=str(file_path),
                rows_ingested=len(df),
                columns=df.columns.tolist(),
                data_hash=data_hash,
                timestamp=datetime.utcnow().isoformat(),
                status="success"
            )
            self.db.log_ingestion(log)
            
            # Store reference statistics for drift detection
            self._store_reference_stats(ingestion_id, df)
            
            self._current_ingestion_id = ingestion_id
            self._current_data = df
            
            logger.info(f"Successfully ingested {len(df)} rows with {len(df.columns)} columns")
            logger.info(f"Ingestion ID: {ingestion_id}")
            
            return df
            
        except Exception as e:
            logger.error(f"Ingestion failed: {str(e)}")
            
            # Log failure
            log = IngestionLog(
                ingestion_id=ingestion_id,
                source_path=str(file_path),
                rows_ingested=0,
                columns=[],
                data_hash="",
                timestamp=datetime.utcnow().isoformat(),
                status="failed",
                error_message=str(e)
            )
            self.db.log_ingestion(log)
            raise
    
    def ingest_dataframe(self, 
                         df: pd.DataFrame,
                         source_name: str = "dataframe") -> pd.DataFrame:
        """
        Ingest data from an existing DataFrame.
        
        Args:
            df: DataFrame to ingest
            source_name: Name to identify this data source
        
        Returns:
            DataFrame (possibly modified during validation)
        """
        ingestion_id = generate_id("ING_")
        
        try:
            logger.info(f"Starting ingestion from DataFrame: {source_name}")
            
            # Validate data
            if not self.validator.validate_not_empty(df):
                raise ValueError("Ingested data is empty")
            
            self.validator.validate_no_all_null_columns(df)
            
            # Generate quality report
            self._quality_report = self.validator.generate_quality_report(df)
            
            # Compute data hash
            data_hash = self._compute_data_hash(df)
            
            # Log ingestion
            log = IngestionLog(
                ingestion_id=ingestion_id,
                source_path=source_name,
                rows_ingested=len(df),
                columns=df.columns.tolist(),
                data_hash=data_hash,
                timestamp=datetime.utcnow().isoformat(),
                status="success"
            )
            self.db.log_ingestion(log)
            
            # Store reference statistics
            self._store_reference_stats(ingestion_id, df)
            
            self._current_ingestion_id = ingestion_id
            self._current_data = df.copy()
            
            logger.info(f"Successfully ingested {len(df)} rows with {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            logger.error(f"Ingestion failed: {str(e)}")
            raise
    
    def ingest_json(self, 
                    file_path: Union[str, Path],
                    **pandas_kwargs) -> pd.DataFrame:
        """
        Ingest data from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            **pandas_kwargs: Additional arguments for pd.read_json
        
        Returns:
            DataFrame with ingested data
        """
        file_path = Path(file_path)
        ingestion_id = generate_id("ING_")
        
        try:
            logger.info(f"Starting JSON ingestion from: {file_path}")
            
            df = pd.read_json(file_path, **pandas_kwargs)
            
            if not self.validator.validate_not_empty(df):
                raise ValueError("Ingested data is empty")
            
            self._quality_report = self.validator.generate_quality_report(df)
            data_hash = self._compute_data_hash(df)
            
            log = IngestionLog(
                ingestion_id=ingestion_id,
                source_path=str(file_path),
                rows_ingested=len(df),
                columns=df.columns.tolist(),
                data_hash=data_hash,
                timestamp=datetime.utcnow().isoformat(),
                status="success"
            )
            self.db.log_ingestion(log)
            
            self._store_reference_stats(ingestion_id, df)
            
            self._current_ingestion_id = ingestion_id
            self._current_data = df
            
            logger.info(f"Successfully ingested {len(df)} rows")
            
            return df
            
        except Exception as e:
            logger.error(f"JSON ingestion failed: {str(e)}")
            raise
    
    def ingest_excel(self, 
                     file_path: Union[str, Path],
                     sheet_name: Union[str, int] = 0,
                     **pandas_kwargs) -> pd.DataFrame:
        """
        Ingest data from an Excel file.
        
        Args:
            file_path: Path to the Excel file
            sheet_name: Sheet name or index to read
            **pandas_kwargs: Additional arguments for pd.read_excel
        
        Returns:
            DataFrame with ingested data
        """
        file_path = Path(file_path)
        ingestion_id = generate_id("ING_")
        
        try:
            logger.info(f"Starting Excel ingestion from: {file_path}")
            
            df = pd.read_excel(file_path, sheet_name=sheet_name, **pandas_kwargs)
            
            if not self.validator.validate_not_empty(df):
                raise ValueError("Ingested data is empty")
            
            self._quality_report = self.validator.generate_quality_report(df)
            data_hash = self._compute_data_hash(df)
            
            log = IngestionLog(
                ingestion_id=ingestion_id,
                source_path=str(file_path),
                rows_ingested=len(df),
                columns=df.columns.tolist(),
                data_hash=data_hash,
                timestamp=datetime.utcnow().isoformat(),
                status="success"
            )
            self.db.log_ingestion(log)
            
            self._store_reference_stats(ingestion_id, df)
            
            self._current_ingestion_id = ingestion_id
            self._current_data = df
            
            logger.info(f"Successfully ingested {len(df)} rows")
            
            return df
            
        except Exception as e:
            logger.error(f"Excel ingestion failed: {str(e)}")
            raise
    
    def _store_reference_stats(self, ingestion_id: str, df: pd.DataFrame) -> None:
        """Store reference statistics for drift detection."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                self.db.store_reference_stats(
                    ingestion_id=ingestion_id,
                    column_name=col,
                    mean=float(col_data.mean()),
                    std=float(col_data.std()),
                    min_val=float(col_data.min()),
                    max_val=float(col_data.max()),
                    sample_data=col_data.tolist()
                )
    
    @property
    def ingestion_id(self) -> Optional[str]:
        """Get the current ingestion ID."""
        return self._current_ingestion_id
    
    @property
    def quality_report(self) -> Optional[DataQualityReport]:
        """Get the data quality report."""
        return self._quality_report
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """Get the currently loaded data."""
        return self._current_data


def create_sample_dataset(n_samples: int = 1000, n_features: int = 10) -> pd.DataFrame:
    """
    Create a generic sample dataset for testing.
    
    This generates a synthetic dataset with mixed feature types that works well
    for demonstrating the pipeline with any domain.
    
    Args:
        n_samples: Number of rows in the dataset
        n_features: Number of features (default 10)
    
    Returns:
        DataFrame with synthetic data
    """
    np.random.seed(42)
    
    data = {}
    
    # Generate numerical features
    for i in range(n_features):
        data[f'feature_{i+1}'] = np.random.normal(100, 20, n_samples)
    
    # Generate categorical features (commented out to simplify tests)
    # for i in range(int(n_features * 0.2)):
    #     data[f'category_{i+1}'] = np.random.choice(['A', 'B', 'C', 'D'], n_samples)
    
    df = pd.DataFrame(data)
    
    # Add some missing values for realism (but NOT in target)
    numeric_features = [col for col in df.select_dtypes(include=[np.number]).columns]
    for col in numeric_features:
        mask = np.random.random(n_samples) < 0.02
        df.loc[mask, col] = np.nan
    
    # Generate target variable AFTER adding missing values (so target has no NaN)
    y = np.zeros(n_samples)
    for col in [c for c in data.keys() if c.startswith('feature_')]:
        # Use fillna to handle any NaN in features when creating target
        y += df[col].fillna(df[col].mean()) * np.random.uniform(-1, 1)
    
    df['target'] = y + np.random.normal(0, 10, n_samples)
    
    return df


def create_sample_healthcare_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create sample domain-specific dataset for testing.
    
    Customize this function for your specific domain.
    Replace with your own data generation logic.
    
    Example domains:
    - Healthcare: patient vitals, diagnoses
    - Finance: transaction data, risk scores
    - E-commerce: customer behavior, purchase history
    - Manufacturing: sensor data, equipment metrics
    """
    # Instead of hardcoding domain data, use generic dataset
    return create_sample_dataset(n_samples, n_features=15)


# Alias for backward compatibility with main.py import
create_sample_agriharv_data = create_sample_healthcare_data
