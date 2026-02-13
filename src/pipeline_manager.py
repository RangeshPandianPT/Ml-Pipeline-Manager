"""
Pipeline Manager for ML Pipeline.
Orchestrates the entire MLOps workflow: ingestion, feature engineering,
drift monitoring, and model retraining.
Domain-agnostic implementation suitable for any ML project.
"""

import pandas as pd
import numpy as np
import json
import logging
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import xgboost as xgb
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from .config import PipelineConfig, default_config
from .database import MetadataDatabase, ModelLog, generate_id
from .ingestion import DataIngestion
from .feature_eng import FeatureEngineer, AutoFeatureFactory
from .monitor import DriftMonitor, DriftReport, detect_drift, simulate_drift, DriftResult

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Machine learning task type."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class PipelineState(Enum):
    """Pipeline execution state."""
    IDLE = "idle"
    INGESTING = "ingesting"
    FEATURE_ENGINEERING = "feature_engineering"
    MONITORING = "monitoring"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineMetrics:
    """Container for pipeline metrics."""
    ingestion_rows: int = 0
    ingestion_columns: int = 0
    features_created: int = 0
    drift_detected: bool = False
    drift_columns: List[str] = None
    model_trained: bool = False
    model_metrics: Dict[str, float] = None
    execution_time_seconds: float = 0.0
    
    def __post_init__(self):
        if self.drift_columns is None:
            self.drift_columns = []
        if self.model_metrics is None:
            self.model_metrics = {}


class ModelFactory:
    """Factory for creating ML models."""
    
    @staticmethod
    def create_model(model_type: str, task_type: TaskType, **kwargs):
        """
        Create a model instance.
        
        Args:
            model_type: Type of model ("random_forest", "gradient_boosting", etc.)
            task_type: Regression or classification
            **kwargs: Additional model parameters
        
        Returns:
            Scikit-learn model instance
        """
        models = {
            TaskType.REGRESSION: {
                "random_forest": RandomForestRegressor,
                "gradient_boosting": GradientBoostingRegressor,
                "linear": LinearRegression,
                "xgboost": xgb.XGBRegressor,
            },
            TaskType.CLASSIFICATION: {
                "random_forest": RandomForestClassifier,
                "logistic_regression": LogisticRegression,
                "xgboost": xgb.XGBClassifier,
            }
        }
        
        model_class = models.get(task_type, {}).get(model_type)
        if model_class is None:
            raise ValueError(f"Unknown model type: {model_type} for {task_type}")
        
        # Set default parameters
        default_params = {
            "random_forest": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
            "gradient_boosting": {"n_estimators": 100, "random_state": 42},
            "linear": {},
            "logistic_regression": {"random_state": 42, "max_iter": 1000},
            "xgboost": {"n_estimators": 100, "random_state": 42, "tree_method": "hist", "device": "cuda"},
        }
        
        params = {**default_params.get(model_type, {}), **kwargs}
        
        # Remove unsupported parameters for specific models
        if model_type in ["linear"]:
            params.pop("random_state", None)
            params.pop("n_estimators", None)
            params.pop("n_jobs", None)
        
        return model_class(**params)


class ModelEvaluator:
    """Evaluates model performance."""
    
    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate regression model."""
        return {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred))
        }
    
    @staticmethod
    def evaluate_classification(y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 y_proba: np.ndarray = None) -> Dict[str, float]:
        """Evaluate classification model."""
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }
        
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            except Exception:
                pass
        
        return metrics


class MLPipeline:
    """
    Main ML Pipeline Manager.
    
    Orchestrates the complete MLOps workflow:
    1. Data Ingestion
    2. Feature Engineering
    3. Drift Monitoring
    4. Model Training/Retraining
    5. Evaluation and Logging
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or default_config
        self.db = MetadataDatabase(self.config.database)
        
        # Initialize components
        self.ingestion = DataIngestion(self.config, self.db)
        self.feature_engineer = FeatureEngineer(self.config, self.db)
        self.drift_monitor = DriftMonitor(self.config, self.db)
        
        # State
        self._state = PipelineState.IDLE
        self._current_model = None
        self._current_model_id: Optional[str] = None
        self._metrics = PipelineMetrics()
        self._task_type = TaskType.REGRESSION
        
        # Data storage
        self._raw_data: Optional[pd.DataFrame] = None
        self._processed_data: Optional[pd.DataFrame] = None
        self._X_train: Optional[pd.DataFrame] = None
        self._X_test: Optional[pd.DataFrame] = None
        self._y_train: Optional[np.ndarray] = None
        self._y_test: Optional[np.ndarray] = None
        
        # Setup logging
        self._setup_logging()
        
        # Ensure directories exist
        self._setup_directories()
    
    def _setup_logging(self):
        """Configure logging."""
        log_file = self.config.log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _setup_directories(self):
        """Create required directories."""
        for path in [self.config.data_dir, self.config.raw_data_path, 
                     self.config.processed_data_path, self.config.model_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def ingest(self, 
               source: Union[str, Path, pd.DataFrame],
               source_type: str = "auto",
               **kwargs) -> pd.DataFrame:
        """
        Ingest data from various sources.
        
        Args:
            source: File path or DataFrame
            source_type: "csv", "json", "excel", "dataframe", or "auto"
            **kwargs: Additional arguments for pandas readers
        
        Returns:
            Ingested DataFrame
        """
        self._state = PipelineState.INGESTING
        start_time = datetime.utcnow()
        
        try:
            if isinstance(source, pd.DataFrame):
                df = self.ingestion.ingest_dataframe(source, "dataframe")
            else:
                source = Path(source)
                
                if source_type == "auto":
                    suffix = source.suffix.lower()
                    source_type = {
                        ".csv": "csv",
                        ".json": "json",
                        ".xlsx": "excel",
                        ".xls": "excel"
                    }.get(suffix, "csv")
                
                if source_type == "csv":
                    df = self.ingestion.ingest_csv(source, **kwargs)
                elif source_type == "json":
                    df = self.ingestion.ingest_json(source, **kwargs)
                elif source_type == "excel":
                    df = self.ingestion.ingest_excel(source, **kwargs)
                else:
                    raise ValueError(f"Unknown source type: {source_type}")
            
            self._raw_data = df
            self._metrics.ingestion_rows = len(df)
            self._metrics.ingestion_columns = len(df.columns)
            
            logger.info(f"Ingestion successful: {len(df)} rows, {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            self._state = PipelineState.FAILED
            logger.error(f"Ingestion failed: {str(e)}")
            raise
    
    def engineer_features(self,
                          df: pd.DataFrame = None,
                          auto: bool = True,
                          transformations: List[str] = None,
                          target_column: str = None) -> dict:
        """
        Perform feature engineering.
        
        Args:
            df: Input DataFrame (uses ingested data if None)
            auto: Use automatic feature engineering
            transformations: List of specific transformations to apply
            target_column: Target column to exclude from transformations
        
        Returns:
            Dictionary containing:
                - transformed_data: Transformed DataFrame
                - features_created: Number of new features created
                - original_features: Original feature count
                - new_features: New feature count
        """
        self._state = PipelineState.FEATURE_ENGINEERING
        
        if df is None:
            df = self._raw_data
        
        if df is None:
            raise ValueError("No data available. Run ingest() first.")
        
        target_col = target_column or self.config.features.target_column
        ingestion_id = self.ingestion.ingestion_id
        
        try:
            if auto:
                result_df = self.feature_engineer.auto_transform(
                    df, ingestion_id, target_col
                )
            else:
                self.feature_engineer.reset_pipeline()
                if transformations:
                    for t in transformations:
                        self.feature_engineer.add_transformation(t)
                else:
                    self.feature_engineer.add_transformations_from_config()
                
                result_df = self.feature_engineer.transform(
                    df, ingestion_id, target_col
                )
            
            self._processed_data = result_df
            original_features = len(df.columns)
            new_features = len(result_df.columns)
            self._metrics.features_created = new_features - original_features
            
            logger.info(f"Feature engineering complete: {original_features} -> {new_features} features")
            
            # Return dict for compatibility with Streamlit app and other consumers
            return {
                'transformed_data': result_df,
                'features_created': self._metrics.features_created,
                'original_features': original_features,
                'new_features': new_features
            }
            
        except Exception as e:
            self._state = PipelineState.FAILED
            logger.error(f"Feature engineering failed: {str(e)}")
            raise
    
    def monitor_drift(self,
                      current_data: pd.DataFrame = None,
                      set_as_reference: bool = False) -> DriftReport:
        """
        Monitor for data drift.
        
        Args:
            current_data: Data to check for drift (uses processed data if None)
            set_as_reference: If True, set this data as the new reference
        
        Returns:
            DriftReport with drift analysis
        """
        self._state = PipelineState.MONITORING
        
        if current_data is None:
            current_data = self._processed_data if self._processed_data is not None else self._raw_data
        
        if current_data is None:
            raise ValueError("No data available for drift monitoring")
        
        try:
            # Try to load reference from database if not set
            if self.drift_monitor._reference_data is None:
                if not self.drift_monitor.load_reference_from_db():
                    # No reference exists - set current as reference
                    logger.info("No reference data found. Setting current data as reference.")
                    self.drift_monitor.set_reference(current_data, self.ingestion.ingestion_id)
                    return self.drift_monitor._create_empty_report(
                        generate_id("DRIFT_"),
                        datetime.utcnow().isoformat()
                    )
            
            # Detect drift
            report = self.drift_monitor.detect_drift(
                current_data,
                self.ingestion.ingestion_id
            )
            
            self._metrics.drift_detected = report.overall_drift_detected
            self._metrics.drift_columns = report.columns_with_drift
            
            logger.info(f"Drift monitoring complete: drift={'YES' if report.overall_drift_detected else 'NO'}")
            
            # Update reference if requested
            if set_as_reference:
                self.drift_monitor.set_reference(current_data, self.ingestion.ingestion_id)
            
            return report
            
        except Exception as e:
            self._state = PipelineState.FAILED
            logger.error(f"Drift monitoring failed: {str(e)}")
            raise
    
    def prepare_training_data(self,
                               df: pd.DataFrame = None,
                               target_column: str = None,
                               test_size: float = None) -> tuple:
        """
        Prepare data for model training.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            test_size: Fraction for test set
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if df is None:
            df = self._processed_data if self._processed_data is not None else self._raw_data
        
        if df is None:
            raise ValueError("No data available")
        
        target_col = target_column or self.config.features.target_column
        test_size = test_size or self.config.validation_split
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Ensure target has no NaN values
        if y.isnull().any():
            logger.warning(f"Target column '{target_col}' has {y.isnull().sum()} NaN values. Removing them.")
            valid_indices = ~y.isnull()
            X = X[valid_indices]
            y = y[valid_indices]
        
        # Remove non-numeric columns for modeling
        X = X.select_dtypes(include=[np.number])
        
        # Handle any remaining NaN values
        X = X.fillna(X.mean())
        
        # Determine task type
        if y.dtype == 'object' or y.nunique() <= 10:
            self._task_type = TaskType.CLASSIFICATION
            y = y.astype('category').cat.codes
        else:
            self._task_type = TaskType.REGRESSION
        
        # Split data
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            X, y.values,
            test_size=test_size,
            random_state=self.config.random_state
        )
        
        logger.info(f"Training data prepared: {len(self._X_train)} train, {len(self._X_test)} test samples")
        logger.info(f"Task type: {self._task_type.value}")
        
        return self._X_train, self._X_test, self._y_train, self._y_test
    
    def train_model(self,
                    model_type: str = None,
                    X_train: pd.DataFrame = None,
                    y_train: np.ndarray = None,
                    **model_params) -> Any:
        """
        Train a machine learning model.
        
        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training target
            **model_params: Additional model parameters
        
        Returns:
            Trained model
        """
        self._state = PipelineState.TRAINING
        
        X_train = X_train if X_train is not None else self._X_train
        y_train = y_train if y_train is not None else self._y_train
        model_type = model_type or self.config.model_type
        
        if X_train is None or y_train is None:
            raise ValueError("Training data not prepared. Call prepare_training_data() first.")
        
        try:
            # Create model
            print(f"DEBUG: Creating model of type {model_type}")
            model = ModelFactory.create_model(model_type, self._task_type, **model_params)
            
            # Train
            print(f"DEBUG: Training model... X_train type: {type(X_train)}")
            model.fit(X_train, y_train)
            
            self._current_model = model
            self._current_model_id = generate_id("MODEL_")
            self._metrics.model_trained = True
            
            logger.info("Model training complete")
            
            return model
            
        except Exception as e:
            self._state = PipelineState.FAILED
            logger.error(f"Model training failed: {str(e)}")
            raise
    
    def evaluate_model(self,
                       model = None,
                       X_test: pd.DataFrame = None,
                       y_test: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Model to evaluate (uses current model if None)
            X_test: Test features
            y_test: Test target
        
        Returns:
            Dictionary of metrics
        """
        self._state = PipelineState.EVALUATING
        
        model = model or self._current_model
        X_test = X_test if X_test is not None else self._X_test
        y_test = y_test if y_test is not None else self._y_test
        
        if model is None:
            raise ValueError("No model to evaluate. Train a model first.")
        
        if X_test is None or y_test is None:
            raise ValueError("Test data not available")
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Evaluate based on task type
        if self._task_type == TaskType.REGRESSION:
            metrics = ModelEvaluator.evaluate_regression(y_test, y_pred)
        else:
            y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            metrics = ModelEvaluator.evaluate_classification(y_test, y_pred, y_proba)
        
        self._metrics.model_metrics = metrics
        
        logger.info(f"Model evaluation complete: {metrics}")
        
        return metrics
    
    def save_model(self, 
                   model = None,
                   model_name: str = None) -> str:
        """
        Save model to disk and log to database.
        
        Args:
            model: Model to save
            model_name: Custom name for the model file
        
        Returns:
            Path to saved model
        """
        model = model or self._current_model
        
        if model is None:
            raise ValueError("No model to save")
        
        model_name = model_name or f"{self._current_model_id}.joblib"
        model_path = Path(self.config.model_path) / model_name
        
        # Save model
        joblib.dump(model, model_path)
        
        # Log to database
        log = ModelLog(
            model_id=self._current_model_id,
            feature_id=self.feature_engineer.feature_id or "unknown",
            model_type=self.config.model_type,
            hyperparameters=model.get_params() if hasattr(model, 'get_params') else {},
            metrics=self._metrics.model_metrics or {},
            model_path=str(model_path),
            timestamp=datetime.utcnow().isoformat(),
            is_active=True,
            status="saved"
        )
        self.db.log_model(log)
        
        logger.info(f"Model saved to: {model_path}")
        
        return str(model_path)
    
    def load_model(self, model_path: str = None) -> Any:
        """
        Load a model from disk.
        
        Args:
            model_path: Path to model file (loads active model if None)
        
        Returns:
            Loaded model
        """
        if model_path is None:
            # Get active model from database
            active = self.db.get_active_model()
            if active:
                model_path = active['model_path']
            else:
                raise ValueError("No model path provided and no active model found")
        
        model = joblib.load(model_path)
        self._current_model = model
        
        logger.info(f"Model loaded from: {model_path}")
        
        return model
    
    def run_full_pipeline(self,
                          source: Union[str, Path, pd.DataFrame],
                          target_column: str = None,
                          auto_features: bool = True,
                          check_drift: bool = True,
                          train_model: bool = True) -> Dict[str, Any]:
        """
        Run the complete ML pipeline.
        
        Args:
            source: Data source (file path or DataFrame)
            target_column: Target column name
            auto_features: Use automatic feature engineering
            check_drift: Check for data drift
            train_model: Train a new model
        
        Returns:
            Pipeline results summary
        """
        start_time = datetime.utcnow()
        results = {
            "status": "started",
            "steps_completed": [],
            "metrics": None,
            "drift_report": None,
            "model_metrics": None,
            "errors": []
        }
        
        try:
            # Step 1: Ingest data
            logger.info("=" * 50)
            logger.info("STEP 1: Data Ingestion")
            df = self.ingest(source)
            results["steps_completed"].append("ingestion")
            
            # Step 2: Feature Engineering
            logger.info("=" * 50)
            logger.info("STEP 2: Feature Engineering")
            df = self.engineer_features(df, auto=auto_features, target_column=target_column)
            results["steps_completed"].append("feature_engineering")
            
            # Step 3: Drift Monitoring
            drift_report = None
            if check_drift:
                logger.info("=" * 50)
                logger.info("STEP 3: Drift Monitoring")
                try:
                    drift_report = self.monitor_drift(df)
                    results["drift_report"] = drift_report.to_dict() if drift_report else None
                    results["steps_completed"].append("drift_monitoring")
                except Exception as e:
                    logger.warning(f"Drift monitoring skipped: {str(e)}")
                    results["errors"].append(f"Drift monitoring: {str(e)}")
            
            # Step 4: Model Training (if needed)
            if train_model:
                should_train = True
                
                # Check if drift requires retraining
                if drift_report and drift_report.overall_drift_detected:
                    logger.info("Drift detected - triggering model retraining")
                elif drift_report and not drift_report.overall_drift_detected:
                    # Check if we have an existing model
                    active_model = self.db.get_active_model()
                    if active_model and not self.config.monitor.auto_retrain:
                        logger.info("No drift detected and auto_retrain is disabled. Skipping training.")
                        should_train = False
                
                if should_train:
                    logger.info("=" * 50)
                    logger.info("STEP 4: Model Training")
                    
                    self.prepare_training_data(df, target_column)
                    self.train_model()
                    results["steps_completed"].append("model_training")
                    
                    # Step 5: Evaluation
                    logger.info("=" * 50)
                    logger.info("STEP 5: Model Evaluation")
                    
                    metrics = self.evaluate_model()
                    results["model_metrics"] = metrics
                    results["steps_completed"].append("evaluation")
                    
                    # Save model
                    self.save_model()
                    results["steps_completed"].append("model_saved")
            
            # Update reference data for future drift checks
            self.drift_monitor.set_reference(df, self.ingestion.ingestion_id)
            
            # Complete
            self._state = PipelineState.COMPLETED
            end_time = datetime.utcnow()
            self._metrics.execution_time_seconds = (end_time - start_time).total_seconds()
            
            results["status"] = "completed"
            results["metrics"] = {
                "ingestion_rows": self._metrics.ingestion_rows,
                "ingestion_columns": self._metrics.ingestion_columns,
                "features_created": self._metrics.features_created,
                "drift_detected": self._metrics.drift_detected,
                "model_trained": self._metrics.model_trained,
                "execution_time_seconds": self._metrics.execution_time_seconds
            }
            
            logger.info("=" * 50)
            logger.info(f"Pipeline completed successfully in {self._metrics.execution_time_seconds:.2f}s")
            
            return results
            
        except Exception as e:
            self._state = PipelineState.FAILED
            results["status"] = "failed"
            results["errors"].append(str(e))
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def predict(self, 
                X: pd.DataFrame,
                apply_features: bool = True) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            apply_features: Apply the same feature transformations
        
        Returns:
            Predictions
        """
        if self._current_model is None:
            self.load_model()
        
        if apply_features:
            # Apply same transformations
            X_transformed = self.feature_engineer.transform(X)
            X = X_transformed.select_dtypes(include=[np.number])
        
        # Ensure columns match training data
        if self._X_train is not None:
            missing_cols = set(self._X_train.columns) - set(X.columns)
            for col in missing_cols:
                X[col] = 0
            X = X[self._X_train.columns]
        
        return self._current_model.predict(X)
    
    @property
    def state(self) -> PipelineState:
        """Get current pipeline state."""
        return self._state
    
    @property
    def model(self) -> Any:
        """Get current model."""
        return self._current_model
    
    @property
    def task_type(self) -> Optional[TaskType]:
        """Get current task type."""
        return self._task_type
    
    @property
    def metrics(self) -> PipelineMetrics:
        """Get pipeline metrics."""
        return self._metrics
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the pipeline configuration and state."""
        return {
            "state": self._state.value,
            "task_type": self._task_type.value if self._task_type else None,
            "model_type": self.config.model_type,
            "current_model_id": self._current_model_id,
            "metrics": {
                "ingestion_rows": self._metrics.ingestion_rows,
                "features_created": self._metrics.features_created,
                "drift_detected": self._metrics.drift_detected,
                "model_trained": self._metrics.model_trained
            },
            "config": {
                "validation_split": self.config.validation_split,
                "ks_significance_level": self.config.monitor.ks_significance_level,
                "auto_retrain": self.config.monitor.auto_retrain
            }
        }


# =============================================================================
# PipelineWorkflow — Phase 2 Coordinator
# =============================================================================

class PipelineWorkflow:
    """
    High-level workflow that coordinates all Phase 2 modules:
      - DataIngestion.ingest_from_sources (mock weather API + CSV)
      - AutoFeatureFactory (auto feature engineering)
      - detect_drift (standalone K-S based monitoring)
      - Automatic Random Forest retraining when drift is detected
      - Logs new model's R² score to a local JSON metadata file.
    """

    METADATA_FILE = "model_metadata.json"

    def __init__(
        self,
        config: PipelineConfig = None,
        metadata_dir: Union[str, Path] = "models",
    ):
        self.config = config or default_config
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.ingestion = DataIngestion(self.config)
        self.feature_factory = AutoFeatureFactory()
        self._reference_data: Optional[pd.DataFrame] = None
        self._current_model = None
        self._model_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        reference_data: Optional[pd.DataFrame] = None,
        target_column: str = "target",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Execute the full coordinated workflow.

        Steps:
          1. Ingest data from mock weather API + optional CSV.
          2. Run AutoFeatureFactory for automatic feature engineering.
          3. Call detect_drift() against reference data.
          4. If retrain_flag is True, train a Random Forest on the
             combined old + new data and log R² to JSON.
          5. Return a summary dict.

        Args:
            csv_path: Optional local CSV to ingest alongside API data.
            reference_data: Baseline DataFrame for drift comparison.
                If None, first run sets data as baseline (no drift check).
            target_column: Name of the target variable.
            test_size: Fraction used for test split during retraining.
            random_state: Seed for reproducibility.

        Returns:
            Summary dict with ingestion, drift, and training info.
        """
        summary: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "steps_completed": [],
            "drift": None,
            "model_metrics": None,
            "model_path": None,
        }

        # ── Step 1: Ingest ─────────────────────────────────────────────
        logger.info("=" * 50)
        logger.info("WORKFLOW STEP 1: Ingest data from sources")
        new_data = self.ingestion.ingest_from_sources(csv_path=csv_path)
        summary["steps_completed"].append("ingestion")
        logger.info(f"Ingested shape: {new_data.shape}")

        # ── Step 2: Feature Engineering ─────────────────────────────────
        logger.info("=" * 50)
        logger.info("WORKFLOW STEP 2: AutoFeatureFactory")
        engineered_data = self.feature_factory.fit_transform(
            new_data, target_column=target_column
        )
        summary["steps_completed"].append("feature_engineering")
        summary["feature_summary"] = self.feature_factory.feature_summary
        logger.info(f"Engineered shape: {engineered_data.shape}")

        # ── Step 3: Drift Detection ─────────────────────────────────────
        logger.info("=" * 50)
        logger.info("WORKFLOW STEP 3: Drift Detection (K-S test, p < 0.05)")

        if reference_data is None and self._reference_data is None:
            # First run — set as baseline
            self._reference_data = engineered_data.copy()
            logger.info("No reference data — setting current as baseline.")
            summary["drift"] = {"retrain_flag": False, "reason": "baseline_set"}
            summary["steps_completed"].append("drift_baseline_set")
        else:
            ref = reference_data if reference_data is not None else self._reference_data

            # Only engineer reference if it hasn't been engineered already
            # (internal _reference_data is always stored post-engineering)
            if reference_data is not None and target_column in ref.columns:
                ref_engineered = self.feature_factory.fit_transform(
                    ref, target_column=target_column
                )
            else:
                ref_engineered = ref

            drift_result: DriftResult = detect_drift(
                reference_data=ref_engineered,
                current_data=engineered_data,
                significance_level=0.05,
                drift_threshold=0.20,
            )
            summary["drift"] = drift_result.to_dict()
            summary["steps_completed"].append("drift_detection")

            # ── Step 4: Retrain if drift ────────────────────────────────
            if drift_result.retrain_flag:
                logger.info("=" * 50)
                logger.info("WORKFLOW STEP 4: Retraining — drift triggered")

                # Combine old + new data
                combined = pd.concat(
                    [ref_engineered, engineered_data], ignore_index=True
                )

                metrics, model, model_path = self._train_random_forest(
                    combined,
                    target_column=target_column,
                    test_size=test_size,
                    random_state=random_state,
                )
                summary["model_metrics"] = metrics
                summary["model_path"] = model_path
                summary["steps_completed"].append("retrain")

                # Log to JSON metadata file
                self._log_to_json(metrics, model_path)
                summary["steps_completed"].append("metadata_logged")
            else:
                logger.info("No significant drift — skipping retraining.")

            # Update reference for next run
            self._reference_data = engineered_data.copy()

        summary["status"] = "completed"
        logger.info(f"Workflow finished — steps: {summary['steps_completed']}")
        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_random_forest(
        self,
        data: pd.DataFrame,
        target_column: str,
        test_size: float,
        random_state: int,
    ) -> tuple:
        """Train a Random Forest on combined data and return (metrics, model, path)."""
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not in data")

        X = data.drop(columns=[target_column]).select_dtypes(include=[np.number])
        y = data[target_column]

        # Drop rows where target is NaN
        valid = ~y.isnull()
        X, y = X[valid], y[valid]
        X = X.fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model = RandomForestRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))

        metrics = {"r2": round(r2, 6), "rmse": round(rmse, 6), "mae": round(mae, 6)}
        logger.info(f"Random Forest retrained — R²={r2:.4f}, RMSE={rmse:.4f}")

        # Save model
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        model_id = generate_id("MODEL_")
        model_path = str(self.metadata_dir / f"{model_id}.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Model saved: {model_path}")

        self._current_model = model
        self._model_path = model_path
        return metrics, model, model_path

    def _log_to_json(self, metrics: Dict[str, float], model_path: str) -> None:
        """Append model metrics (including R²) to a local JSON metadata file."""
        meta_file = self.metadata_dir / self.METADATA_FILE

        # Load existing entries
        entries: List[Dict[str, Any]] = []
        if meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    entries = json.load(f)
            except (json.JSONDecodeError, IOError):
                entries = []

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_path": model_path,
            "r2_score": metrics["r2"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
        }
        entries.append(entry)

        with open(meta_file, "w") as f:
            json.dump(entries, f, indent=2)

        logger.info(f"Model metadata (R²={metrics['r2']:.4f}) logged to {meta_file}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_model(self):
        """Currently trained model (or None)."""
        return self._current_model

    @property
    def model_path(self) -> Optional[str]:
        """Path to the most recently saved model."""
        return self._model_path
