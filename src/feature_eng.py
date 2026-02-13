"""
Feature Engineering Module for ML Pipeline.
Implements decorator pattern and config-driven transformations.
Domain-agnostic implementation suitable for any dataset.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Callable, Dict, List, Any, Optional, Union
from functools import wraps
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings

from .config import PipelineConfig, FeatureConfig, default_config
from .database import MetadataDatabase, FeatureLog, generate_id

logger = logging.getLogger(__name__)

# Global registry for feature transformations
TRANSFORMATION_REGISTRY: Dict[str, Callable] = {}


def register_transformation(name: str):
    """
    Decorator to register a transformation function.
    
    Usage:
        @register_transformation("my_transform")
        def my_transform(df, columns, **kwargs):
            # transformation logic
            return df
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"Applying transformation: {name}")
            return func(*args, **kwargs)
        TRANSFORMATION_REGISTRY[name] = wrapper
        return wrapper
    return decorator


def get_available_transformations():
    """
    Get list of all available transformation names.
    
    Returns:
        List of transformation names registered in TRANSFORMATION_REGISTRY
    """
    return list(TRANSFORMATION_REGISTRY.keys())



def transformation(name: str = None, log_execution: bool = True):
    """
    Advanced decorator for feature transformations with logging.
    
    Usage:
        @transformation("standardize", log_execution=True)
        def standardize_features(df, columns):
            ...
    """
    def decorator(func: Callable):
        transform_name = name or func.__name__
        
        @wraps(func)
        def wrapper(df: pd.DataFrame, columns: List[str] = None, **kwargs):
            start_time = datetime.utcnow()
            original_cols = set(df.columns)
            
            try:
                result = func(df.copy(), columns, **kwargs)
                
                new_cols = set(result.columns) - original_cols
                
                if log_execution:
                    duration = (datetime.utcnow() - start_time).total_seconds()
                    logger.info(f"Transformation '{transform_name}' completed in {duration:.3f}s")
                    if new_cols:
                        logger.info(f"New columns created: {new_cols}")
                
                return result
            except Exception as e:
                logger.error(f"Transformation '{transform_name}' failed: {str(e)}")
                raise
        
        TRANSFORMATION_REGISTRY[transform_name] = wrapper
        return wrapper
    return decorator


# ==================== BUILT-IN TRANSFORMATIONS ====================

@register_transformation("standardize")
def standardize(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> pd.DataFrame:
    """Standardize numerical columns (z-score normalization)."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    columns = [c for c in columns if c in df.columns]
    if not columns:
        return df
    
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns].fillna(df[columns].mean()))
    return df


@register_transformation("normalize")
def normalize(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> pd.DataFrame:
    """Min-Max normalization (scale to 0-1 range)."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    columns = [c for c in columns if c in df.columns]
    if not columns:
        return df
    
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns].fillna(df[columns].mean()))
    return df


@register_transformation("log_transform")
def log_transform(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> pd.DataFrame:
    """Apply log(1+x) transformation for skewed data."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    columns = [c for c in columns if c in df.columns]
    suffix = kwargs.get('suffix', '_log')
    
    for col in columns:
        # Only apply to non-negative columns
        if df[col].min() >= 0:
            df[f'{col}{suffix}'] = np.log1p(df[col].fillna(0))
    
    return df


@register_transformation("sqrt_transform")
def sqrt_transform(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> pd.DataFrame:
    """Apply square root transformation."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    columns = [c for c in columns if c in df.columns]
    suffix = kwargs.get('suffix', '_sqrt')
    
    for col in columns:
        if df[col].min() >= 0:
            df[f'{col}{suffix}'] = np.sqrt(df[col].fillna(0))
    
    return df


@register_transformation("power_transform")
def power_transform(df: pd.DataFrame, columns: List[str] = None, power: float = 2, **kwargs) -> pd.DataFrame:
    """Apply power transformation."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    columns = [c for c in columns if c in df.columns]
    suffix = kwargs.get('suffix', f'_pow{power}')
    
    for col in columns:
        df[f'{col}{suffix}'] = np.power(df[col].fillna(0), power)
    
    return df


@register_transformation("binning")
def binning(df: pd.DataFrame, columns: List[str] = None, n_bins: int = 5, **kwargs) -> pd.DataFrame:
    """Create bins (quantile-based discretization)."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    columns = [c for c in columns if c in df.columns]
    suffix = kwargs.get('suffix', '_binned')
    
    for col in columns:
        try:
            df[f'{col}{suffix}'] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
        except ValueError:
            # Fall back to regular cut if quantile fails
            df[f'{col}{suffix}'] = pd.cut(df[col], bins=n_bins, labels=False)
    
    return df


@register_transformation("one_hot_encode")
def one_hot_encode(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    columns = [c for c in columns if c in df.columns]
    if not columns:
        return df
    
    drop_original = kwargs.get('drop_original', True)
    
    df = pd.get_dummies(df, columns=columns, drop_first=False)
    return df


@register_transformation("label_encode")
def label_encode(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> pd.DataFrame:
    """Label encode categorical columns."""
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    columns = [c for c in columns if c in df.columns]
    suffix = kwargs.get('suffix', '_encoded')
    
    for col in columns:
        le = LabelEncoder()
        # Handle NaN values
        df[f'{col}{suffix}'] = df[col].fillna('MISSING')
        df[f'{col}{suffix}'] = le.fit_transform(df[f'{col}{suffix}'].astype(str))
    
    return df


@register_transformation("impute_mean")
def impute_mean(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> pd.DataFrame:
    """Impute missing values with mean."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    columns = [c for c in columns if c in df.columns]
    
    imputer = SimpleImputer(strategy='mean')
    df[columns] = imputer.fit_transform(df[columns])
    return df


@register_transformation("impute_median")
def impute_median(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> pd.DataFrame:
    """Impute missing values with median."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    columns = [c for c in columns if c in df.columns]
    
    imputer = SimpleImputer(strategy='median')
    df[columns] = imputer.fit_transform(df[columns])
    return df


@register_transformation("impute_mode")
def impute_mode(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> pd.DataFrame:
    """Impute missing values with mode (most frequent)."""
    if columns is None:
        columns = df.columns.tolist()
    
    columns = [c for c in columns if c in df.columns]
    
    imputer = SimpleImputer(strategy='most_frequent')
    df[columns] = imputer.fit_transform(df[columns])
    return df


@register_transformation("drop_na")
def drop_na(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> pd.DataFrame:
    """Drop rows with missing values."""
    if columns is None:
        return df.dropna()
    
    columns = [c for c in columns if c in df.columns]
    return df.dropna(subset=columns)


@register_transformation("interaction_features")
def interaction_features(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> pd.DataFrame:
    """Create interaction features (pairwise multiplication)."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()[:5]  # Limit to avoid explosion
    
    columns = [c for c in columns if c in df.columns]
    
    for i, col1 in enumerate(columns):
        for col2 in columns[i+1:]:
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
    
    return df


@register_transformation("polynomial_features")
def polynomial_features(df: pd.DataFrame, columns: List[str] = None, degree: int = 2, **kwargs) -> pd.DataFrame:
    """Create polynomial features."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    columns = [c for c in columns if c in df.columns]
    
    for col in columns:
        for d in range(2, degree + 1):
            df[f'{col}_pow{d}'] = np.power(df[col].fillna(0), d)
    
    return df


@register_transformation("rolling_stats")
def rolling_stats(df: pd.DataFrame, columns: List[str] = None, window: int = 3, **kwargs) -> pd.DataFrame:
    """Create rolling window statistics."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    columns = [c for c in columns if c in df.columns]
    
    for col in columns:
        df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
        df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
    
    return df


@register_transformation("lag_features")
def lag_features(df: pd.DataFrame, columns: List[str] = None, lags: List[int] = None, **kwargs) -> pd.DataFrame:
    """Create lag features for time series data."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if lags is None:
        lags = [1, 2, 3]
    
    columns = [c for c in columns if c in df.columns]
    
    for col in columns:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df


@register_transformation("datetime_features")
def datetime_features(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> pd.DataFrame:
    """Extract features from datetime columns."""
    if columns is None:
        columns = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    columns = [c for c in columns if c in df.columns]
    
    for col in columns:
        dt_col = pd.to_datetime(df[col])
        df[f'{col}_year'] = dt_col.dt.year
        df[f'{col}_month'] = dt_col.dt.month
        df[f'{col}_day'] = dt_col.dt.day
        df[f'{col}_dayofweek'] = dt_col.dt.dayofweek
        df[f'{col}_hour'] = dt_col.dt.hour
        df[f'{col}_is_weekend'] = (dt_col.dt.dayofweek >= 5).astype(int)
    
    return df


@register_transformation("clip_outliers")
def clip_outliers(df: pd.DataFrame, columns: List[str] = None, n_std: float = 3, **kwargs) -> pd.DataFrame:
    """Clip outliers beyond n standard deviations."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    columns = [c for c in columns if c in df.columns]
    
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        lower_bound = mean - n_std * std
        upper_bound = mean + n_std * std
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df


@register_transformation("remove_outliers_iqr")
def remove_outliers_iqr(df: pd.DataFrame, columns: List[str] = None, multiplier: float = 1.5, **kwargs) -> pd.DataFrame:
    """Remove outliers using IQR method."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    columns = [c for c in columns if c in df.columns]
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df


# ==================== FEATURE ENGINEERING ENGINE ====================

@dataclass
class TransformationStep:
    """Represents a single transformation step."""
    name: str
    columns: Optional[List[str]] = None
    params: Dict[str, Any] = field(default_factory=dict)


class FeatureEngineer:
    """
    Feature Engineering Engine with config-driven and decorator-based transformations.
    """
    
    def __init__(self, config: PipelineConfig = None, db: MetadataDatabase = None):
        self.config = config or default_config
        self.db = db or MetadataDatabase(self.config.database)
        self._transformation_pipeline: List[TransformationStep] = []
        self._fitted_transformers: Dict[str, Any] = {}
        self._current_feature_id: Optional[str] = None
        self._original_columns: List[str] = []
    
    def add_transformation(self, 
                          name: str, 
                          columns: List[str] = None, 
                          **params) -> 'FeatureEngineer':
        """
        Add a transformation to the pipeline.
        
        Args:
            name: Name of the registered transformation
            columns: Columns to apply transformation to (None = auto-select)
            **params: Additional parameters for the transformation
        
        Returns:
            self for method chaining
        """
        if name not in TRANSFORMATION_REGISTRY:
            raise ValueError(f"Unknown transformation: {name}. Available: {list(TRANSFORMATION_REGISTRY.keys())}")
        
        step = TransformationStep(name=name, columns=columns, params=params)
        self._transformation_pipeline.append(step)
        logger.debug(f"Added transformation: {name}")
        return self
    
    def add_transformations_from_config(self, feature_config: FeatureConfig = None) -> 'FeatureEngineer':
        """Load transformations from configuration."""
        config = feature_config or self.config.features
        
        # Add numerical transformations
        for col, transforms in config.numerical_transformations.items():
            for transform in transforms:
                if col == "default":
                    self.add_transformation(transform, columns=None)
                else:
                    self.add_transformation(transform, columns=[col])
        
        # Add categorical transformations
        for col, transforms in config.categorical_transformations.items():
            for transform in transforms:
                if col == "default":
                    self.add_transformation(transform, columns=None)
                else:
                    self.add_transformation(transform, columns=[col])
        
        return self
    
    def transform(self, 
                  df: pd.DataFrame, 
                  ingestion_id: str = None,
                  target_column: str = None) -> pd.DataFrame:
        """
        Apply all transformations in the pipeline.
        
        Args:
            df: Input DataFrame
            ingestion_id: ID from ingestion for logging
            target_column: Column to exclude from transformations
        
        Returns:
            Transformed DataFrame
        """
        feature_id = generate_id("FE_")
        self._current_feature_id = feature_id
        self._original_columns = df.columns.tolist()
        
        target_col = target_column or self.config.features.target_column
        original_shape = df.shape
        applied_transforms = []
        
        try:
            result_df = df.copy()
            
            # Exclude target from transformations if present
            target_data = None
            if target_col in result_df.columns:
                target_data = result_df[target_col].copy()
            
            # Apply each transformation
            for step in self._transformation_pipeline:
                transform_func = TRANSFORMATION_REGISTRY[step.name]
                
                # Filter columns to exclude target
                step_columns = step.columns
                if step_columns and target_col in step_columns:
                    step_columns = [c for c in step_columns if c != target_col]
                
                result_df = transform_func(result_df, columns=step_columns, **step.params)
                applied_transforms.append(step.name)
                logger.debug(f"Applied: {step.name}, shape: {result_df.shape}")
            
            # Restore target column if it was modified
            if target_data is not None and target_col in result_df.columns:
                result_df[target_col] = target_data
            
            # Log feature engineering
            log = FeatureLog(
                feature_id=feature_id,
                ingestion_id=ingestion_id or "unknown",
                transformations_applied=applied_transforms,
                features_created=result_df.shape[1] - original_shape[1],
                features_dropped=len(set(self._original_columns) - set(result_df.columns)),
                timestamp=datetime.utcnow().isoformat(),
                status="success"
            )
            self.db.log_feature_engineering(log)
            
            logger.info(f"Feature engineering complete: {original_shape} -> {result_df.shape}")
            logger.info(f"Feature ID: {feature_id}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            
            log = FeatureLog(
                feature_id=feature_id,
                ingestion_id=ingestion_id or "unknown",
                transformations_applied=applied_transforms,
                features_created=0,
                features_dropped=0,
                timestamp=datetime.utcnow().isoformat(),
                status="failed",
                error_message=str(e)
            )
            self.db.log_feature_engineering(log)
            raise
    
    def auto_transform(self, 
                       df: pd.DataFrame,
                       ingestion_id: str = None,
                       target_column: str = None) -> pd.DataFrame:
        """
        Automatically apply sensible transformations based on data types.
        
        This implements automatic feature engineering without manual intervention.
        """
        target_col = target_column or self.config.features.target_column
        
        # Clear existing pipeline
        self._transformation_pipeline = []
        
        # Get column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Remove target from numeric columns
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        # Step 1: Handle missing values
        if df[numeric_cols].isnull().any().any():
            self.add_transformation("impute_median", columns=numeric_cols)
        
        if categorical_cols and df[categorical_cols].isnull().any().any():
            self.add_transformation("impute_mode", columns=categorical_cols)
        
        # Step 2: Handle datetime columns
        if datetime_cols:
            self.add_transformation("datetime_features", columns=datetime_cols)
        
        # Step 3: Detect and handle skewed columns
        skewed_cols = []
        for col in numeric_cols:
            if col in df.columns:
                skewness = df[col].skew()
                if abs(skewness) > 1 and df[col].min() >= 0:
                    skewed_cols.append(col)
        
        if skewed_cols:
            self.add_transformation("log_transform", columns=skewed_cols)
        
        # Step 4: Handle outliers
        self.add_transformation("clip_outliers", columns=numeric_cols, n_std=3)
        
        # Step 5: Encode categorical variables
        if categorical_cols:
            # Use one-hot for low cardinality, label encode for high cardinality
            low_cardinality = [c for c in categorical_cols if df[c].nunique() <= 10]
            high_cardinality = [c for c in categorical_cols if df[c].nunique() > 10]
            
            if low_cardinality:
                self.add_transformation("one_hot_encode", columns=low_cardinality)
            if high_cardinality:
                self.add_transformation("label_encode", columns=high_cardinality)
        
        # Step 6: Create interaction features for important numeric columns
        if len(numeric_cols) >= 2 and len(numeric_cols) <= 10:
            self.add_transformation("interaction_features", columns=numeric_cols[:5])
        
        # Step 7: Normalize numeric features
        remaining_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in remaining_numeric:
            remaining_numeric.remove(target_col)
        
        logger.info(f"Auto-configured {len(self._transformation_pipeline)} transformations")
        
        return self.transform(df, ingestion_id, target_column)
    
    def get_pipeline_summary(self) -> List[Dict[str, Any]]:
        """Get summary of the transformation pipeline."""
        return [
            {
                'step': i + 1,
                'name': step.name,
                'columns': step.columns,
                'params': step.params
            }
            for i, step in enumerate(self._transformation_pipeline)
        ]
    
    def reset_pipeline(self) -> 'FeatureEngineer':
        """Reset the transformation pipeline."""
        self._transformation_pipeline = []
        return self
    
    @property
    def feature_id(self) -> Optional[str]:
        """Get the current feature engineering ID."""
        return self._current_feature_id
    
    @staticmethod
    def list_available_transformations() -> List[str]:
        """List all registered transformations."""
        return list(TRANSFORMATION_REGISTRY.keys())


def quick_transform(df: pd.DataFrame, 
                    transformations: List[str],
                    target_column: str = None) -> pd.DataFrame:
    """
    Quick utility function to apply transformations without full pipeline setup.
    
    Args:
        df: Input DataFrame
        transformations: List of transformation names to apply
        target_column: Column to exclude from transformations
    
    Returns:
        Transformed DataFrame
    """
    engineer = FeatureEngineer()
    for t in transformations:
        engineer.add_transformation(t)
    return engineer.transform(df, target_column=target_column)


# =============================================================================
# AutoFeatureFactory — Phase 2 Automatic Feature Engineering Engine
# =============================================================================

class AutoFeatureFactory:
    """
    Automatic Feature Engineering Factory.

    Given any DataFrame it will:
      1. Auto-detect numerical columns.
      2. Apply StandardScaler to all numerical features.
      3. Identify high-correlation pairs and create polynomial (interaction)
         features for them.
      4. If a 'timestamp' column exists, generate time-series lag features.
      5. Return a **Feature Store ready** DataFrame (no NaNs in features,
         all numeric, deterministic column naming).
    """

    def __init__(
        self,
        correlation_threshold: float = 0.7,
        polynomial_degree: int = 2,
        lag_periods: Optional[List[int]] = None,
        timestamp_column: str = "timestamp",
    ):
        """
        Args:
            correlation_threshold: Absolute Pearson correlation above which
                polynomial interaction features are created.
            polynomial_degree: Max degree for polynomial expansion.
            lag_periods: List of lag periods (default [1, 2, 3]).
            timestamp_column: Name of the timestamp column to look for.
        """
        self.correlation_threshold: float = correlation_threshold
        self.polynomial_degree: int = polynomial_degree
        self.lag_periods: List[int] = lag_periods or [1, 2, 3]
        self.timestamp_column: str = timestamp_column
        self._scaler: Optional[StandardScaler] = None
        self._numeric_columns: List[str] = []
        self._high_corr_pairs: List[tuple] = []
        self._feature_log: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Detect features, engineer them, and return a Feature Store ready DF.

        Args:
            df: Input DataFrame (raw or partially processed).
            target_column: Column to preserve untouched (e.g. label).

        Returns:
            Feature-Store-ready DataFrame with all engineered features.
        """
        result = df.copy()
        target_data = None
        if target_column and target_column in result.columns:
            target_data = result[target_column].copy()
            result = result.drop(columns=[target_column])

        # 0. Sort by timestamp if it exists (needed before lag features)
        has_timestamp = self.timestamp_column in result.columns
        if has_timestamp:
            try:
                result[self.timestamp_column] = pd.to_datetime(
                    result[self.timestamp_column], errors="coerce"
                )
                result = result.sort_values(self.timestamp_column).reset_index(drop=True)
            except Exception:
                has_timestamp = False

        # De-duplicate columns that may exist from previous runs
        result = result.loc[:, ~result.columns.duplicated()]

        # 1. Detect numerical columns
        self._numeric_columns = result.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"AutoFeatureFactory: detected {len(self._numeric_columns)} numerical columns")

        # Impute NaNs in numeric cols with median before scaling
        for col in self._numeric_columns:
            if result[col].isnull().any():
                result[col] = result[col].fillna(result[col].median())

        # 2. StandardScaler on numerical columns
        if self._numeric_columns:
            self._scaler = StandardScaler()
            scaled_values = self._scaler.fit_transform(result[self._numeric_columns])
            scaled_cols = [f"{c}_scaled" for c in self._numeric_columns]
            scaled_df = pd.DataFrame(scaled_values, columns=scaled_cols, index=result.index)
            result = pd.concat([result, scaled_df], axis=1)
            logger.info(f"AutoFeatureFactory: applied StandardScaler → {len(scaled_cols)} scaled features")

        # 3. Polynomial features for high-correlation pairs
        self._high_corr_pairs = self._find_high_correlation_pairs(result)
        poly_count = 0
        for col_a, col_b in self._high_corr_pairs:
            for degree in range(2, self.polynomial_degree + 1):
                result[f"{col_a}_x_{col_b}_deg{degree}"] = (
                    result[col_a] * result[col_b]
                ) ** (degree / 2)
                poly_count += 1
        if poly_count:
            logger.info(
                f"AutoFeatureFactory: created {poly_count} polynomial features "
                f"from {len(self._high_corr_pairs)} high-corr pairs"
            )

        # 4. Time-series lag features if timestamp exists
        lag_count = 0
        if has_timestamp:
            for col in self._numeric_columns:
                for lag in self.lag_periods:
                    result[f"{col}_lag{lag}"] = result[col].shift(lag)
                    lag_count += 1
            # Fill NaN introduced by shifts
            result = result.fillna(0)
            logger.info(f"AutoFeatureFactory: created {lag_count} lag features")

        # 5. Drop non-numeric leftovers (e.g. original timestamp, categorical)
        #    to make the result Feature-Store ready
        non_numeric = result.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            result = result.drop(columns=non_numeric)

        # Handle any residual NaN
        result = result.fillna(0)

        # Re-attach target if it was separated
        if target_data is not None:
            result[target_column] = target_data.values[: len(result)]

        # Log summary
        self._feature_log = {
            "numeric_columns_detected": len(self._numeric_columns),
            "scaled_features": len(self._numeric_columns),
            "high_corr_pairs": len(self._high_corr_pairs),
            "polynomial_features": poly_count,
            "lag_features": lag_count,
            "total_features": len(result.columns),
            "timestamp_detected": has_timestamp,
        }
        logger.info(f"AutoFeatureFactory: Feature Store DF ready — shape {result.shape}")

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_high_correlation_pairs(self, df: pd.DataFrame) -> List[tuple]:
        """Return pairs of numeric columns with |corr| >= threshold."""
        num_df = df.select_dtypes(include=[np.number])
        # De-duplicate columns (can happen on re-runs)
        num_df = num_df.loc[:, ~num_df.columns.duplicated()]
        if num_df.shape[1] < 2:
            return []
        corr_matrix = num_df.corr().abs()
        pairs = []
        cols = corr_matrix.columns.tolist()
        for i, col_a in enumerate(cols):
            for col_b in cols[i + 1 :]:
                val = corr_matrix.loc[col_a, col_b]
                # Safely handle scalar vs Series
                if np.isscalar(val) and val >= self.correlation_threshold:
                    pairs.append((col_a, col_b))
        return pairs

    @property
    def feature_summary(self) -> Dict[str, Any]:
        """Return a summary dict of the last fit_transform run."""
        return self._feature_log
