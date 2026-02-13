"""
Configuration module for ML Pipeline.
Centralized settings for database, feature engineering, and monitoring.
Domain-agnostic configuration suitable for any ML project.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    db_type: str = "sqlite"  # "sqlite" or "postgresql"
    db_name: str = "pipeline_metadata.db"  # Metadata database name
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "postgres"
    db_password: str = ""
    
    @property
    def connection_string(self) -> str:
        if self.db_type == "sqlite":
            return f"sqlite:///{self.db_name}"
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


@dataclass
class FeatureConfig:
    """Configuration for feature engineering transformations."""
    # Define which transformations to apply to which columns
    # Format: {"column_name": ["transformation1", "transformation2"]}
    numerical_transformations: Dict[str, List[str]] = field(default_factory=lambda: {
        "default": ["standardize", "log_transform"]
    })
    categorical_transformations: Dict[str, List[str]] = field(default_factory=lambda: {
        "default": ["one_hot_encode"]
    })
    # Columns to drop
    drop_columns: List[str] = field(default_factory=list)
    # Target column name
    target_column: str = "target"
    # Date columns for temporal features
    date_columns: List[str] = field(default_factory=list)


@dataclass
class MonitorConfig:
    """Configuration for data drift monitoring."""
    # Significance level for K-S test (alpha)
    ks_significance_level: float = 0.05
    # Minimum samples required for drift detection
    min_samples: int = 30
    # Columns to monitor for drift
    monitor_columns: List[str] = field(default_factory=list)  # Empty means monitor all numerical
    # Drift threshold - percentage of columns with drift to trigger alert
    drift_threshold: float = 0.3
    # Enable automatic retraining on drift
    auto_retrain: bool = True


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    # Data paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    raw_data_path: Path = field(default_factory=lambda: Path("data/raw"))
    processed_data_path: Path = field(default_factory=lambda: Path("data/processed"))
    model_path: Path = field(default_factory=lambda: Path("models"))
    
    # Pipeline settings
    batch_size: int = 1000
    validation_split: float = 0.2
    random_state: int = 42
    
    # Model settings
    model_type: str = "xgboost"  # "random_forest", "gradient_boosting", "logistic_regression", "xgboost"
    
    # Logging
    log_level: str = "INFO"
    log_file: Path = field(default_factory=lambda: Path("logs/pipeline.log"))
    
    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)


# Default configuration instance
default_config = PipelineConfig()


def load_config_from_dict(config_dict: Dict[str, Any]) -> PipelineConfig:
    """Load configuration from a dictionary."""
    db_config = DatabaseConfig(**config_dict.get("database", {}))
    feature_config = FeatureConfig(**config_dict.get("features", {}))
    monitor_config = MonitorConfig(**config_dict.get("monitor", {}))
    
    pipeline_dict = {k: v for k, v in config_dict.items() 
                     if k not in ["database", "features", "monitor"]}
    
    return PipelineConfig(
        **pipeline_dict,
        database=db_config,
        features=feature_config,
        monitor=monitor_config
    )


def load_config_from_yaml(yaml_path: str) -> PipelineConfig:
    """Load configuration from a YAML file."""
    import yaml
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return load_config_from_dict(config_dict)
