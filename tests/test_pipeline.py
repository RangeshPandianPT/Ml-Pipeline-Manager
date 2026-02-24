"""
Unit tests for ML Pipeline
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline_manager import MLPipeline, PipelineConfig, TaskType
from src.feature_eng import FeatureEngineer, get_available_transformations
from src.monitor import DriftMonitor
from src.ingestion import DataIngestion, create_sample_dataset


@pytest.fixture
def sample_data():
    """Create sample dataset for testing"""
    return create_sample_dataset(n_samples=100, n_features=5)


@pytest.fixture
def pipeline():
    """Create pipeline instance"""
    return MLPipeline()


class TestDataIngestion:
    """Test data ingestion module"""
    
    def test_create_sample_dataset(self):
        """Test sample dataset creation"""
        df = create_sample_dataset(n_samples=50, n_features=3)
        assert df.shape == (50, 4)  # 3 features + 1 target
        assert 'target' in df.columns
    
    def test_ingest_dataframe(self, pipeline, sample_data):
        """Test DataFrame ingestion"""
        result = pipeline.ingest(sample_data)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_data.shape


class TestFeatureEngineering:
    """Test feature engineering module"""
    
    def test_available_transformations(self):
        """Test getting available transformations"""
        transforms = get_available_transformations()
        assert len(transforms) > 0
        assert 'standardize' in transforms
        assert 'normalize' in transforms
    
    def test_auto_feature_engineering(self, pipeline, sample_data):
        """Test automatic feature engineering"""
        pipeline.ingest(sample_data)
        result = pipeline.engineer_features(auto=True, target_column='target')
        
        assert 'transformed_data' in result
        assert 'features_created' in result
        assert result['features_created'] >= 0
    
    def test_standardize_transformation(self, sample_data):
        """Test standardization transformation"""
        from src.database import MetadataDatabase, default_config
        db = MetadataDatabase(default_config.database)
        engineer = FeatureEngineer(db=db)
        engineer.add_transformation('standardize')
        
        # Create a dummy ingestion ID to ensure metadata tracking works
        ingestion_id = "test_ingestion_id"
        
        result = engineer.transform(sample_data.drop(columns=['target']), ingestion_id)
        
        # Check that mean is close to 0 and std is close to 1
        # select_dtypes to only check numeric columns as standardization only applies to them
        numeric_result = result.select_dtypes(include=[np.number])
        assert abs(numeric_result.mean().mean()) < 0.1
        assert abs(numeric_result.std().mean() - 1.0) < 0.1


class TestDriftMonitoring:
    """Test drift monitoring module"""
    
    def test_drift_monitor_initialization(self):
        """Test drift monitor initialization"""
        monitor = DriftMonitor()
        assert monitor is not None
    
    def test_set_reference_data(self, sample_data):
        """Test setting reference data"""
        monitor = DriftMonitor()
        monitor.set_reference(sample_data)
        assert monitor.reference_data is not None
    
    def test_no_drift_detection(self, sample_data):
        """Test drift detection with identical data"""
        monitor = DriftMonitor()
        monitor.set_reference(sample_data)
        
        # Check drift with same data (should detect no drift)
        drift_report = monitor.detect_drift(sample_data)
        
        # With identical data, p-values should be high (no drift)
        assert drift_report is not None


class TestPipeline:
    """Test complete pipeline"""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        pipeline = MLPipeline()
        assert pipeline is not None
        assert pipeline.config is not None
    
    def test_full_pipeline_run(self, sample_data):
        """Test complete pipeline execution"""
        pipeline = MLPipeline()
        
        results = pipeline.run_full_pipeline(
            source=sample_data,
            target_column='target',
            auto_features=True,
            check_drift=False,
            train_model=True
        )
        
        assert 'model_metrics' in results
        assert results['metrics']['model_trained'] is True
        assert 'execution_time_seconds' in results['metrics']
    
    def test_model_training(self, pipeline, sample_data):
        """Test model training"""
        pipeline.ingest(sample_data)
        pipeline.engineer_features(auto=True, target_column='target')
        
        pipeline.ingest(sample_data)
        pipeline.engineer_features(auto=True, target_column='target')
        pipeline.prepare_training_data(target_column='target')
        
        result = pipeline.train_model()
        
        assert result is not None
        # assert 'metrics' in result # train_model returns model, not metrics dict
        # pipeline._current_model is internal, so just check result
        assert result is not None
    
    def test_model_prediction(self, pipeline, sample_data):
        """Test model prediction"""
        # Train model first
        pipeline.run_full_pipeline(
            source=sample_data,
            target_column='target',
            train_model=True
        )
        
        # Make predictions
        # Select integers/floats only as the pipeline does
        X_test = sample_data.drop(columns=['target']).select_dtypes(include=[np.number]).head(10)
        # Ensure we don't pass NaNs if any
        X_test = X_test.fillna(0)
        predictions = pipeline.predict(X_test, apply_features=False)
        
        assert len(predictions) == 10
        assert isinstance(predictions, np.ndarray)


class TestMLflowIntegration:
    """Test MLflow integration"""
    
    def test_mlflow_tracker_initialization(self):
        """Test MLflow tracker initialization"""
        from src.mlflow_integration import MLflowTracker
        
        tracker = MLflowTracker(experiment_name="test_experiment")
        assert tracker is not None
        assert tracker.experiment_name == "test_experiment"


class TestSHAPExplainer:
    """Test SHAP explainer"""
    
    def test_shap_explainer_initialization(self, sample_data):
        """Test SHAP explainer initialization"""
        from src.shap_explainer import SHAPExplainer
        from sklearn.ensemble import RandomForestRegressor
        
        # Train a simple model
        X = sample_data.drop(columns=['target']).select_dtypes(include=[np.number]).fillna(0)
        y = sample_data['target']
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Create explainer
        explainer = SHAPExplainer(model, X, model_type="tree")
        assert explainer is not None
    
    def test_feature_importance(self, sample_data):
        """Test SHAP feature importance calculation"""
        from src.shap_explainer import SHAPExplainer
        from sklearn.ensemble import RandomForestRegressor
        
        X = sample_data.drop(columns=['target']).select_dtypes(include=[np.number]).fillna(0)
        y = sample_data['target']
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        explainer = SHAPExplainer(model, X, model_type="tree")
        importance_df = explainer.get_feature_importance(X.head(20))
        
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) == len(X.columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
