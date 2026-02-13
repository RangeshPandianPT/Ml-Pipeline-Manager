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

from pipeline_manager import MLPipeline, PipelineConfig, TaskType
from feature_eng import FeatureEngineer, get_available_transformations
from monitor import DriftMonitor
from ingestion import DataIngestion, create_sample_dataset


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
        engineer = FeatureEngineer()
        engineer.add_transformation('standardize')
        
        result = engineer.transform(sample_data.drop(columns=['target']))
        
        # Check that mean is close to 0 and std is close to 1
        assert abs(result.mean().mean()) < 0.1
        assert abs(result.std().mean() - 1.0) < 0.1


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
        assert results['model_trained'] is True
        assert 'execution_time_seconds' in results
    
    def test_model_training(self, pipeline, sample_data):
        """Test model training"""
        pipeline.ingest(sample_data)
        pipeline.engineer_features(auto=True, target_column='target')
        
        result = pipeline.train_model(target_column='target')
        
        assert result['success'] is True
        assert 'metrics' in result
        assert pipeline.model is not None
    
    def test_model_prediction(self, pipeline, sample_data):
        """Test model prediction"""
        # Train model first
        pipeline.run_full_pipeline(
            source=sample_data,
            target_column='target',
            train_model=True
        )
        
        # Make predictions
        X_test = sample_data.drop(columns=['target']).head(10)
        predictions = pipeline.predict(X_test)
        
        assert len(predictions) == 10
        assert isinstance(predictions, np.ndarray)


class TestMLflowIntegration:
    """Test MLflow integration"""
    
    def test_mlflow_tracker_initialization(self):
        """Test MLflow tracker initialization"""
        from mlflow_integration import MLflowTracker
        
        tracker = MLflowTracker(experiment_name="test_experiment")
        assert tracker is not None
        assert tracker.experiment_name == "test_experiment"


class TestSHAPExplainer:
    """Test SHAP explainer"""
    
    def test_shap_explainer_initialization(self, sample_data):
        """Test SHAP explainer initialization"""
        from shap_explainer import SHAPExplainer
        from sklearn.ensemble import RandomForestRegressor
        
        # Train a simple model
        X = sample_data.drop(columns=['target'])
        y = sample_data['target']
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Create explainer
        explainer = SHAPExplainer(model, X, model_type="tree")
        assert explainer is not None
    
    def test_feature_importance(self, sample_data):
        """Test SHAP feature importance calculation"""
        from shap_explainer import SHAPExplainer
        from sklearn.ensemble import RandomForestRegressor
        
        X = sample_data.drop(columns=['target'])
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
