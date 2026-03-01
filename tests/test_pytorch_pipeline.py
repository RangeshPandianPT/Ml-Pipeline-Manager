"""
Unit tests for PyTorch Deep Learning integration
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline_manager import MLPipeline, PipelineConfig
from src.pytorch_model import PyTorchMLPWrapper
from src.ingestion import create_sample_dataset


@pytest.fixture
def sample_regression_data():
    """Create sample dataset for regression"""
    df = create_sample_dataset(n_samples=100, n_features=4)
    df['target'] = df['feature_1'] * 2.5 + df['feature_2'] * -1.5 + np.random.randn(100)
    return df


@pytest.fixture
def sample_classification_data():
    """Create sample dataset for classification"""
    df = create_sample_dataset(n_samples=100, n_features=4)
    # create a simple binary target
    df['target'] = (df['feature_1'] + df['feature_2'] > 0).astype(int)
    return df


class TestPyTorchWrapper:
    """Test PyTorch wrapper implementation independent of the pipeline"""
    
    def test_wrapper_initialization(self):
        """Test wrapper initialization defaults"""
        model = PyTorchMLPWrapper()
        assert model.task_type == "regression"
        assert model.hidden_sizes == [64, 32]
        assert model.lr == 0.001
        
    def test_wrapper_regression_fit_predict(self, sample_regression_data):
        """Test simple fit and predict for regression"""
        from sklearn.preprocessing import StandardScaler
        X = sample_regression_data.drop(columns=['target'])
        y = sample_regression_data['target']
        
        # Impute missing values since the sample dataset includes them
        X = X.fillna(0)
        
        # Scale for stable PyTorch training
        X = StandardScaler().fit_transform(X)
        
        # Fast training settings for tests
        model = PyTorchMLPWrapper(task_type="regression", epochs=2, hidden_sizes=[16], device="cpu")
        model.fit(X, y)
        preds = model.predict(X)
        
        assert len(preds) == len(X)
        assert isinstance(preds, np.ndarray)
        assert not np.isnan(preds).any()
        
    def test_wrapper_classification_fit_predict(self, sample_classification_data):
        """Test simple fit, predict, predict_proba for classification"""
        from sklearn.preprocessing import StandardScaler
        X = sample_classification_data.drop(columns=['target'])
        y = sample_classification_data['target']
        
        X = X.fillna(0)
        X = StandardScaler().fit_transform(X)
        
        model = PyTorchMLPWrapper(task_type="classification", epochs=2, hidden_sizes=[16], device="cpu")
        model.fit(X, y)
        
        preds = model.predict(X)
        probas = model.predict_proba(X)
        
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0.0, 1.0})
        assert probas.shape == (len(X), 2)
        assert np.all(probas >= 0) and np.all(probas <= 1)


class TestPyTorchPipeline:
    """Test full pipeline integration with PyTorch"""
    
    def test_pipeline_pytorch_regression(self, sample_regression_data):
        """Test running pipeline end-to-end with PyTorch for regression"""
        config = PipelineConfig()
        config.model_type = "pytorch_nn"
        
        pipeline = MLPipeline(config)
        
        results = pipeline.run_full_pipeline(
            source=sample_regression_data,
            target_column='target',
            auto_features=True, # Enable to standardize
            check_drift=False,
            train_model=True
        )
        
        assert results['status'] == 'completed'
        assert results['model_metrics'] is not None
        assert 'mse' in results['model_metrics']
        assert isinstance(pipeline.model, PyTorchMLPWrapper)
        assert pipeline.model.task_type == "regression"

    def test_pipeline_pytorch_classification(self, sample_classification_data):
        """Test running pipeline end-to-end with PyTorch for classification"""
        config = PipelineConfig()
        config.model_type = "pytorch_nn"
        
        pipeline = MLPipeline(config)
        
        results = pipeline.run_full_pipeline(
            source=sample_classification_data,
            target_column='target',
            auto_features=True, # Enable to standardize
            check_drift=False,
            train_model=True
        )
        
        assert results['status'] == 'completed'
        assert results['model_metrics'] is not None
        assert 'accuracy' in results['model_metrics']
        # Don't strictly assert roc_auc since the 20 test samples might lack positive class 
        # due to random noise in small generated dataset.
        assert isinstance(pipeline.model, PyTorchMLPWrapper)
        assert pipeline.model.task_type == "classification"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
