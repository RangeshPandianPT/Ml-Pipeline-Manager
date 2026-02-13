"""
API endpoint tests
"""

import pytest
from fastapi.testclient import TestClient
import pandas as pd
import sys
from pathlib import Path
import io

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from api.main import app
from ingestion import create_sample_dataset


client = TestClient(app)


class TestAPIEndpoints:
    """Test FastAPI endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    def test_list_models(self):
        """Test list models endpoint"""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "models" in data
    
    def test_pipeline_state(self):
        """Test pipeline state endpoint"""
        response = client.get("/pipeline/state")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "state" in data
    
    def test_train_endpoint(self):
        """Test model training endpoint"""
        # Create sample data
        df = create_sample_dataset(n_samples=100, n_features=5)
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        # Prepare file upload
        files = {
            "file": ("test_data.csv", csv_buffer.getvalue(), "text/csv")
        }
        
        # Prepare request data
        data = {
            "target_column": "target",
            "model_type": "random_forest",
            "auto_features": True,
            "check_drift": False,
            "validation_split": 0.2
        }
        
        # Make request
        response = client.post("/train", data=data, files=files)
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "model_metrics" in result
        assert "model_name" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
