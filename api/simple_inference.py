import joblib
import pandas as pd
import pathlib
import uvicorn
import glob
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List

app = FastAPI(
    title="ML Pipeline API",
    description="API for serving predictions from the latest trained model.",
    version="1.0.0"
)

# -----------------------------------------------------------------------------
# Model Loading Logic
# -----------------------------------------------------------------------------
MODEL_DIR = "models"
model = None

def load_latest_model():
    """Finds and loads the latest .joblib model from the models directory."""
    global model
    try:
        # Get list of model files
        model_files = glob.glob(os.path.join(MODEL_DIR, "MODEL_*.joblib"))
        
        if not model_files:
            print("No models found in models/ directory.")
            return None
            
        # Sort by modification time (or name if timestamp is in name)
        # Using name is safer given the format MODEL_YYYYMMDD...
        latest_file = sorted(model_files)[-1]
        print(f"Loading latest model: {latest_file}")
        
        model = joblib.load(latest_file)
        return latest_file
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load model on startup
latest_model_path = load_latest_model()

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

class PredictRequest(BaseModel):
    features: List[Dict[str, Any]]

    class Config:
        schema_extra = {
            "example": {
                "features": [
                    {"feature1": 0.5, "feature2": 1.2, "category": "A"},
                    {"feature1": 0.1, "feature2": 0.8, "category": "B"}
                ]
            }
        }

@app.get("/")
def health_check():
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "active_model": os.path.basename(latest_model_path) if latest_model_path else None
    }

@app.post("/predict")
def predict(request: PredictRequest):
    global model
    if not model:
        # Try loading again in case a new model was trained
        load_latest_model()
        if not model:
            raise HTTPException(status_code=503, detail="No model available to serve predictions.")
    
    try:
        # Convert list of dicts to DataFrame
        input_df = pd.DataFrame(request.features)
        
        # Note: In a real production scenario, you must apply the same 
        # Feature Engineering steps here as in the pipeline.
        # For this demo, we assume the input matches the model's expected features 
        # OR the model pipeline includes the feature engineering steps.
        
        # Filter for numeric columns if model expects only numeric
        # (Naive approach - simply passing everything to predict)
        predictions = model.predict(input_df)
        
        return {
            "predictions": predictions.tolist(),
            "model_version": os.path.basename(latest_model_path)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/reload")
def reload_model():
    """Force reload of the latest model (useful after retraining)."""
    layout_path = load_latest_model()
    if layout_path:
        return {"message": f"Successfully reloaded model: {os.path.basename(layout_path)}"}
    else:
        raise HTTPException(status_code=404, detail="No model found to load.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
