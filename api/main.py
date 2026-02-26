"""
FastAPI REST API for ML Pipeline
Production-ready API for model training, prediction, and monitoring
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from fastapi.security import OAuth2PasswordRequestForm
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import pandas as pd
import numpy as np
import io
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline_manager import MLPipeline, PipelineConfig, TaskType
from src.monitor import DriftReport

from api.security import (
    Token,
    create_access_token,
    get_current_active_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    FAKE_USERS_DB,
    verify_password,
    get_user,
    get_current_user
)
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Pipeline API",
    description="Production-ready MLOps API for automated feature engineering, drift monitoring, and model training",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup SlowAPI Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = MLPipeline()

# ==================== Request/Response Models ====================

class TrainRequest(BaseModel):
    """Request model for training endpoint"""
    target_column: str = Field(..., description="Name of the target column")
    model_type: str = Field(default="random_forest", description="Type of model to train")
    auto_features: bool = Field(default=True, description="Enable automatic feature engineering")
    check_drift: bool = Field(default=False, description="Check for drift before training")
    validation_split: float = Field(default=0.2, ge=0.0, le=0.5, description="Validation split ratio")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        allowed = ['random_forest', 'gradient_boosting', 'linear', 'logistic_regression', 'xgboost']
        if v not in allowed:
            raise ValueError(f"model_type must be one of {allowed}")
        return v

class PredictRequest(BaseModel):
    """Request model for prediction endpoint"""
    data: List[Dict[str, Any]] = Field(..., description="List of samples to predict")
    model_name: Optional[str] = Field(None, description="Specific model to use (uses active model if None)")

class DriftCheckRequest(BaseModel):
    """Request model for drift detection endpoint"""
    set_as_reference: bool = Field(default=False, description="Set uploaded data as new reference")

class TrainResponse(BaseModel):
    """Response model for training endpoint"""
    success: bool
    message: str
    model_metrics: Optional[Dict[str, float]] = None
    model_name: Optional[str] = None
    features_created: int = 0
    training_time_seconds: float = 0.0
    drift_detected: Optional[bool] = None

class PredictResponse(BaseModel):
    """Response model for prediction endpoint"""
    success: bool
    predictions: List[Union[float, int, str]]
    probabilities: Optional[List[List[float]]] = None
    model_used: str
    num_samples: int

class DriftResponse(BaseModel):
    """Response model for drift detection endpoint"""
    success: bool
    drift_detected: bool
    should_retrain: bool
    num_drifted_columns: int
    drifted_columns: List[str]
    drift_summary: Dict[str, Any]

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    version: str
    pipeline_state: str
    models_available: List[str]
    timestamp: str

# ==================== Helper Functions ====================

def parse_uploaded_file(file: UploadFile) -> pd.DataFrame:
    """Parse uploaded file into DataFrame"""
    try:
        content = file.file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.BytesIO(content))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise ValueError(f"Unsupported file format: {file.filename}")
        
        logger.info(f"Parsed file {file.filename}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    except Exception as e:
        logger.error(f"Error parsing file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")

def get_available_models() -> List[str]:
    """Get list of available trained models"""
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    
    model_files = list(models_dir.glob("*.joblib"))
    return [m.stem for m in model_files]

# ==================== API Endpoints ====================

@app.get("/", response_model=Dict[str, str])
@limiter.limit("20/minute")
async def root(request: Request):
    """Root endpoint"""
    return {
        "message": "ML Pipeline API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
@limiter.limit("20/minute")
async def health_check(request: Request):
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        pipeline_state=pipeline.state.value,
        models_available=get_available_models(),
        timestamp=datetime.now().isoformat()
    )

@app.post("/token", response_model=Token)
@limiter.limit("10/minute")
async def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(FAKE_USERS_DB, form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

def train_request_form(
    target_column: str = Form(..., description="Name of the target column"),
    model_type: str = Form("random_forest", description="Type of model to train"),
    auto_features: bool = Form(True, description="Enable automatic feature engineering"),
    check_drift: bool = Form(False, description="Check for drift before training"),
    validation_split: float = Form(0.2, ge=0.0, le=0.5, description="Validation split ratio")
) -> TrainRequest:
    return TrainRequest(
        target_column=target_column,
        model_type=model_type,
        auto_features=auto_features,
        check_drift=check_drift,
        validation_split=validation_split
    )

@app.post("/train", response_model=TrainResponse)
@limiter.limit("5/minute")
async def train_model(
    request_obj: Request,
    background_tasks: BackgroundTasks,
    request: TrainRequest = Depends(train_request_form),
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Train a new model on uploaded data
    
    - **file**: CSV, JSON, or Excel file with training data
    - **target_column**: Name of the target column
    - **model_type**: Type of model to train
    - **auto_features**: Enable automatic feature engineering
    - **check_drift**: Check for drift before training
    """
    try:
        start_time = datetime.now()
        
        # Parse uploaded file
        df = parse_uploaded_file(file)
        
        # Validate target column
        if request.target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{request.target_column}' not found in data"
            )
        
        # Update pipeline config
        pipeline.config.model_type = request.model_type
        pipeline.config.validation_split = request.validation_split
        
        # Run full pipeline
        logger.info(f"Starting training pipeline with {request.model_type}")
        results = pipeline.run_full_pipeline(
            source=df,
            target_column=request.target_column,
            auto_features=request.auto_features,
            check_drift=request.check_drift,
            train_model=True
        )
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Generate model name
        model_name = f"{request.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save model
        pipeline.save_model(model_name=model_name)
        
        logger.info(f"Training completed successfully: {model_name}")
        
        return TrainResponse(
            success=True,
            message=f"Model trained successfully: {model_name}",
            model_metrics=results.get('model_metrics', {}),
            model_name=model_name,
            features_created=results.get('features_created', 0),
            training_time_seconds=execution_time,
            drift_detected=results.get('drift_detected', None)
        )
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict", response_model=PredictResponse)
@limiter.limit("10/minute")
async def predict(
    request_obj: Request, 
    request: PredictRequest,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Make predictions using trained model
    
    - **data**: List of samples to predict (JSON format)
    - **model_name**: Specific model to use (optional)
    """
    try:
        # Convert request data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Load model if specified
        if request.model_name:
            model_path = Path("models") / f"{request.model_name}.joblib"
            if not model_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{request.model_name}' not found"
                )
            pipeline.load_model(str(model_path))
            model_used = request.model_name
        else:
            if pipeline.model is None:
                raise HTTPException(
                    status_code=400,
                    detail="No active model. Train a model first or specify model_name"
                )
            model_used = "active_model"
        
        # Make predictions
        predictions = pipeline.predict(df)
        
        # Get probabilities if classification
        probabilities = None
        if hasattr(pipeline.model, 'predict_proba'):
            try:
                probabilities = pipeline.model.predict_proba(df).tolist()
            except:
                pass
        
        logger.info(f"Predictions made for {len(predictions)} samples")
        
        return PredictResponse(
            success=True,
            predictions=predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            probabilities=probabilities,
            model_used=model_used,
            num_samples=len(predictions)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/drift/check", response_model=DriftResponse)
@limiter.limit("5/minute")
async def check_drift(
    request_obj: Request,
    request: DriftCheckRequest,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Check for data drift in uploaded data
    
    - **file**: CSV, JSON, or Excel file with new data
    - **set_as_reference**: Set this data as new reference baseline
    """
    try:
        # Parse uploaded file
        df = parse_uploaded_file(file)
        
        # Check drift
        drift_report = pipeline.monitor_drift(
            current_data=df,
            set_as_reference=request.set_as_reference
        )
        
        if drift_report is None:
            raise HTTPException(
                status_code=400,
                detail="No reference data set. Upload reference data first with set_as_reference=true"
            )
        
        # Extract drift information
        drift_detected = drift_report.overall_drift_detected
        should_retrain = pipeline.drift_monitor.should_retrain()
        
        drifted_columns = [
            col for col, result in drift_report.column_results.items()
            if result.drift_detected
        ]
        
        drift_summary = {
            "total_columns_checked": len(drift_report.column_results),
            "num_drifted_columns": len(drifted_columns),
            "drift_percentage": len(drifted_columns) / len(drift_report.column_results) * 100,
            "recommendation": drift_report.recommendation
        }
        
        logger.info(f"Drift check completed: {drift_detected}, should_retrain: {should_retrain}")
        
        return DriftResponse(
            success=True,
            drift_detected=drift_detected,
            should_retrain=should_retrain,
            num_drifted_columns=len(drifted_columns),
            drifted_columns=drifted_columns,
            drift_summary=drift_summary
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Drift check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Drift check failed: {str(e)}")

@app.get("/models", response_model=Dict[str, Any])
@limiter.limit("20/minute")
async def list_models(
    request_obj: Request,
    current_user: dict = Depends(get_current_active_user)
):
    """List all available trained models"""
    try:
        models = get_available_models()
        
        # Get model details
        model_details = []
        for model_name in models:
            model_path = Path("models") / f"{model_name}.joblib"
            stat = model_path.stat()
            
            model_details.append({
                "name": model_name,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        return {
            "success": True,
            "count": len(models),
            "models": model_details
        }
    
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.delete("/models/{model_name}")
@limiter.limit("5/minute")
async def delete_model(
    request_obj: Request,
    model_name: str,
    current_user: dict = Depends(get_current_active_user)
):
    """Delete a specific model"""
    try:
        model_path = Path("models") / f"{model_name}.joblib"
        
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found"
            )
        
        model_path.unlink()
        logger.info(f"Model deleted: {model_name}")
        
        return {
            "success": True,
            "message": f"Model '{model_name}' deleted successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")

@app.get("/pipeline/state")
@limiter.limit("20/minute")
async def get_pipeline_state(
    request_obj: Request,
    current_user: dict = Depends(get_current_active_user)
):
    """Get current pipeline state and statistics"""
    try:
        return {
            "success": True,
            "state": pipeline.state.value,
            "task_type": pipeline.task_type.value if pipeline.task_type else None,
            "has_model": pipeline.model is not None,
            "has_reference_data": pipeline.drift_monitor.reference_data is not None if pipeline.drift_monitor else False,
            "config": {
                "model_type": pipeline.config.model_type,
                "validation_split": pipeline.config.validation_split,
                "drift_threshold": pipeline.config.monitor.drift_threshold
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get pipeline state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline state: {str(e)}")

# ==================== Startup/Shutdown Events ====================

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    logger.info("ML Pipeline API starting up...")
    logger.info(f"Available models: {get_available_models()}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("ML Pipeline API shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
