# Project Refactoring and GPU Support Walkthrough

## Overview
This walkthrough covers the recent updates to secure the FastAPI application by adding JSON Web Token (JWT) authentication and Rate Limiting, as well as previous refactoring and GPU support updates.

## Changes

### 1. API Security (Authentication & Rate Limiting)
- **JWT Authentication**: 
  - Created `api/security.py` to handle password hashing (`passlib[bcrypt]`) and JWT token generation/validation (`python-jose[cryptography]`).
  - Added a `POST /token` endpoint in `api/main.py` for clients to obtain an access token using OAuth2 password flow.
  - Secured endpoints (`/train`, `/predict`, `/drift/check`, `/models`, `/pipeline/state`) using FastAPI's `Depends(get_current_active_user)`.
- **Rate Limiting**:
  - Integrated `slowapi` to protect against brute-force and DDoS attacks.
  - Applied varied rate limits: `"20/minute"` for health/info endpoints, `"10/minute"` for login/predict, and `"5/minute"` for heavy operations like training and drift checking.
- **Testing**: Updated `tests/test_api.py` to obtain and pass authentication headers, and added tests for unauthorized access verification.

### 2. Project Structure Refactoring
- **Source Code**: Moved all core logic to `src/`.
- **Scripts**: Moved executables to `scripts/`.
- **Documentation**: Consolidated in `docs/`.
- **Imports**: Updated all imports to use absolute package references (e.g., `src.pipeline_manager`).

### 2. CI/CD Fixes
- **Test Import Resolution**: Fixed `ModuleNotFoundError` in tests by adjusting imports and setting up `setup.py` (implied or PYTHONPATH handling).
- **Test Logic Fixes**:
    - Resolved "Ambiguous DataFrame" errors in `prepare_training_data` and `drift_monitor` by fixing boolean checks.
    - Fixed API 500 errors by adding `model` and `task_type` properties to `MLPipeline`.
    - Added missing `model_trained` key to pipeline metrics.
    - Updated `test_pipeline.py` to match the current API and model return types.
    - **Streamlit Compatibility**: Updated `engineer_features` to return a dictionary (with `transformed_data` key) instead of a DataFrame, resolving a KeyError in the dashboard.

### 3. GPU Support
- **XGBoost Integration**: Added `xgboost` as a supported model type in `src/pipeline_manager.py`.
- **GPU Configuration**: Configured `XGBoost` to use `device='cuda'` and `tree_method='hist'` by default.
- **Default Model**: Updated `src/config.py` to use `xgboost` as the default model type, ensuring GPU usage out-of-the-box.

## Usage

### Running Locally
To run the pipeline with the new structure:
```bash
python -m scripts.run_pipeline
```

### Running Tests
To verify all changes:
```bash
pytest tests/
```

### GPU Requirements
Ensure you have:
- NVIDIA GPU with CUDA support.
- `xgboost` installed with GPU support (`pip install xgboost`).
- CUDA toolkit compatible with your drivers.

## Verification Results
- **Pipeline Tests**: All 15 tests passed, confirming data ingestion, feature engineering, model training, and drift monitoring.
- **API Tests**: Tests fully updated to incorporate OAuth2 Bearer token authentication headers. All 6 tests (including `test_unauthorized_access`) passed, confirming endpoint security, model management, and state retrieval.
