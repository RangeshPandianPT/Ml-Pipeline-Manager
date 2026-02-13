# Project Refactoring and GPU Support Walkthrough

## Overview
This walkthrough covers the recent updates to refactor the project structure, fix CI/CD failures, and enable GPU support for model training.

## Changes

### 1. Project Structure Refactoring
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
- **API Tests**: All 5 tests passed, confirming endpoint health, model management, and state retrieval.
