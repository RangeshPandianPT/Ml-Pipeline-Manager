# ML Pipeline Pro v2.0 🚀

**Next-Generation MLOps Platform** | Production-Ready ML Pipeline with Advanced Features

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 What's New in v2.0

- ✨ **FastAPI REST API** - Production-ready API for model serving
- 📊 **Premium Streamlit Dashboard** - Modern, interactive UI with real-time monitoring
- 🔬 **MLflow Integration** - Experiment tracking and model registry
- 🧠 **SHAP Explanations** - Model interpretability and feature importance
- 🐳 **Docker Support** - Containerized deployment with docker-compose
- 🔄 **CI/CD Pipeline** - Automated testing and deployment with GitHub Actions
- 📈 **Advanced Drift Detection** - SHAP-based feature drift monitoring
- 🧪 **Comprehensive Testing** - Unit tests, integration tests, and API tests

## 🎯 Features

### Core Capabilities
- **Automated Data Ingestion**: CSV, JSON, Excel, and DataFrame support
- **Smart Feature Engineering**: 20+ transformations with auto-detection
- **Data Drift Detection**: K-S test, PSI, and SHAP-based drift monitoring
- **Auto Model Retraining**: Triggered by drift detection
- **Model Explainability**: SHAP values and feature importance
- **Experiment Tracking**: MLflow integration for versioning and comparison
- **Production API**: FastAPI with Swagger documentation
- **Interactive Dashboard**: Premium Streamlit UI

### Advanced Features
- **Model Registry**: Version control and stage management
- **Batch Predictions**: Efficient bulk inference
- **Real-time Monitoring**: Live drift detection dashboard
- **Comprehensive Logging**: SQLite/PostgreSQL metadata tracking
- **Docker Deployment**: One-command full-stack deployment
- **CI/CD Ready**: GitHub Actions workflow included

## 📦 Installation

### Quick Start (Local)

```bash
# Clone the repository
git clone https://github.com/yourusername/ML-Pipeline.git
cd ML-Pipeline

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_api.txt
pip install -r requirements_dashboard.txt

# Additional dependencies for advanced features
pip install mlflow shap xgboost
```

### Docker Deployment (Recommended)

```bash
# Build and run all services
docker-compose up -d

# Services will be available at:
# - API: http://localhost:8000
# - Dashboard: http://localhost:8501
# - MLflow: http://localhost:5000
```

## 🚀 Quick Start Guide

### 1. Using the REST API

```bash
# Start the API server
uvicorn api.main:app --reload

# Access API documentation
# http://localhost:8000/docs
```

**Train a Model:**
```python
import requests

# Upload data and train
with open('your_data.csv', 'rb') as f:
    files = {'file': f}
    data = {
        'target_column': 'target',
        'model_type': 'random_forest',
        'auto_features': True
    }
    response = requests.post('http://localhost:8000/train', data=data, files=files)
    print(response.json())
```

**Make Predictions:**
```python
# Predict with trained model
data = {
    'data': [
        {'feature1': 1.0, 'feature2': 2.0},
        {'feature1': 1.5, 'feature2': 2.5}
    ]
}
response = requests.post('http://localhost:8000/predict', json=data)
print(response.json())
```

### 2. Using the Premium Dashboard

```bash
# Start the dashboard
streamlit run streamlit_app/app_premium.py

# Access dashboard
# http://localhost:8501
```

**Features:**
- 📁 Drag-and-drop dataset upload
- 🔧 Interactive feature engineering
- 🤖 One-click model training
- 📊 Real-time drift monitoring
- 📈 Model performance visualization
- 💾 Export results and reports

### 3. Using Python API

```python
from pipeline_manager import MLPipeline
from ingestion import create_sample_dataset

# Create pipeline
pipeline = MLPipeline()

# Generate sample data or load your own
df = create_sample_dataset(n_samples=1000, n_features=10)

# Run complete pipeline
results = pipeline.run_full_pipeline(
    source=df,
    target_column='target',
    auto_features=True,
    check_drift=True,
    train_model=True
)

print(f"Model Accuracy: {results['model_metrics']['accuracy']:.3f}")
```

### 4. MLflow Experiment Tracking

```python
from mlflow_integration import MLflowTracker

# Initialize tracker
tracker = MLflowTracker(experiment_name="My_Experiment")

# Start run
tracker.start_run(run_name="experiment_1")

# Log parameters and metrics
tracker.log_params({'model_type': 'random_forest', 'n_estimators': 100})
tracker.log_metrics({'accuracy': 0.95, 'f1_score': 0.93})

# Log model
tracker.log_model(model, registered_model_name="my_model")

# End run
tracker.end_run()

# View experiments at http://localhost:5000
```

### 5. SHAP Model Explanations

```python
from shap_explainer import SHAPExplainer

# Create explainer
explainer = SHAPExplainer(model, X_train, model_type="tree")

# Get feature importance
importance_df = explainer.get_feature_importance(X_test)
print(importance_df)

# Generate plots
explainer.save_all_plots(X_test, output_dir="shap_plots")

# Explain single prediction
explanation = explainer.explain_prediction(X_test.iloc[0])
print(explanation['top_features'])
```

## 📚 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/train` | POST | Train new model |
| `/predict` | POST | Make predictions |
| `/drift/check` | POST | Check for data drift |
| `/models` | GET | List all models |
| `/models/{name}` | DELETE | Delete model |
| `/pipeline/state` | GET | Get pipeline state |

### Example Requests

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Train Model:**
```bash
curl -X POST "http://localhost:8000/train" \
  -F "file=@data.csv" \
  -F "target_column=target" \
  -F "model_type=random_forest"
```

**Check Drift:**
```bash
curl -X POST "http://localhost:8000/drift/check" \
  -F "file=@new_data.csv" \
  -F "set_as_reference=false"
```

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild after changes
docker-compose up -d --build
```

### Individual Containers

```bash
# Build API
docker build -f Dockerfile.api -t ml-pipeline-api .
docker run -p 8000:8000 ml-pipeline-api

# Build Dashboard
docker build -f Dockerfile.dashboard -t ml-pipeline-dashboard .
docker run -p 8501:8501 ml-pipeline-dashboard
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_pipeline.py -v

# Run API tests
pytest tests/test_api.py -v
```

## 📊 Project Structure

```
ML-Pipeline/
├── api/                        # FastAPI application
│   ├── main.py                # API endpoints
│   └── __init__.py
├── streamlit_app/             # Streamlit dashboard
│   └── app_premium.py         # Premium UI
├── tests/                     # Test suite
│   ├── test_pipeline.py       # Pipeline tests
│   ├── test_api.py           # API tests
│   └── __init__.py
├── .github/workflows/         # CI/CD
│   └── ci-cd.yml             # GitHub Actions
├── pipeline_manager.py        # Core pipeline
├── feature_eng.py            # Feature engineering
├── monitor.py                # Drift detection
├── ingestion.py              # Data ingestion
├── database.py               # Metadata logging
├── mlflow_integration.py     # MLflow tracking
├── shap_explainer.py         # SHAP explanations
├── config.py                 # Configuration
├── Dockerfile.api            # API container
├── Dockerfile.dashboard      # Dashboard container
├── docker-compose.yml        # Full stack deployment
├── requirements.txt          # Core dependencies
├── requirements_api.txt      # API dependencies
└── requirements_dashboard.txt # Dashboard dependencies
```

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
# Model settings
model_type: "random_forest"
validation_split: 0.2

# Drift monitoring
monitor:
  ks_significance_level: 0.05
  drift_threshold: 0.3
  auto_retrain: true

# Feature engineering
features:
  target_column: "target"
  numerical_transformations:
    default: ["impute_median", "standardize"]
```

## 🌐 Deployment Options

### AWS Deployment
```bash
# Use AWS ECS or EKS
# See docs/aws_deployment.md
```

### Azure Deployment
```bash
# Use Azure Container Instances
# See docs/azure_deployment.md
```

### Heroku Deployment
```bash
# Deploy with Heroku CLI
heroku container:push web
heroku container:release web
```

## 📈 Performance & Scalability

- **Batch Processing**: Handle datasets with millions of rows
- **Parallel Processing**: Multi-core feature engineering
- **Efficient Storage**: Optimized model serialization
- **Caching**: Smart caching for repeated operations
- **GPU Support**: CUDA-enabled for deep learning (optional)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FastAPI for the amazing web framework
- Streamlit for the interactive dashboard capabilities
- MLflow for experiment tracking
- SHAP for model interpretability
- Scikit-learn for ML algorithms

## 📞 Support

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/ML-Pipeline/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/ML-Pipeline/discussions)

---

**Built with ❤️ for the ML Community** | v2.0.0 | 2026
