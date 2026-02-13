# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Option 1: Docker (Easiest)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ML-Pipeline.git
cd ML-Pipeline

# 2. Start all services
docker-compose up -d

# 3. Access the applications
# - API Docs: http://localhost:8000/docs
# - Dashboard: http://localhost:8501
# - MLflow: http://localhost:5000
```

### Option 2: Local Installation

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements_api.txt
pip install -r requirements_dashboard.txt
pip install mlflow shap xgboost

# 3. Run the demo
python main.py
```

## 📊 Your First Model

### Using the Dashboard (Recommended for Beginners)

1. **Start the dashboard:**
   ```bash
   streamlit run streamlit_app/app_premium.py
   ```

2. **Upload your data:**
   - Go to http://localhost:8501
   - Click "Data Upload" tab
   - Drag and drop your CSV file

3. **Train a model:**
   - Select target column
   - Click "Train Model"
   - View results in real-time

### Using Python Code

```python
from pipeline_manager import MLPipeline

# Initialize pipeline
pipeline = MLPipeline()

# Load your data (or use sample data)
from ingestion import create_sample_dataset
df = create_sample_dataset(n_samples=1000, n_features=10)

# Run complete pipeline
results = pipeline.run_full_pipeline(
    source=df,
    target_column='target',
    auto_features=True,
    train_model=True
)

# View results
print(f"Accuracy: {results['model_metrics']['accuracy']:.3f}")
```

### Using the REST API

```python
import requests

# Train a model
with open('your_data.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/train',
        files={'file': f},
        data={
            'target_column': 'target',
            'model_type': 'random_forest'
        }
    )

print(response.json())
```

## 🎯 Common Use Cases

### 1. Classification Problem

```python
pipeline = MLPipeline()
pipeline.config.model_type = 'random_forest'

results = pipeline.run_full_pipeline(
    source='customer_churn.csv',
    target_column='churned',
    auto_features=True,
    train_model=True
)
```

### 2. Regression Problem

```python
pipeline = MLPipeline()
pipeline.config.model_type = 'gradient_boosting'

results = pipeline.run_full_pipeline(
    source='house_prices.csv',
    target_column='price',
    auto_features=True,
    train_model=True
)
```

### 3. Drift Monitoring

```python
# Set reference data
pipeline.monitor_drift(reference_data, set_as_reference=True)

# Check new data for drift
drift_report = pipeline.monitor_drift(new_data)

if drift_report.overall_drift_detected:
    print("Drift detected! Retraining recommended.")
```

## 📖 Next Steps

- Read the [Full Documentation](README.md)
- Explore [API Endpoints](README.md#-api-endpoints)
- Learn about [SHAP Explanations](README.md#5-shap-model-explanations)
- Set up [MLflow Tracking](README.md#4-mlflow-experiment-tracking)
- Deploy with [Docker](README.md#-docker-deployment)

## 🆘 Troubleshooting

**Issue: Port already in use**
```bash
# Change ports in docker-compose.yml
# Or stop conflicting services
```

**Issue: Module not found**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Issue: Out of memory**
```bash
# Reduce batch size in config.yaml
batch_size: 500  # Default is 1000
```

## 💡 Tips

- Start with the dashboard for visual feedback
- Use sample data to test features
- Check MLflow UI for experiment comparison
- Enable drift monitoring for production
- Use SHAP for model debugging

---

**Ready to build amazing ML pipelines!** 🚀
