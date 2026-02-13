"""
Main Entry Point for ML Pipeline.
Demonstrates the complete MLOps workflow with sample/custom data.
Domain-agnostic - works with any dataset.
"""

import logging
import argparse
from pathlib import Path

from src.config import PipelineConfig, default_config
from src.ingestion import create_sample_agriharv_data, create_sample_healthcare_data
from src.pipeline_manager import MLPipeline
from src.monitor import simulate_drift

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_demo_pipeline(dataset: str = "generic", n_samples: int = 1000):
    """
    Run a demonstration of the complete ML pipeline.
    
    Args:
        dataset: "generic", "healthcare", or path to CSV file
        n_samples: Number of samples to generate (if using generic)
    """
    print("\n" + "=" * 60)
    print("ML Pipeline - Automated Feature Engineering & Monitoring")
    print("=" * 60 + "\n")
    
    # Load or create data
    if dataset == "generic":
        print(f"Generating generic sample dataset ({n_samples} samples)...")
        from src.ingestion import create_sample_dataset
        df = create_sample_dataset(n_samples=n_samples)
        target_column = "target"
    elif dataset == "healthcare":
        print(f"Generating healthcare sample dataset ({n_samples} samples)...")
        from src.ingestion import create_sample_healthcare_data
        df = create_sample_healthcare_data(n_samples=n_samples)
        target_column = "cvd_risk_score"
    else:
        # Assume it's a file path
        print(f"Loading data from: {dataset}")
        df = pd.read_csv(dataset)
        target_column = "target"
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
    print()
    
    # Initialize pipeline
    config = default_config
    config.features.target_column = target_column
    
    pipeline = MLPipeline(config)
    
    # Run the full pipeline
    print("\n" + "-" * 40)
    print("Running Full Pipeline...")
    print("-" * 40)
    
    results = pipeline.run_full_pipeline(
        source=df,
        target_column=target_column,
        auto_features=True,
        check_drift=True,
        train_model=True
    )
    
    # Print results
    print("\n" + "-" * 40)
    print("Pipeline Results")
    print("-" * 40)
    print(f"Status: {results['status']}")
    print(f"Steps completed: {results['steps_completed']}")
    print(f"Metrics: {results['metrics']}")
    if results.get('model_metrics'):
        print(f"Model metrics: {results['model_metrics']}")
    
    return pipeline, results


def demo_drift_detection(pipeline: MLPipeline, dataset: str = "generic"):
    """
    Demonstrate drift detection by simulating data drift.
    """
    print("\n" + "=" * 60)
    print("Drift Detection Demo")
    print("=" * 60 + "\n")
    
    # Create new data with drift
    if dataset == "generic":
        from src.ingestion import create_sample_dataset
        new_data = create_sample_dataset(n_samples=500)
        drift_columns = [c for c in new_data.columns if c.startswith('feature')][:3]
    else:
        from src.ingestion import create_sample_healthcare_data
        new_data = create_sample_healthcare_data(n_samples=500)
        drift_columns = ['age', 'bmi', 'blood_pressure_systolic']
    
    # Simulate drift in selected columns
    drifted_data = simulate_drift(new_data, drift_columns, drift_magnitude=1.0)
    
    print(f"Simulated drift in columns: {drift_columns}")
    print(f"Drifted data shape: {drifted_data.shape}")
    
    # Run drift detection
    pipeline.ingest(drifted_data)
    drift_report = pipeline.monitor_drift()
    
    print("\n" + drift_report.summary())
    
    # Check if retraining is needed
    if pipeline.drift_monitor.should_retrain():
        print("\n[ACTION] Triggering automatic retraining...")
        pipeline.engineer_features()
        pipeline.prepare_training_data()
        pipeline.train_model()
        metrics = pipeline.evaluate_model()
        print(f"New model metrics: {metrics}")
        pipeline.save_model()
    
    return drift_report


def demo_feature_engineering():
    """
    Demonstrate feature engineering capabilities.
    """
    from src.feature_eng import FeatureEngineer
    
    print("\n" + "=" * 60)
    print("Feature Engineering Demo")
    print("=" * 60 + "\n")
    
    # List available transformations
    engineer = FeatureEngineer()
    print("Available transformations:")
    for t in engineer.list_available_transformations():
        print(f"  - {t}")
    
    # Create sample data
    from src.ingestion import create_sample_dataset
    df = create_sample_dataset(n_samples=100, n_features=10)
    print(f"\nOriginal shape: {df.shape}")
    
    # Manual transformation pipeline
    engineer.reset_pipeline()
    engineer.add_transformation("impute_median")
    engineer.add_transformation("log_transform", columns=['nitrogen_level', 'phosphorus_level'])
    engineer.add_transformation("standardize")
    engineer.add_transformation("one_hot_encode")
    
    print("\nTransformation pipeline:")
    for step in engineer.get_pipeline_summary():
        print(f"  {step['step']}. {step['name']} -> {step['columns'] or 'auto'}")
    
    # Apply transformations
    result = engineer.transform(df, target_column='target')
    print(f"\nTransformed shape: {result.shape}")
    print(f"New columns created: {len(result.columns) - len(df.columns)}")
    
    return result


def demo_monitoring_only(data_path: str = None):
    """
    Run monitoring on new data without retraining.
    """
    from src.monitor import DriftMonitor
    from src.ingestion import DataIngestion, create_sample_dataset
    
    print("\n" + "=" * 60)
    print("Monitoring Only Mode")
    print("=" * 60 + "\n")
    
    # Initialize components
    monitor = DriftMonitor()
    ingestion = DataIngestion()
    
    # Load reference from database
    if monitor.load_reference_from_db():
        print("Loaded reference data from database")
    else:
        print("No reference data found. Using generated data as reference.")
        ref_data = create_sample_dataset(500)
        ingestion.ingest_dataframe(ref_data, "reference")
        monitor.set_reference(ref_data, ingestion.ingestion_id)
    
    # Load or generate current data
    if data_path:
        current_data = ingestion.ingest_csv(data_path)
    else:
        current_data = create_sample_dataset(500)
        ingestion.ingest_dataframe(current_data, "current")
    
    # Run comprehensive drift check
    report = monitor.comprehensive_drift_check(current_data, ingestion.ingestion_id)
    
    print(f"\nKS Test Results:")
    print(f"  Overall drift: {report['ks_report']['overall_drift_detected']}")
    print(f"  Columns with drift: {report['ks_report']['columns_with_drift']}")
    
    print(f"\nPSI Scores:")
    for col, score in list(report['psi_scores'].items())[:5]:
        print(f"  {col}: {score:.4f}")
    
    print(f"\nRecommendation: {report['overall_recommendation']}")
    
    return report


def main():
    """Main entry point with CLI support."""
    parser = argparse.ArgumentParser(
        description='ML Pipeline - Automated Feature Engineering & Monitoring'
    )
    
    parser.add_argument(
        '--mode',
        choices=['demo', 'features', 'monitor', 'drift', 'full'],
        default='demo',
        help='Pipeline mode to run (default: demo)'
    )
    
    parser.add_argument(
        '--dataset',
        choices=['generic', 'healthcare'],
        default='generic',
        help='Sample dataset to use (default: generic). Use CSV file path for custom data.'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=1000,
        help='Number of samples for synthetic data (default: 1000)'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to custom input data file (CSV)'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default='target',
        help='Target column name (default: target)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        # Use data path if provided, otherwise use sample dataset
        if args.data_path:
            pipeline, results = run_demo_pipeline(args.data_path)
        else:
            pipeline, results = run_demo_pipeline(args.dataset, args.samples)
        print("\n--- Demo completed successfully ---\n")
        
    elif args.mode == 'features':
        demo_feature_engineering()
        print("\n--- Feature engineering demo completed ---\n")
        
    elif args.mode == 'monitor':
        demo_monitoring_only(args.data_path)
        print("\n--- Monitoring completed ---\n")
        
    elif args.mode == 'drift':
        if args.data_path:
            pipeline, _ = run_demo_pipeline(args.data_path)
        else:
            pipeline, _ = run_demo_pipeline(args.dataset, args.samples)
        demo_drift_detection(pipeline, args.dataset)
        print("\n--- Drift detection demo completed ---\n")
        
    elif args.mode == 'full':
        if not args.data_path:
            print("Error: --data-path required for full mode")
            return
        
        pipeline = MLPipeline()
        results = pipeline.run_full_pipeline(
            source=args.data_path,
            target_column=args.target,
            auto_features=True,
            check_drift=True,
            train_model=True
        )
        print(f"\nPipeline completed: {results['status']}")


if __name__ == "__main__":
    main()
