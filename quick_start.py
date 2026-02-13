"""
Quick Start Script - Run this to test the pipeline immediately!
No configuration needed - works out of the box.
"""

from pipeline_manager import MLPipeline
from ingestion import create_sample_dataset

def quick_start():
    """
    Quickest way to see the pipeline in action.
    Generates sample data and runs the full pipeline.
    """
    print("\n" + "="*60)
    print("ML Pipeline - Quick Start Demo")
    print("="*60 + "\n")
    
    # Generate sample data (no CSV file needed!)
    print("Generating sample dataset...")
    df = create_sample_dataset(n_samples=1000, n_features=10)
    print(f"✓ Created dataset: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    # Initialize pipeline
    print("Initializing ML Pipeline...")
    pipeline = MLPipeline()
    print("✓ Pipeline initialized\n")
    
    # Run the full pipeline
    print("Running complete pipeline:")
    print("  → Data ingestion")
    print("  → Automatic feature engineering")
    print("  → Drift detection")
    print("  → Model training")
    print("  → Evaluation\n")
    
    results = pipeline.run_full_pipeline(
        source=df,
        target_column="target",
        auto_features=True,
        check_drift=True,
        train_model=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"✓ Status: {results['status'].upper()}")
    print(f"✓ Steps completed: {len(results['steps_completed'])}")
    
    if results.get('metrics'):
        print(f"\nDataset Metrics:")
        print(f"  - Rows ingested: {results['metrics']['ingestion_rows']}")
        print(f"  - Original columns: {results['metrics']['ingestion_columns']}")
        print(f"  - Features created: {results['metrics']['features_created']}")
        print(f"  - Drift detected: {results['metrics']['drift_detected']}")
    
    if results.get('model_metrics'):
        print(f"\nModel Performance:")
        for metric, value in results['model_metrics'].items():
            print(f"  - {metric.upper()}: {value:.4f}")
    
    print(f"\n✓ Execution time: {results['metrics']['execution_time_seconds']:.2f} seconds")
    print("\n" + "="*60)
    print("✓ Pipeline completed successfully!")
    print("="*60 + "\n")
    
    print("Next steps:")
    print("  1. Check 'logs/' directory for detailed logs")
    print("  2. Check 'models/' directory for saved model")
    print("  3. Run with your own data: python main.py --data-path your_data.csv")
    print("  4. See README.md for more examples\n")
    
    return pipeline, results


if __name__ == "__main__":
    quick_start()
