"""
MLflow Integration for ML Pipeline
Experiment tracking, model registry, and versioning
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    MLflow integration for experiment tracking and model management
    """
    
    def __init__(self, 
                 tracking_uri: str = "mlruns",
                 experiment_name: str = "ML_Pipeline_Experiments"):
        """
        Initialize MLflow tracker
        
        Args:
            tracking_uri: URI for MLflow tracking server
            experiment_name: Name of the experiment
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment: {experiment_name}")
        except:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name}")
        
        mlflow.set_experiment(experiment_name)
        
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.active_run = None
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Start a new MLflow run
        
        Args:
            run_name: Name for the run
            tags: Additional tags for the run
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.active_run = mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"Started MLflow run: {run_name} (ID: {self.active_run.info.run_id})")
        
        return self.active_run
    
    def end_run(self):
        """End the active MLflow run"""
        if self.active_run:
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {self.active_run.info.run_id}")
            self.active_run = None
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow
        
        Args:
            params: Dictionary of parameters
        """
        for key, value in params.items():
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(f"Failed to log param {key}: {str(e)}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        for key, value in metrics.items():
            try:
                mlflow.log_metric(key, value, step=step)
            except Exception as e:
                logger.warning(f"Failed to log metric {key}: {str(e)}")
    
    def log_model(self, 
                  model,
                  artifact_path: str = "model",
                  signature=None,
                  input_example=None,
                  registered_model_name: Optional[str] = None):
        """
        Log model to MLflow
        
        Args:
            model: Trained model
            artifact_path: Path within the run's artifact directory
            signature: Model signature
            input_example: Example input
            registered_model_name: Name for model registry
        """
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
            logger.info(f"Logged model to MLflow: {artifact_path}")
        except Exception as e:
            logger.error(f"Failed to log model: {str(e)}")
            raise
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log artifact to MLflow
        
        Args:
            local_path: Local file path
            artifact_path: Path within the run's artifact directory
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.warning(f"Failed to log artifact {local_path}: {str(e)}")
    
    def log_dict(self, dictionary: Dict, filename: str):
        """
        Log dictionary as JSON artifact
        
        Args:
            dictionary: Dictionary to log
            filename: Name of the JSON file
        """
        try:
            mlflow.log_dict(dictionary, filename)
            logger.info(f"Logged dictionary: {filename}")
        except Exception as e:
            logger.warning(f"Failed to log dictionary {filename}: {str(e)}")
    
    def log_figure(self, figure, filename: str):
        """
        Log matplotlib/plotly figure
        
        Args:
            figure: Figure object
            filename: Name of the figure file
        """
        try:
            mlflow.log_figure(figure, filename)
            logger.info(f"Logged figure: {filename}")
        except Exception as e:
            logger.warning(f"Failed to log figure {filename}: {str(e)}")
    
    def log_dataframe(self, df: pd.DataFrame, filename: str):
        """
        Log pandas DataFrame as artifact
        
        Args:
            df: DataFrame to log
            filename: Name of the CSV file
        """
        try:
            temp_path = Path(f"temp_{filename}")
            df.to_csv(temp_path, index=False)
            self.log_artifact(str(temp_path))
            temp_path.unlink()
            logger.info(f"Logged dataframe: {filename}")
        except Exception as e:
            logger.warning(f"Failed to log dataframe {filename}: {str(e)}")
    
    def set_tags(self, tags: Dict[str, str]):
        """
        Set tags for the active run
        
        Args:
            tags: Dictionary of tags
        """
        for key, value in tags.items():
            try:
                mlflow.set_tag(key, value)
            except Exception as e:
                logger.warning(f"Failed to set tag {key}: {str(e)}")
    
    def get_run_info(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a run
        
        Args:
            run_id: Run ID (uses active run if None)
        
        Returns:
            Dictionary with run information
        """
        if run_id is None and self.active_run:
            run_id = self.active_run.info.run_id
        
        if run_id is None:
            raise ValueError("No active run and no run_id provided")
        
        run = self.client.get_run(run_id)
        
        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "params": run.data.params,
            "metrics": run.data.metrics,
            "tags": run.data.tags
        }
    
    def search_runs(self, 
                    filter_string: str = "",
                    max_results: int = 100,
                    order_by: List[str] = None) -> pd.DataFrame:
        """
        Search for runs in the experiment
        
        Args:
            filter_string: Filter string (e.g., "metrics.accuracy > 0.9")
            max_results: Maximum number of results
            order_by: List of columns to order by
        
        Returns:
            DataFrame with run information
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by
        )
        
        return runs
    
    def get_best_run(self, metric: str, ascending: bool = False) -> Dict[str, Any]:
        """
        Get the best run based on a metric
        
        Args:
            metric: Metric name
            ascending: If True, lower is better
        
        Returns:
            Dictionary with best run information
        """
        runs = self.search_runs(
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )
        
        if len(runs) == 0:
            raise ValueError("No runs found")
        
        best_run = runs.iloc[0]
        
        return {
            "run_id": best_run.run_id,
            "metrics": {k.replace('metrics.', ''): v for k, v in best_run.items() if k.startswith('metrics.')},
            "params": {k.replace('params.', ''): v for k, v in best_run.items() if k.startswith('params.')}
        }
    
    def load_model(self, run_id: str, artifact_path: str = "model"):
        """
        Load a model from MLflow
        
        Args:
            run_id: Run ID
            artifact_path: Path to the model artifact
        
        Returns:
            Loaded model
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded model from run {run_id}")
        
        return model
    
    def register_model(self, 
                       run_id: str,
                       model_name: str,
                       artifact_path: str = "model") -> str:
        """
        Register a model in the model registry
        
        Args:
            run_id: Run ID
            model_name: Name for the registered model
            artifact_path: Path to the model artifact
        
        Returns:
            Model version
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        
        result = mlflow.register_model(model_uri, model_name)
        logger.info(f"Registered model {model_name} version {result.version}")
        
        return result.version
    
    def transition_model_stage(self, 
                               model_name: str,
                               version: str,
                               stage: str):
        """
        Transition a model to a different stage
        
        Args:
            model_name: Name of the registered model
            version: Model version
            stage: Target stage (Staging, Production, Archived)
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        logger.info(f"Transitioned {model_name} v{version} to {stage}")
    
    def get_latest_model_version(self, model_name: str, stage: Optional[str] = None):
        """
        Get the latest version of a registered model
        
        Args:
            model_name: Name of the registered model
            stage: Optional stage filter
        
        Returns:
            Latest model version
        """
        if stage:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
        else:
            versions = self.client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            raise ValueError(f"No versions found for model {model_name}")
        
        latest = max(versions, key=lambda x: int(x.version))
        
        return latest
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple runs
        
        Args:
            run_ids: List of run IDs to compare
        
        Returns:
            DataFrame with comparison
        """
        runs_data = []
        
        for run_id in run_ids:
            run_info = self.get_run_info(run_id)
            runs_data.append({
                "run_id": run_id,
                **run_info["params"],
                **run_info["metrics"]
            })
        
        return pd.DataFrame(runs_data)
    
    def delete_run(self, run_id: str):
        """
        Delete a run
        
        Args:
            run_id: Run ID to delete
        """
        self.client.delete_run(run_id)
        logger.info(f"Deleted run: {run_id}")
    
    def log_pipeline_run(self,
                        params: Dict[str, Any],
                        metrics: Dict[str, float],
                        model,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        artifacts: Optional[Dict[str, str]] = None,
                        tags: Optional[Dict[str, str]] = None,
                        registered_model_name: Optional[str] = None):
        """
        Comprehensive logging for a complete pipeline run
        
        Args:
            params: Pipeline parameters
            metrics: Model metrics
            model: Trained model
            X_train: Training features
            y_train: Training target
            artifacts: Dictionary of artifact paths to log
            tags: Additional tags
            registered_model_name: Name for model registry
        """
        # Start run
        run_name = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_run(run_name=run_name, tags=tags)
        
        try:
            # Log parameters
            self.log_params(params)
            
            # Log metrics
            self.log_metrics(metrics)
            
            # Create model signature
            signature = infer_signature(X_train, y_train)
            
            # Create input example
            input_example = X_train.head(5)
            
            # Log model
            self.log_model(
                model=model,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
            
            # Log artifacts
            if artifacts:
                for artifact_path in artifacts.values():
                    if Path(artifact_path).exists():
                        self.log_artifact(artifact_path)
            
            # Log additional metadata
            metadata = {
                "training_samples": len(X_train),
                "features": list(X_train.columns),
                "timestamp": datetime.now().isoformat()
            }
            self.log_dict(metadata, "metadata.json")
            
            logger.info(f"Successfully logged complete pipeline run: {run_name}")
            
            return self.active_run.info.run_id
        
        finally:
            # End run
            self.end_run()


# Convenience function for quick tracking
def track_experiment(experiment_name: str = "ML_Pipeline_Experiments"):
    """
    Decorator for tracking function execution with MLflow
    
    Args:
        experiment_name: Name of the experiment
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracker = MLflowTracker(experiment_name=experiment_name)
            tracker.start_run(run_name=func.__name__)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                tracker.end_run()
        
        return wrapper
    return decorator
