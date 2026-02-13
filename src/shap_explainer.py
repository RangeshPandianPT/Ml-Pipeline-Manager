"""
SHAP Integration for Model Explainability
Feature importance and model interpretation
"""

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Optional, List, Dict, Any, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based model explainability and interpretation
    """
    
    def __init__(self, model, X_train: pd.DataFrame, model_type: str = "tree"):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained model
            X_train: Training data for background
            model_type: Type of explainer ("tree", "linear", "kernel", "deep")
        """
        self.model = model
        self.X_train = X_train
        self.model_type = model_type
        self.feature_names = X_train.columns.tolist()
        
        # Initialize explainer based on model type
        self.explainer = self._create_explainer()
        self.shap_values = None
        self.expected_value = None
    
    def _create_explainer(self):
        """Create appropriate SHAP explainer based on model type"""
        try:
            if self.model_type == "tree":
                # For tree-based models (RandomForest, XGBoost, etc.)
                explainer = shap.TreeExplainer(self.model)
                logger.info("Created TreeExplainer")
            
            elif self.model_type == "linear":
                # For linear models
                explainer = shap.LinearExplainer(self.model, self.X_train)
                logger.info("Created LinearExplainer")
            
            elif self.model_type == "kernel":
                # For any model (slower but universal)
                # Use a sample for background to speed up
                background = shap.sample(self.X_train, min(100, len(self.X_train)))
                explainer = shap.KernelExplainer(self.model.predict, background)
                logger.info("Created KernelExplainer")
            
            else:
                # Default to TreeExplainer
                explainer = shap.TreeExplainer(self.model)
                logger.info("Created default TreeExplainer")
            
            return explainer
        
        except Exception as e:
            logger.warning(f"Failed to create {self.model_type} explainer: {str(e)}")
            logger.info("Falling back to KernelExplainer")
            background = shap.sample(self.X_train, min(100, len(self.X_train)))
            return shap.KernelExplainer(self.model.predict, background)
    
    def calculate_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate SHAP values for given data
        
        Args:
            X: Data to explain
        
        Returns:
            SHAP values array
        """
        try:
            self.shap_values = self.explainer.shap_values(X)
            
            # Handle multi-output case (classification)
            if isinstance(self.shap_values, list):
                # For binary classification, use positive class
                self.shap_values = self.shap_values[1] if len(self.shap_values) == 2 else self.shap_values[0]
            
            # Get expected value
            if hasattr(self.explainer, 'expected_value'):
                self.expected_value = self.explainer.expected_value
                if isinstance(self.expected_value, (list, np.ndarray)):
                    self.expected_value = self.expected_value[1] if len(self.expected_value) == 2 else self.expected_value[0]
            
            logger.info(f"Calculated SHAP values for {len(X)} samples")
            return self.shap_values
        
        except Exception as e:
            logger.error(f"Failed to calculate SHAP values: {str(e)}")
            raise
    
    def get_feature_importance(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Get global feature importance based on mean absolute SHAP values
        
        Args:
            X: Data to explain (uses training data if None)
        
        Returns:
            DataFrame with feature importance
        """
        if X is None:
            X = self.X_train
        
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        # Calculate mean absolute SHAP values
        importance = np.abs(self.shap_values).mean(axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_summary(self, X: Optional[pd.DataFrame] = None, max_display: int = 20, save_path: Optional[str] = None):
        """
        Create SHAP summary plot
        
        Args:
            X: Data to explain
            max_display: Maximum number of features to display
            save_path: Path to save the plot
        """
        if X is None:
            X = self.X_train
        
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X, max_display=max_display, show=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved summary plot to {save_path}")
        
        plt.close()
    
    def plot_bar(self, X: Optional[pd.DataFrame] = None, max_display: int = 20, save_path: Optional[str] = None):
        """
        Create SHAP bar plot (feature importance)
        
        Args:
            X: Data to explain
            max_display: Maximum number of features to display
            save_path: Path to save the plot
        """
        if X is None:
            X = self.X_train
        
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X, plot_type="bar", max_display=max_display, show=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved bar plot to {save_path}")
        
        plt.close()
    
    def plot_waterfall(self, sample_idx: int, X: Optional[pd.DataFrame] = None, save_path: Optional[str] = None):
        """
        Create SHAP waterfall plot for a single prediction
        
        Args:
            sample_idx: Index of the sample to explain
            X: Data to explain
            save_path: Path to save the plot
        """
        if X is None:
            X = self.X_train
        
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        
        # Create explanation object
        explanation = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=self.expected_value,
            data=X.iloc[sample_idx].values,
            feature_names=self.feature_names
        )
        
        shap.waterfall_plot(explanation, show=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved waterfall plot to {save_path}")
        
        plt.close()
    
    def plot_force(self, sample_idx: int, X: Optional[pd.DataFrame] = None, save_path: Optional[str] = None):
        """
        Create SHAP force plot for a single prediction
        
        Args:
            sample_idx: Index of the sample to explain
            X: Data to explain
            save_path: Path to save the plot
        """
        if X is None:
            X = self.X_train
        
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        # Create force plot
        force_plot = shap.force_plot(
            self.expected_value,
            self.shap_values[sample_idx],
            X.iloc[sample_idx],
            matplotlib=True,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved force plot to {save_path}")
        
        plt.close()
    
    def plot_dependence(self, feature: str, X: Optional[pd.DataFrame] = None, 
                       interaction_feature: Optional[str] = "auto", save_path: Optional[str] = None):
        """
        Create SHAP dependence plot
        
        Args:
            feature: Feature to plot
            X: Data to explain
            interaction_feature: Feature to use for coloring
            save_path: Path to save the plot
        """
        if X is None:
            X = self.X_train
        
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, self.shap_values, X, interaction_index=interaction_feature, show=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved dependence plot to {save_path}")
        
        plt.close()
    
    def create_interactive_importance_plot(self, X: Optional[pd.DataFrame] = None, top_n: int = 20) -> go.Figure:
        """
        Create interactive Plotly feature importance plot
        
        Args:
            X: Data to explain
            top_n: Number of top features to display
        
        Returns:
            Plotly figure
        """
        importance_df = self.get_feature_importance(X)
        top_features = importance_df.head(top_n)
        
        fig = go.Figure(go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker=dict(
                color=top_features['importance'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=[f"{val:.4f}" for val in top_features['importance']],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f"Top {top_n} Feature Importances (SHAP)",
            xaxis_title="Mean |SHAP Value|",
            yaxis_title="Features",
            height=max(400, top_n * 25),
            template="plotly_white",
            font=dict(family="Inter, sans-serif", size=12)
        )
        
        return fig
    
    def explain_prediction(self, sample: pd.Series, top_n: int = 10) -> Dict[str, Any]:
        """
        Get detailed explanation for a single prediction
        
        Args:
            sample: Single sample to explain (as Series or 1-row DataFrame)
            top_n: Number of top contributing features
        
        Returns:
            Dictionary with explanation details
        """
        # Convert to DataFrame if Series
        if isinstance(sample, pd.Series):
            X = pd.DataFrame([sample])
        else:
            X = sample
        
        # Calculate SHAP values
        shap_vals = self.explainer.shap_values(X)
        
        # Handle multi-output
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]
        
        # Get SHAP values for this sample
        sample_shap = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
        
        # Create feature contributions
        contributions = pd.DataFrame({
            'feature': self.feature_names,
            'value': X.iloc[0].values,
            'shap_value': sample_shap,
            'abs_shap': np.abs(sample_shap)
        }).sort_values('abs_shap', ascending=False)
        
        top_contributions = contributions.head(top_n)
        
        return {
            'base_value': self.expected_value,
            'prediction': self.model.predict(X)[0],
            'top_features': top_contributions.to_dict('records'),
            'all_contributions': contributions.to_dict('records')
        }
    
    def detect_feature_drift_shap(self, X_reference: pd.DataFrame, X_current: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect feature drift using SHAP values
        
        Args:
            X_reference: Reference dataset
            X_current: Current dataset
        
        Returns:
            Dictionary with drift analysis
        """
        # Calculate SHAP values for both datasets
        shap_ref = self.calculate_shap_values(X_reference)
        shap_curr = self.explainer.shap_values(X_current)
        
        # Handle multi-output
        if isinstance(shap_curr, list):
            shap_curr = shap_curr[1] if len(shap_curr) == 2 else shap_curr[0]
        
        # Calculate mean absolute SHAP values
        importance_ref = np.abs(shap_ref).mean(axis=0)
        importance_curr = np.abs(shap_curr).mean(axis=0)
        
        # Calculate drift score (relative change)
        drift_scores = np.abs(importance_curr - importance_ref) / (importance_ref + 1e-10)
        
        # Create drift report
        drift_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_reference': importance_ref,
            'importance_current': importance_curr,
            'drift_score': drift_scores
        }).sort_values('drift_score', ascending=False)
        
        # Identify drifted features (threshold: 50% change)
        drifted_features = drift_df[drift_df['drift_score'] > 0.5]
        
        return {
            'drift_detected': len(drifted_features) > 0,
            'num_drifted_features': len(drifted_features),
            'drifted_features': drifted_features['feature'].tolist(),
            'drift_details': drift_df.to_dict('records'),
            'overall_drift_score': drift_scores.mean()
        }
    
    def save_all_plots(self, X: Optional[pd.DataFrame] = None, output_dir: str = "shap_plots"):
        """
        Generate and save all SHAP plots
        
        Args:
            X: Data to explain
            output_dir: Directory to save plots
        """
        if X is None:
            X = self.X_train
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"Generating SHAP plots in {output_dir}")
        
        # Summary plot
        self.plot_summary(X, save_path=str(output_path / "summary_plot.png"))
        
        # Bar plot
        self.plot_bar(X, save_path=str(output_path / "bar_plot.png"))
        
        # Waterfall plot for first sample
        self.plot_waterfall(0, X, save_path=str(output_path / "waterfall_plot.png"))
        
        # Dependence plots for top 3 features
        importance_df = self.get_feature_importance(X)
        top_features = importance_df.head(3)['feature'].tolist()
        
        for feature in top_features:
            safe_name = feature.replace('/', '_').replace(' ', '_')
            self.plot_dependence(feature, X, save_path=str(output_path / f"dependence_{safe_name}.png"))
        
        logger.info(f"Saved all SHAP plots to {output_dir}")
