"""
Data Drift Monitoring Module for ML Pipeline.
Implements Kolmogorov-Smirnov (K-S) test and other statistical tests for drift detection.
Domain-agnostic implementation suitable for any dataset.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from scipy import stats
from enum import Enum

from config import PipelineConfig, MonitorConfig, default_config
from database import MetadataDatabase, DriftLog, generate_id

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of data drift."""
    NO_DRIFT = "no_drift"
    COVARIATE_DRIFT = "covariate_drift"  # Change in feature distributions
    CONCEPT_DRIFT = "concept_drift"  # Change in relationship between features and target
    LABEL_DRIFT = "label_drift"  # Change in target distribution


@dataclass
class DriftTestResult:
    """Result of a single drift test on a column."""
    column_name: str
    test_name: str
    statistic: float
    p_value: float
    drift_detected: bool
    threshold: float
    reference_stats: Dict[str, float] = field(default_factory=dict)
    current_stats: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'column_name': self.column_name,
            'test_name': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'drift_detected': self.drift_detected,
            'threshold': self.threshold,
            'reference_stats': self.reference_stats,
            'current_stats': self.current_stats
        }


@dataclass
class DriftReport:
    """Comprehensive drift detection report."""
    drift_id: str
    timestamp: str
    overall_drift_detected: bool
    drift_percentage: float
    total_columns_tested: int
    columns_with_drift: List[str]
    test_results: Dict[str, DriftTestResult]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'drift_id': self.drift_id,
            'timestamp': self.timestamp,
            'overall_drift_detected': self.overall_drift_detected,
            'drift_percentage': self.drift_percentage,
            'total_columns_tested': self.total_columns_tested,
            'columns_with_drift': self.columns_with_drift,
            'test_results': {k: v.to_dict() for k, v in self.test_results.items()},
            'recommendations': self.recommendations
        }
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"=== Drift Detection Report ===",
            f"Drift ID: {self.drift_id}",
            f"Timestamp: {self.timestamp}",
            f"Overall Drift Detected: {'YES' if self.overall_drift_detected else 'NO'}",
            f"Columns Tested: {self.total_columns_tested}",
            f"Columns with Drift: {len(self.columns_with_drift)} ({self.drift_percentage:.1%})",
        ]
        
        if self.columns_with_drift:
            lines.append(f"\nDrift detected in:")
            for col in self.columns_with_drift:
                result = self.test_results[col]
                lines.append(f"  - {col}: p-value={result.p_value:.4f}, statistic={result.statistic:.4f}")
        
        if self.recommendations:
            lines.append(f"\nRecommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")
        
        return "\n".join(lines)


class StatisticalTests:
    """Collection of statistical tests for drift detection."""
    
    @staticmethod
    def kolmogorov_smirnov_test(reference: np.ndarray, 
                                 current: np.ndarray) -> Tuple[float, float]:
        """
        Perform two-sample Kolmogorov-Smirnov test.
        
        The K-S test compares the cumulative distributions of two samples.
        A low p-value indicates the samples come from different distributions.
        
        Args:
            reference: Reference (baseline) data
            current: Current (new) data
        
        Returns:
            Tuple of (KS statistic, p-value)
        """
        # Remove NaN values
        ref_clean = reference[~np.isnan(reference)]
        cur_clean = current[~np.isnan(current)]
        
        if len(ref_clean) < 2 or len(cur_clean) < 2:
            return 0.0, 1.0
        
        statistic, p_value = stats.ks_2samp(ref_clean, cur_clean)
        return float(statistic), float(p_value)
    
    @staticmethod
    def chi_squared_test(reference: np.ndarray, 
                          current: np.ndarray) -> Tuple[float, float]:
        """
        Perform Chi-squared test for categorical data.
        
        Args:
            reference: Reference categorical data
            current: Current categorical data
        
        Returns:
            Tuple of (chi-squared statistic, p-value)
        """
        # Get combined categories
        all_categories = np.unique(np.concatenate([reference, current]))
        
        # Count frequencies
        ref_counts = pd.Series(reference).value_counts()
        cur_counts = pd.Series(current).value_counts()
        
        # Align to same categories
        ref_freq = np.array([ref_counts.get(cat, 0) for cat in all_categories])
        cur_freq = np.array([cur_counts.get(cat, 0) for cat in all_categories])
        
        # Normalize to proportions
        ref_prop = ref_freq / ref_freq.sum() if ref_freq.sum() > 0 else ref_freq
        cur_prop = cur_freq / cur_freq.sum() if cur_freq.sum() > 0 else cur_freq
        
        # Expected frequencies based on reference
        expected = ref_prop * cur_freq.sum()
        expected = np.where(expected == 0, 1e-10, expected)  # Avoid division by zero
        
        # Chi-squared statistic
        chi2_stat = np.sum((cur_freq - expected) ** 2 / expected)
        
        # Degrees of freedom
        dof = max(len(all_categories) - 1, 1)
        
        # P-value
        p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
        
        return float(chi2_stat), float(p_value)
    
    @staticmethod
    def population_stability_index(reference: np.ndarray, 
                                    current: np.ndarray, 
                                    n_bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change
        
        Args:
            reference: Reference data
            current: Current data
            n_bins: Number of bins for discretization
        
        Returns:
            PSI value
        """
        # Remove NaN values
        ref_clean = reference[~np.isnan(reference)]
        cur_clean = current[~np.isnan(current)]
        
        if len(ref_clean) < 2 or len(cur_clean) < 2:
            return 0.0
        
        # Create bins from reference data
        try:
            _, bin_edges = np.histogram(ref_clean, bins=n_bins)
        except ValueError:
            return 0.0
        
        # Calculate proportions in each bin
        ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
        cur_counts, _ = np.histogram(cur_clean, bins=bin_edges)
        
        ref_prop = ref_counts / len(ref_clean)
        cur_prop = cur_counts / len(cur_clean)
        
        # Avoid log(0) by replacing zeros with small value
        ref_prop = np.where(ref_prop == 0, 1e-10, ref_prop)
        cur_prop = np.where(cur_prop == 0, 1e-10, cur_prop)
        
        # Calculate PSI
        psi = np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop))
        
        return float(psi)
    
    @staticmethod
    def jensen_shannon_divergence(reference: np.ndarray, 
                                   current: np.ndarray,
                                   n_bins: int = 10) -> float:
        """
        Calculate Jensen-Shannon Divergence.
        
        JSD is a symmetric measure of difference between two distributions.
        Values range from 0 (identical) to 1 (completely different).
        
        Args:
            reference: Reference data
            current: Current data
            n_bins: Number of bins
        
        Returns:
            JSD value
        """
        ref_clean = reference[~np.isnan(reference)]
        cur_clean = current[~np.isnan(current)]
        
        if len(ref_clean) < 2 or len(cur_clean) < 2:
            return 0.0
        
        # Create common bins
        combined = np.concatenate([ref_clean, cur_clean])
        _, bin_edges = np.histogram(combined, bins=n_bins)
        
        # Get histograms
        ref_hist, _ = np.histogram(ref_clean, bins=bin_edges, density=True)
        cur_hist, _ = np.histogram(cur_clean, bins=bin_edges, density=True)
        
        # Normalize
        ref_hist = ref_hist / ref_hist.sum() if ref_hist.sum() > 0 else ref_hist
        cur_hist = cur_hist / cur_hist.sum() if cur_hist.sum() > 0 else cur_hist
        
        # Replace zeros
        ref_hist = np.where(ref_hist == 0, 1e-10, ref_hist)
        cur_hist = np.where(cur_hist == 0, 1e-10, cur_hist)
        
        # Calculate JSD
        m = 0.5 * (ref_hist + cur_hist)
        
        kl_pm = np.sum(ref_hist * np.log(ref_hist / m))
        kl_qm = np.sum(cur_hist * np.log(cur_hist / m))
        
        jsd = 0.5 * (kl_pm + kl_qm)
        
        return float(np.sqrt(jsd))  # Return square root (JS distance)
    
    @staticmethod
    def wasserstein_distance(reference: np.ndarray, 
                              current: np.ndarray) -> float:
        """
        Calculate Wasserstein (Earth Mover's) distance.
        
        Represents the minimum "cost" to transform one distribution to another.
        
        Args:
            reference: Reference data
            current: Current data
        
        Returns:
            Wasserstein distance
        """
        ref_clean = reference[~np.isnan(reference)]
        cur_clean = current[~np.isnan(current)]
        
        if len(ref_clean) < 2 or len(cur_clean) < 2:
            return 0.0
        
        return float(stats.wasserstein_distance(ref_clean, cur_clean))


class DriftMonitor:
    """
    Data Drift Monitor using statistical tests.
    
    Primary method: Kolmogorov-Smirnov test for numerical features.
    Also supports PSI, JS divergence, and Chi-squared for categorical features.
    """
    
    def __init__(self, config: PipelineConfig = None, db: MetadataDatabase = None):
        self.config = config or default_config
        self.monitor_config = self.config.monitor
        self.db = db or MetadataDatabase(self.config.database)
        self._reference_data: Optional[pd.DataFrame] = None
        self._reference_ingestion_id: Optional[str] = None
        self._last_report: Optional[DriftReport] = None
    
    def set_reference(self, 
                      df: pd.DataFrame, 
                      ingestion_id: str = None) -> None:
        """
        Set the reference (baseline) data for drift comparison.
        
        Args:
            df: Reference DataFrame
            ingestion_id: ID of the ingestion for this data
        """
        self._reference_data = df.copy()
        self._reference_ingestion_id = ingestion_id
        logger.info(f"Reference data set with {len(df)} samples, {len(df.columns)} columns")
    
    def load_reference_from_db(self, ingestion_id: str = None) -> bool:
        """
        Load reference statistics from database.
        
        Args:
            ingestion_id: Specific ingestion ID or None for latest
        
        Returns:
            True if reference loaded successfully
        """
        if ingestion_id is None:
            ingestion_id = self.db.get_latest_ingestion_id()
        
        if ingestion_id is None:
            logger.warning("No reference data found in database")
            return False
        
        stats = self.db.get_reference_stats(ingestion_id)
        if not stats:
            logger.warning(f"No statistics found for ingestion: {ingestion_id}")
            return False
        
        # Reconstruct reference data from stored samples
        ref_data = {}
        for col, col_stats in stats.items():
            ref_data[col] = col_stats['sample_data']
        
        max_len = max(len(v) for v in ref_data.values())
        for col in ref_data:
            if len(ref_data[col]) < max_len:
                ref_data[col].extend([np.nan] * (max_len - len(ref_data[col])))
        
        self._reference_data = pd.DataFrame(ref_data)
        self._reference_ingestion_id = ingestion_id
        logger.info(f"Loaded reference from ingestion: {ingestion_id}")
        return True
    
    def detect_drift(self, 
                     current_data: pd.DataFrame,
                     current_ingestion_id: str = None,
                     columns: List[str] = None) -> DriftReport:
        """
        Detect data drift using Kolmogorov-Smirnov test.
        
        Args:
            current_data: Current DataFrame to compare
            current_ingestion_id: ID of current data ingestion
            columns: Specific columns to test (None = all numeric)
        
        Returns:
            DriftReport with test results
        """
        if self._reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")
        
        drift_id = generate_id("DRIFT_")
        timestamp = datetime.utcnow().isoformat()
        
        # Determine columns to test
        if columns is None:
            columns = self.monitor_config.monitor_columns or []
            if not columns:
                # Auto-select: numerical columns present in both datasets
                ref_numeric = set(self._reference_data.select_dtypes(include=[np.number]).columns)
                cur_numeric = set(current_data.select_dtypes(include=[np.number]).columns)
                columns = list(ref_numeric & cur_numeric)
        
        if not columns:
            logger.warning("No columns to test for drift")
            return self._create_empty_report(drift_id, timestamp)
        
        # Run K-S test on each column
        test_results: Dict[str, DriftTestResult] = {}
        columns_with_drift: List[str] = []
        ks_statistics: Dict[str, float] = {}
        p_values: Dict[str, float] = {}
        
        for col in columns:
            if col not in self._reference_data.columns or col not in current_data.columns:
                logger.debug(f"Skipping column {col}: not in both datasets")
                continue
            
            ref_values = self._reference_data[col].values
            cur_values = current_data[col].values
            
            # Check minimum samples
            ref_valid = np.sum(~np.isnan(ref_values))
            cur_valid = np.sum(~np.isnan(cur_values))
            
            if ref_valid < self.monitor_config.min_samples or cur_valid < self.monitor_config.min_samples:
                logger.debug(f"Skipping column {col}: insufficient samples")
                continue
            
            # Perform K-S test
            statistic, p_value = StatisticalTests.kolmogorov_smirnov_test(ref_values, cur_values)
            
            # Determine if drift detected
            drift_detected = p_value < self.monitor_config.ks_significance_level
            
            # Calculate additional stats
            ref_clean = ref_values[~np.isnan(ref_values)]
            cur_clean = cur_values[~np.isnan(cur_values)]
            
            result = DriftTestResult(
                column_name=col,
                test_name="kolmogorov_smirnov",
                statistic=statistic,
                p_value=p_value,
                drift_detected=drift_detected,
                threshold=self.monitor_config.ks_significance_level,
                reference_stats={
                    'mean': float(np.mean(ref_clean)),
                    'std': float(np.std(ref_clean)),
                    'min': float(np.min(ref_clean)),
                    'max': float(np.max(ref_clean)),
                    'n_samples': len(ref_clean)
                },
                current_stats={
                    'mean': float(np.mean(cur_clean)),
                    'std': float(np.std(cur_clean)),
                    'min': float(np.min(cur_clean)),
                    'max': float(np.max(cur_clean)),
                    'n_samples': len(cur_clean)
                }
            )
            
            test_results[col] = result
            ks_statistics[col] = statistic
            p_values[col] = p_value
            
            if drift_detected:
                columns_with_drift.append(col)
                logger.info(f"Drift detected in '{col}': KS={statistic:.4f}, p={p_value:.4f}")
        
        # Calculate overall drift metrics
        total_tested = len(test_results)
        drift_percentage = len(columns_with_drift) / total_tested if total_tested > 0 else 0
        overall_drift = drift_percentage >= self.monitor_config.drift_threshold
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_drift, drift_percentage, columns_with_drift, test_results
        )
        
        # Create report
        report = DriftReport(
            drift_id=drift_id,
            timestamp=timestamp,
            overall_drift_detected=overall_drift,
            drift_percentage=drift_percentage,
            total_columns_tested=total_tested,
            columns_with_drift=columns_with_drift,
            test_results=test_results,
            recommendations=recommendations
        )
        
        # Log to database
        log = DriftLog(
            drift_id=drift_id,
            reference_ingestion_id=self._reference_ingestion_id or "unknown",
            current_ingestion_id=current_ingestion_id or "unknown",
            columns_tested=list(test_results.keys()),
            drift_detected=overall_drift,
            drift_columns=columns_with_drift,
            ks_statistics=ks_statistics,
            p_values=p_values,
            timestamp=timestamp,
            status="completed"
        )
        self.db.log_drift(log)
        
        self._last_report = report
        
        logger.info(f"Drift detection complete: {len(columns_with_drift)}/{total_tested} columns drifted")
        
        return report
    
    def detect_drift_with_psi(self,
                               current_data: pd.DataFrame,
                               psi_threshold: float = 0.2,
                               columns: List[str] = None) -> Dict[str, float]:
        """
        Detect drift using Population Stability Index.
        
        Args:
            current_data: Current DataFrame
            psi_threshold: Threshold for significant drift (default 0.2)
            columns: Columns to test
        
        Returns:
            Dictionary of column -> PSI value
        """
        if self._reference_data is None:
            raise ValueError("Reference data not set")
        
        if columns is None:
            ref_numeric = set(self._reference_data.select_dtypes(include=[np.number]).columns)
            cur_numeric = set(current_data.select_dtypes(include=[np.number]).columns)
            columns = list(ref_numeric & cur_numeric)
        
        psi_scores = {}
        for col in columns:
            if col in self._reference_data.columns and col in current_data.columns:
                psi = StatisticalTests.population_stability_index(
                    self._reference_data[col].values,
                    current_data[col].values
                )
                psi_scores[col] = psi
                
                if psi >= psi_threshold:
                    logger.warning(f"High PSI for '{col}': {psi:.4f}")
        
        return psi_scores
    
    def detect_categorical_drift(self,
                                  current_data: pd.DataFrame,
                                  columns: List[str] = None) -> Dict[str, DriftTestResult]:
        """
        Detect drift in categorical columns using Chi-squared test.
        
        Args:
            current_data: Current DataFrame
            columns: Categorical columns to test
        
        Returns:
            Dictionary of test results
        """
        if self._reference_data is None:
            raise ValueError("Reference data not set")
        
        if columns is None:
            ref_cat = set(self._reference_data.select_dtypes(include=['object', 'category']).columns)
            cur_cat = set(current_data.select_dtypes(include=['object', 'category']).columns)
            columns = list(ref_cat & cur_cat)
        
        results = {}
        for col in columns:
            if col in self._reference_data.columns and col in current_data.columns:
                ref_values = self._reference_data[col].dropna().values
                cur_values = current_data[col].dropna().values
                
                stat, p_value = StatisticalTests.chi_squared_test(ref_values, cur_values)
                
                results[col] = DriftTestResult(
                    column_name=col,
                    test_name="chi_squared",
                    statistic=stat,
                    p_value=p_value,
                    drift_detected=p_value < self.monitor_config.ks_significance_level,
                    threshold=self.monitor_config.ks_significance_level
                )
        
        return results
    
    def comprehensive_drift_check(self,
                                   current_data: pd.DataFrame,
                                   current_ingestion_id: str = None) -> Dict[str, Any]:
        """
        Run comprehensive drift detection using multiple methods.
        
        Args:
            current_data: Current DataFrame
            current_ingestion_id: ID of current ingestion
        
        Returns:
            Comprehensive drift report
        """
        # K-S test for numerical columns
        ks_report = self.detect_drift(current_data, current_ingestion_id)
        
        # PSI analysis
        psi_scores = self.detect_drift_with_psi(current_data)
        
        # Categorical drift
        cat_results = self.detect_categorical_drift(current_data)
        
        return {
            'ks_report': ks_report.to_dict(),
            'psi_scores': psi_scores,
            'categorical_drift': {k: v.to_dict() for k, v in cat_results.items()},
            'overall_recommendation': 'RETRAIN' if ks_report.overall_drift_detected else 'MONITOR'
        }
    
    def _generate_recommendations(self,
                                    overall_drift: bool,
                                    drift_percentage: float,
                                    drift_columns: List[str],
                                    results: Dict[str, DriftTestResult]) -> List[str]:
        """Generate actionable recommendations based on drift analysis."""
        recommendations = []
        
        if not overall_drift and drift_percentage == 0:
            recommendations.append("No significant drift detected. Continue monitoring.")
            return recommendations
        
        if overall_drift:
            recommendations.append("CRITICAL: Significant data drift detected. Model retraining recommended.")
            
            if self.monitor_config.auto_retrain:
                recommendations.append("Auto-retrain is enabled. Pipeline will trigger retraining.")
        
        if drift_columns:
            severe_drift = [c for c, r in results.items() if r.drift_detected and r.p_value < 0.01]
            
            if severe_drift:
                recommendations.append(f"Severe drift in: {', '.join(severe_drift)}")
            
            # Check for mean shifts
            for col in drift_columns:
                result = results[col]
                ref_mean = result.reference_stats.get('mean', 0)
                cur_mean = result.current_stats.get('mean', 0)
                
                if ref_mean != 0:
                    pct_change = abs(cur_mean - ref_mean) / abs(ref_mean) * 100
                    if pct_change > 20:
                        recommendations.append(f"'{col}' mean shifted by {pct_change:.1f}%")
        
        if drift_percentage > 0.5:
            recommendations.append("Over 50% of features show drift. Consider investigating data source.")
        
        return recommendations
    
    def _create_empty_report(self, drift_id: str, timestamp: str) -> DriftReport:
        """Create an empty report when no columns can be tested."""
        return DriftReport(
            drift_id=drift_id,
            timestamp=timestamp,
            overall_drift_detected=False,
            drift_percentage=0.0,
            total_columns_tested=0,
            columns_with_drift=[],
            test_results={},
            recommendations=["No columns available for drift testing"]
        )
    
    @property
    def last_report(self) -> Optional[DriftReport]:
        """Get the last drift report."""
        return self._last_report
    
    def should_retrain(self) -> bool:
        """
        Determine if model should be retrained based on last drift check.
        
        Returns:
            True if retraining is recommended
        """
        if self._last_report is None:
            return False
        
        return self._last_report.overall_drift_detected and self.monitor_config.auto_retrain


def simulate_drift(df: pd.DataFrame, 
                   drift_columns: List[str],
                   drift_magnitude: float = 0.5) -> pd.DataFrame:
    """
    Simulate data drift for testing purposes.
    
    Args:
        df: Original DataFrame
        drift_columns: Columns to apply drift to
        drift_magnitude: How much drift to apply (0-1 scale)
    
    Returns:
        DataFrame with simulated drift
    """
    drifted_df = df.copy()
    
    for col in drift_columns:
        if col in drifted_df.columns and drifted_df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
            # Apply drift: shift mean and scale
            original_mean = drifted_df[col].mean()
            original_std = drifted_df[col].std()
            
            # Shift by drift_magnitude * std
            drifted_df[col] = drifted_df[col] + (drift_magnitude * original_std)
            
            # Add some noise
            noise = np.random.normal(0, original_std * drift_magnitude * 0.5, len(drifted_df))
            drifted_df[col] = drifted_df[col] + noise
    
    logger.info(f"Applied drift to columns: {drift_columns}")
    return drifted_df


# =============================================================================
# Standalone detect_drift() — Phase 2 Core Monitoring Engine
# =============================================================================

@dataclass
class DriftResult:
    """
    Result of the standalone detect_drift function.

    Attributes:
        retrain_flag: True if >20 % of features show distribution drift.
        drift_percentage: Fraction of features with detected drift.
        total_features_tested: Number of numerical features evaluated.
        drifted_features: List of column names that drifted.
        details: Per-column dict with KS statistic and p-value.
    """
    retrain_flag: bool
    drift_percentage: float
    total_features_tested: int
    drifted_features: List[str]
    details: Dict[str, Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrain_flag": self.retrain_flag,
            "drift_percentage": round(self.drift_percentage, 4),
            "total_features_tested": self.total_features_tested,
            "drifted_features": self.drifted_features,
            "details": self.details,
        }


def detect_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    significance_level: float = 0.05,
    drift_threshold: float = 0.20,
) -> DriftResult:
    """
    Core Monitoring Engine — detect distribution shifts between reference
    and current data using the Kolmogorov-Smirnov two-sample test.

    For every shared numerical feature the K-S test is performed.
    If the p-value < ``significance_level`` (default 0.05) the feature is
    flagged as drifted.  When **more than 20 %** of features show drift,
    ``retrain_flag`` is set to ``True``.

    Args:
        reference_data: The baseline / training-time DataFrame.
        current_data: The newly observed DataFrame.
        significance_level: p-value threshold for the K-S test (default 0.05).
        drift_threshold: Fraction of drifted features that triggers retraining
            (default 0.20 = 20 %).

    Returns:
        DriftResult dataclass with retrain_flag, stats, and per-column info.
    """
    # Identify shared numerical columns
    ref_numeric = set(reference_data.select_dtypes(include=[np.number]).columns)
    cur_numeric = set(current_data.select_dtypes(include=[np.number]).columns)
    shared_columns = sorted(ref_numeric & cur_numeric)

    if not shared_columns:
        logger.warning("detect_drift: no shared numerical columns found")
        return DriftResult(
            retrain_flag=False,
            drift_percentage=0.0,
            total_features_tested=0,
            drifted_features=[],
            details={},
        )

    drifted: List[str] = []
    details: Dict[str, Dict[str, float]] = {}

    for col in shared_columns:
        ref_vals = reference_data[col].dropna().values
        cur_vals = current_data[col].dropna().values

        if len(ref_vals) < 2 or len(cur_vals) < 2:
            continue

        ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
        drift_detected = p_value < significance_level

        details[col] = {
            "ks_statistic": round(float(ks_stat), 6),
            "p_value": round(float(p_value), 6),
            "drift_detected": drift_detected,
        }

        if drift_detected:
            drifted.append(col)
            logger.info(f"detect_drift: DRIFT in '{col}' — KS={ks_stat:.4f}, p={p_value:.4f}")

    total_tested = len(details)
    drift_pct = len(drifted) / total_tested if total_tested > 0 else 0.0
    retrain_flag = drift_pct > drift_threshold

    if retrain_flag:
        logger.warning(
            f"detect_drift: {drift_pct:.1%} features drifted (>{drift_threshold:.0%}) "
            f"→ retrain_flag = True"
        )
    else:
        logger.info(
            f"detect_drift: {drift_pct:.1%} features drifted "
            f"(≤{drift_threshold:.0%}) → retrain_flag = False"
        )

    return DriftResult(
        retrain_flag=retrain_flag,
        drift_percentage=drift_pct,
        total_features_tested=total_tested,
        drifted_features=drifted,
        details=details,
    )
