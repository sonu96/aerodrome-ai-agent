"""
Confidence metrics tracking and analysis for the Aerodrome brain.

This module provides comprehensive metrics tracking for confidence scores,
including historical accuracy tracking, prediction validation, and confidence
distribution analysis.
"""

import asyncio
import json
import logging
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import structlog
from pydantic import BaseModel, Field

from .confidence_scorer import MemoryCategory, ConfidenceFactors, MemoryItem

logger = structlog.get_logger(__name__)


class MetricType(Enum):
    """Types of confidence metrics."""
    
    ACCURACY_TREND = "accuracy_trend"
    CONFIDENCE_DISTRIBUTION = "confidence_distribution"
    PREDICTION_VALIDATION = "prediction_validation"
    FACTOR_CORRELATION = "factor_correlation"
    CATEGORY_PERFORMANCE = "category_performance"
    TEMPORAL_ANALYSIS = "temporal_analysis"


@dataclass
class AccuracyMetric:
    """Represents an accuracy measurement."""
    
    timestamp: datetime
    memory_id: str
    category: MemoryCategory
    predicted_confidence: float
    actual_accuracy: float
    factors: Optional[ConfidenceFactors] = None


@dataclass
class PredictionValidation:
    """Validation result for a prediction."""
    
    memory_id: str
    prediction_timestamp: datetime
    validation_timestamp: datetime
    predicted_outcome: Any
    actual_outcome: Any
    was_accurate: bool
    confidence_at_prediction: float
    category: MemoryCategory


@dataclass
class ConfidenceDistribution:
    """Distribution of confidence scores."""
    
    category: MemoryCategory
    timestamp: datetime
    score_buckets: Dict[str, int]  # e.g., "0.0-0.1": 5, "0.1-0.2": 10
    mean: float
    median: float
    std_dev: float
    percentiles: Dict[str, float]  # e.g., "p90": 0.85, "p95": 0.92


@dataclass
class FactorCorrelation:
    """Correlation between confidence factors and accuracy."""
    
    factor_name: str
    correlation_coefficient: float
    p_value: float
    sample_size: int
    timestamp: datetime


class MetricsCollector:
    """Collects and stores confidence metrics."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.accuracy_history: deque = deque(maxlen=max_history)
        self.prediction_validations: deque = deque(maxlen=max_history)
        self.confidence_snapshots: List[Dict] = []
        self.logger = structlog.get_logger(__name__)
    
    async def record_accuracy(
        self,
        memory_id: str,
        category: MemoryCategory,
        predicted_confidence: float,
        actual_accuracy: float,
        factors: Optional[ConfidenceFactors] = None
    ) -> None:
        """Record an accuracy measurement."""
        try:
            metric = AccuracyMetric(
                timestamp=datetime.now(),
                memory_id=memory_id,
                category=category,
                predicted_confidence=predicted_confidence,
                actual_accuracy=actual_accuracy,
                factors=factors
            )
            
            self.accuracy_history.append(metric)
            
            await self.logger.adebug(
                "Recorded accuracy metric",
                memory_id=memory_id,
                category=category.value,
                predicted_confidence=predicted_confidence,
                actual_accuracy=actual_accuracy
            )
            
        except Exception as e:
            await self.logger.aerror(
                "Error recording accuracy metric",
                memory_id=memory_id,
                error=str(e)
            )
    
    async def record_prediction_validation(
        self,
        memory_id: str,
        prediction_timestamp: datetime,
        predicted_outcome: Any,
        actual_outcome: Any,
        confidence_at_prediction: float,
        category: MemoryCategory
    ) -> None:
        """Record a prediction validation result."""
        try:
            # Determine if prediction was accurate
            was_accurate = self._evaluate_prediction_accuracy(
                predicted_outcome, actual_outcome
            )
            
            validation = PredictionValidation(
                memory_id=memory_id,
                prediction_timestamp=prediction_timestamp,
                validation_timestamp=datetime.now(),
                predicted_outcome=predicted_outcome,
                actual_outcome=actual_outcome,
                was_accurate=was_accurate,
                confidence_at_prediction=confidence_at_prediction,
                category=category
            )
            
            self.prediction_validations.append(validation)
            
            await self.logger.ainfo(
                "Recorded prediction validation",
                memory_id=memory_id,
                was_accurate=was_accurate,
                confidence=confidence_at_prediction
            )
            
        except Exception as e:
            await self.logger.aerror(
                "Error recording prediction validation",
                memory_id=memory_id,
                error=str(e)
            )
    
    def _evaluate_prediction_accuracy(
        self, 
        predicted: Any, 
        actual: Any, 
        tolerance: float = 0.1
    ) -> bool:
        """Evaluate if a prediction was accurate."""
        try:
            # Handle numerical predictions
            if isinstance(predicted, (int, float)) and isinstance(actual, (int, float)):
                if actual == 0:
                    return abs(predicted) <= tolerance
                return abs(predicted - actual) / abs(actual) <= tolerance
            
            # Handle boolean predictions
            if isinstance(predicted, bool) and isinstance(actual, bool):
                return predicted == actual
            
            # Handle string predictions (exact match)
            if isinstance(predicted, str) and isinstance(actual, str):
                return predicted.lower() == actual.lower()
            
            # Handle list/dict predictions (structural comparison)
            if type(predicted) == type(actual):
                return predicted == actual
            
            return False
            
        except Exception:
            return False
    
    async def take_confidence_snapshot(
        self, 
        memory_items: List[MemoryItem]
    ) -> None:
        """Take a snapshot of current confidence distribution."""
        try:
            snapshot = {
                "timestamp": datetime.now(),
                "total_items": len(memory_items),
                "by_category": {},
                "overall_stats": {}
            }
            
            # Group by category
            category_scores = defaultdict(list)
            all_scores = []
            
            for item in memory_items:
                category_scores[item.category].append(item.confidence)
                all_scores.append(item.confidence)
            
            # Calculate category-specific stats
            for category, scores in category_scores.items():
                if scores:
                    snapshot["by_category"][category.value] = {
                        "count": len(scores),
                        "mean": statistics.mean(scores),
                        "median": statistics.median(scores),
                        "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                        "min": min(scores),
                        "max": max(scores),
                        "percentiles": {
                            "p25": np.percentile(scores, 25),
                            "p75": np.percentile(scores, 75),
                            "p90": np.percentile(scores, 90),
                            "p95": np.percentile(scores, 95)
                        }
                    }
            
            # Calculate overall stats
            if all_scores:
                snapshot["overall_stats"] = {
                    "mean": statistics.mean(all_scores),
                    "median": statistics.median(all_scores),
                    "std_dev": statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0,
                    "min": min(all_scores),
                    "max": max(all_scores),
                    "percentiles": {
                        "p25": np.percentile(all_scores, 25),
                        "p75": np.percentile(all_scores, 75),
                        "p90": np.percentile(all_scores, 90),
                        "p95": np.percentile(all_scores, 95)
                    }
                }
            
            self.confidence_snapshots.append(snapshot)
            
            # Keep only recent snapshots
            if len(self.confidence_snapshots) > 1000:
                self.confidence_snapshots = self.confidence_snapshots[-1000:]
            
            await self.logger.adebug(
                "Took confidence snapshot",
                total_items=len(memory_items),
                categories=len(category_scores)
            )
            
        except Exception as e:
            await self.logger.aerror(
                "Error taking confidence snapshot",
                error=str(e)
            )


class MetricsAnalyzer:
    """Analyzes confidence metrics to provide insights."""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.logger = structlog.get_logger(__name__)
    
    async def analyze_accuracy_trends(
        self, 
        time_window: timedelta = timedelta(days=7)
    ) -> Dict[str, Any]:
        """Analyze accuracy trends over time."""
        try:
            cutoff_time = datetime.now() - time_window
            recent_metrics = [
                m for m in self.collector.accuracy_history
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return {"error": "No recent accuracy metrics available"}
            
            # Group by category
            category_trends = defaultdict(list)
            for metric in recent_metrics:
                category_trends[metric.category].append({
                    "timestamp": metric.timestamp,
                    "predicted_confidence": metric.predicted_confidence,
                    "actual_accuracy": metric.actual_accuracy,
                    "error": abs(metric.predicted_confidence - metric.actual_accuracy)
                })
            
            results = {}
            for category, metrics in category_trends.items():
                if len(metrics) < 2:
                    continue
                
                # Calculate trends
                errors = [m["error"] for m in metrics]
                predicted_confidences = [m["predicted_confidence"] for m in metrics]
                actual_accuracies = [m["actual_accuracy"] for m in metrics]
                
                # Linear regression for trend
                timestamps_numeric = [
                    (m["timestamp"] - cutoff_time).total_seconds() 
                    for m in metrics
                ]
                
                error_trend = np.polyfit(timestamps_numeric, errors, 1)[0] if len(errors) > 1 else 0
                
                results[category.value] = {
                    "sample_size": len(metrics),
                    "mean_error": statistics.mean(errors),
                    "median_error": statistics.median(errors),
                    "error_trend": float(error_trend),  # Positive = getting worse
                    "mean_predicted_confidence": statistics.mean(predicted_confidences),
                    "mean_actual_accuracy": statistics.mean(actual_accuracies),
                    "correlation": np.corrcoef(predicted_confidences, actual_accuracies)[0, 1]
                    if len(predicted_confidences) > 1 else 0.0
                }
            
            await self.logger.ainfo(
                "Analyzed accuracy trends",
                time_window_days=time_window.days,
                categories_analyzed=len(results)
            )
            
            return results
            
        except Exception as e:
            await self.logger.aerror(
                "Error analyzing accuracy trends",
                error=str(e)
            )
            return {"error": str(e)}
    
    async def analyze_factor_correlations(
        self,
        time_window: timedelta = timedelta(days=7)
    ) -> List[FactorCorrelation]:
        """Analyze correlations between confidence factors and accuracy."""
        try:
            cutoff_time = datetime.now() - time_window
            recent_metrics = [
                m for m in self.collector.accuracy_history
                if m.timestamp >= cutoff_time and m.factors is not None
            ]
            
            if len(recent_metrics) < 10:
                return []
            
            correlations = []
            
            # Extract factor values and actual accuracies
            factor_data = {
                "data_source_reliability": [],
                "historical_accuracy": [],
                "recency": [],
                "corroboration": [],
                "sample_size": [],
                "prediction_success_rate": []
            }
            
            actual_accuracies = []
            
            for metric in recent_metrics:
                actual_accuracies.append(metric.actual_accuracy)
                factor_data["data_source_reliability"].append(metric.factors.data_source_reliability)
                factor_data["historical_accuracy"].append(metric.factors.historical_accuracy)
                factor_data["recency"].append(metric.factors.recency)
                factor_data["corroboration"].append(metric.factors.corroboration)
                factor_data["sample_size"].append(metric.factors.sample_size)
                factor_data["prediction_success_rate"].append(metric.factors.prediction_success_rate)
            
            # Calculate correlations
            for factor_name, factor_values in factor_data.items():
                if len(set(factor_values)) > 1:  # Need variation in factor values
                    correlation_matrix = np.corrcoef(factor_values, actual_accuracies)
                    correlation_coeff = correlation_matrix[0, 1]
                    
                    # Simple p-value estimation (would need proper statistical test in production)
                    p_value = 0.05 if abs(correlation_coeff) > 0.3 else 0.1
                    
                    correlations.append(FactorCorrelation(
                        factor_name=factor_name,
                        correlation_coefficient=float(correlation_coeff),
                        p_value=p_value,
                        sample_size=len(factor_values),
                        timestamp=datetime.now()
                    ))
            
            await self.logger.ainfo(
                "Analyzed factor correlations",
                correlations_found=len(correlations),
                sample_size=len(recent_metrics)
            )
            
            return correlations
            
        except Exception as e:
            await self.logger.aerror(
                "Error analyzing factor correlations",
                error=str(e)
            )
            return []
    
    async def analyze_prediction_performance(
        self,
        time_window: timedelta = timedelta(days=7)
    ) -> Dict[str, Any]:
        """Analyze prediction performance by confidence levels."""
        try:
            cutoff_time = datetime.now() - time_window
            recent_validations = [
                v for v in self.collector.prediction_validations
                if v.validation_timestamp >= cutoff_time
            ]
            
            if not recent_validations:
                return {"error": "No recent prediction validations available"}
            
            # Group by confidence buckets
            confidence_buckets = {
                "0.0-0.3": [],
                "0.3-0.5": [],
                "0.5-0.7": [],
                "0.7-0.9": [],
                "0.9-1.0": []
            }
            
            for validation in recent_validations:
                conf = validation.confidence_at_prediction
                if conf < 0.3:
                    bucket = "0.0-0.3"
                elif conf < 0.5:
                    bucket = "0.3-0.5"
                elif conf < 0.7:
                    bucket = "0.5-0.7"
                elif conf < 0.9:
                    bucket = "0.7-0.9"
                else:
                    bucket = "0.9-1.0"
                
                confidence_buckets[bucket].append(validation)
            
            # Calculate performance for each bucket
            bucket_performance = {}
            for bucket, validations in confidence_buckets.items():
                if validations:
                    accuracy_rate = sum(v.was_accurate for v in validations) / len(validations)
                    avg_confidence = statistics.mean(v.confidence_at_prediction for v in validations)
                    
                    bucket_performance[bucket] = {
                        "sample_size": len(validations),
                        "accuracy_rate": accuracy_rate,
                        "avg_confidence": avg_confidence,
                        "calibration_error": abs(accuracy_rate - avg_confidence)
                    }
            
            # Overall calibration analysis
            all_confidences = [v.confidence_at_prediction for v in recent_validations]
            all_accuracies = [float(v.was_accurate) for v in recent_validations]
            
            overall_stats = {
                "total_predictions": len(recent_validations),
                "overall_accuracy": sum(all_accuracies) / len(all_accuracies),
                "avg_confidence": statistics.mean(all_confidences),
                "confidence_accuracy_correlation": np.corrcoef(all_confidences, all_accuracies)[0, 1]
                if len(all_confidences) > 1 else 0.0,
                "calibration_buckets": bucket_performance
            }
            
            await self.logger.ainfo(
                "Analyzed prediction performance",
                total_predictions=len(recent_validations),
                buckets_analyzed=len([b for b in bucket_performance if bucket_performance[b]])
            )
            
            return overall_stats
            
        except Exception as e:
            await self.logger.aerror(
                "Error analyzing prediction performance",
                error=str(e)
            )
            return {"error": str(e)}
    
    async def generate_confidence_insights(self) -> Dict[str, Any]:
        """Generate comprehensive confidence system insights."""
        try:
            insights = {
                "timestamp": datetime.now().isoformat(),
                "accuracy_trends": await self.analyze_accuracy_trends(),
                "factor_correlations": [
                    {
                        "factor": corr.factor_name,
                        "correlation": corr.correlation_coefficient,
                        "significance": "high" if abs(corr.correlation_coefficient) > 0.5 else "medium" if abs(corr.correlation_coefficient) > 0.3 else "low"
                    }
                    for corr in await self.analyze_factor_correlations()
                ],
                "prediction_performance": await self.analyze_prediction_performance(),
                "recommendations": []
            }
            
            # Generate recommendations based on analysis
            recommendations = []
            
            # Check for poorly calibrated confidence
            pred_performance = insights.get("prediction_performance", {})
            if isinstance(pred_performance, dict) and "calibration_buckets" in pred_performance:
                for bucket, stats in pred_performance["calibration_buckets"].items():
                    if stats["calibration_error"] > 0.2:
                        recommendations.append(
                            f"High calibration error in confidence bucket {bucket}. "
                            f"Consider adjusting factor weights."
                        )
            
            # Check for factor correlations
            factor_corrs = insights.get("factor_correlations", [])
            weak_factors = [
                f["factor"] for f in factor_corrs 
                if abs(f["correlation"]) < 0.1 and f["significance"] == "low"
            ]
            if weak_factors:
                recommendations.append(
                    f"Factors {weak_factors} show weak correlation with accuracy. "
                    f"Consider reducing their weights."
                )
            
            insights["recommendations"] = recommendations
            
            return insights
            
        except Exception as e:
            await self.logger.aerror(
                "Error generating confidence insights",
                error=str(e)
            )
            return {"error": str(e)}


class MetricsReporter:
    """Generates reports and visualizations for confidence metrics."""
    
    def __init__(self, analyzer: MetricsAnalyzer):
        self.analyzer = analyzer
        self.logger = structlog.get_logger(__name__)
    
    async def generate_daily_report(
        self, 
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Generate daily confidence metrics report."""
        try:
            report = {
                "report_date": datetime.now().date().isoformat(),
                "generated_at": datetime.now().isoformat(),
                "summary": {},
                "detailed_analysis": {}
            }
            
            # Get comprehensive insights
            insights = await self.analyzer.generate_confidence_insights()
            report["detailed_analysis"] = insights
            
            # Generate executive summary
            summary = {
                "total_metrics_collected": len(self.analyzer.collector.accuracy_history),
                "recent_predictions_validated": len([
                    v for v in self.analyzer.collector.prediction_validations
                    if (datetime.now() - v.validation_timestamp).days < 1
                ]),
                "system_health": "good"  # Would be based on thresholds
            }
            
            # Determine system health
            pred_perf = insights.get("prediction_performance", {})
            if isinstance(pred_perf, dict) and "overall_accuracy" in pred_perf:
                overall_acc = pred_perf["overall_accuracy"]
                if overall_acc > 0.8:
                    summary["system_health"] = "excellent"
                elif overall_acc > 0.6:
                    summary["system_health"] = "good"
                elif overall_acc > 0.4:
                    summary["system_health"] = "fair"
                else:
                    summary["system_health"] = "poor"
            
            report["summary"] = summary
            
            # Save report if path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                
                await self.logger.ainfo(
                    "Generated daily report",
                    output_path=str(output_path),
                    system_health=summary["system_health"]
                )
            
            return report
            
        except Exception as e:
            await self.logger.aerror(
                "Error generating daily report",
                error=str(e)
            )
            return {"error": str(e)}
    
    def export_metrics_to_dataframe(self) -> pd.DataFrame:
        """Export metrics to pandas DataFrame for analysis."""
        try:
            records = []
            
            for metric in self.analyzer.collector.accuracy_history:
                record = {
                    "timestamp": metric.timestamp,
                    "memory_id": metric.memory_id,
                    "category": metric.category.value,
                    "predicted_confidence": metric.predicted_confidence,
                    "actual_accuracy": metric.actual_accuracy,
                    "error": abs(metric.predicted_confidence - metric.actual_accuracy)
                }
                
                # Add factor data if available
                if metric.factors:
                    record.update({
                        "data_source_reliability": metric.factors.data_source_reliability,
                        "historical_accuracy": metric.factors.historical_accuracy,
                        "recency": metric.factors.recency,
                        "corroboration": metric.factors.corroboration,
                        "sample_size": metric.factors.sample_size,
                        "prediction_success_rate": metric.factors.prediction_success_rate
                    })
                
                records.append(record)
            
            return pd.DataFrame(records)
            
        except Exception as e:
            logger.error("Error exporting metrics to DataFrame", error=str(e))
            return pd.DataFrame()


class ConfidenceMetricsConfig(BaseModel):
    """Configuration for confidence metrics system."""
    
    max_history: int = Field(default=10000, description="Maximum metrics to keep in memory")
    snapshot_interval_minutes: int = Field(default=60, description="Interval for confidence snapshots")
    report_generation_hour: int = Field(default=6, description="Hour to generate daily reports (0-23)")
    export_directory: str = Field(default="./metrics_exports", description="Directory for metric exports")
    
    # Thresholds for alerts
    calibration_error_threshold: float = Field(default=0.2, description="Threshold for calibration errors")
    accuracy_drop_threshold: float = Field(default=0.1, description="Threshold for accuracy drops")
    
    # Retention settings
    accuracy_history_days: int = Field(default=30, description="Days to keep accuracy history")
    prediction_validation_days: int = Field(default=30, description="Days to keep prediction validations")


async def main():
    """Example usage of the confidence metrics system."""
    # Initialize components
    collector = MetricsCollector(max_history=1000)
    analyzer = MetricsAnalyzer(collector)
    reporter = MetricsReporter(analyzer)
    
    # Simulate some metrics
    from .confidence_scorer import MemoryCategory, ConfidenceFactors
    
    # Record some accuracy metrics
    await collector.record_accuracy(
        memory_id="test_1",
        category=MemoryCategory.POOL_PERFORMANCE,
        predicted_confidence=0.8,
        actual_accuracy=0.75,
        factors=ConfidenceFactors(
            data_source_reliability=0.9,
            historical_accuracy=0.7,
            recency=0.8,
            corroboration=0.6,
            sample_size=0.9,
            prediction_success_rate=0.8
        )
    )
    
    # Record prediction validation
    await collector.record_prediction_validation(
        memory_id="test_1",
        prediction_timestamp=datetime.now() - timedelta(hours=1),
        predicted_outcome=1000000,  # Predicted volume
        actual_outcome=950000,  # Actual volume
        confidence_at_prediction=0.8,
        category=MemoryCategory.POOL_PERFORMANCE
    )
    
    # Analyze trends
    trends = await analyzer.analyze_accuracy_trends()
    print("Accuracy trends:")
    print(json.dumps(trends, indent=2, default=str))
    
    # Generate insights
    insights = await analyzer.generate_confidence_insights()
    print("\nConfidence insights:")
    print(json.dumps(insights, indent=2, default=str))
    
    # Generate daily report
    report = await reporter.generate_daily_report()
    print("\nDaily report summary:")
    print(json.dumps(report.get("summary", {}), indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())