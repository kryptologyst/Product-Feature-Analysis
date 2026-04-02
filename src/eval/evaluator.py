"""Evaluation metrics and analysis for Product Feature Analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


class FeatureAnalysisEvaluator:
    """Evaluate feature analysis results."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.results = {}
    
    def calculate_sentiment_metrics(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate sentiment analysis metrics.
        
        Args:
            predictions: Predicted sentiment scores
            targets: True sentiment scores
            
        Returns:
            Dictionary of metrics
        """
        # Convert continuous scores to binary for classification metrics
        pred_binary = (predictions > 0).astype(int)
        target_binary = (targets > 0).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(target_binary, pred_binary),
            'precision': precision_score(target_binary, pred_binary, zero_division=0),
            'recall': recall_score(target_binary, pred_binary, zero_division=0),
            'f1_score': f1_score(target_binary, pred_binary, zero_division=0),
        }
        
        # Regression metrics for continuous sentiment
        metrics.update({
            'mae': np.mean(np.abs(predictions - targets)),
            'rmse': np.sqrt(np.mean((predictions - targets) ** 2)),
            'correlation': np.corrcoef(predictions, targets)[0, 1],
            'r2_score': 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
        })
        
        return metrics
    
    def calculate_feature_metrics(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature extraction metrics.
        
        Args:
            features_df: DataFrame with extracted features
            
        Returns:
            Dictionary of feature metrics
        """
        metrics = {}
        
        # Feature coverage
        total_reviews = features_df['review_id'].nunique()
        features_with_sentiment = features_df[features_df['sentiment'] != 0]
        metrics['feature_coverage'] = len(features_with_sentiment) / total_reviews
        
        # Feature diversity
        unique_features = features_df['feature'].nunique()
        metrics['feature_diversity'] = unique_features
        
        # Sentiment distribution
        sentiment_dist = features_df['sentiment'].value_counts()
        metrics['sentiment_balance'] = 1 - abs(sentiment_dist.get(1, 0) - sentiment_dist.get(-1, 0)) / len(features_df)
        
        # Feature frequency distribution
        feature_counts = features_df['feature'].value_counts()
        metrics['frequency_gini'] = self._calculate_gini(feature_counts.values)
        
        return metrics
    
    def _calculate_gini(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for inequality measurement."""
        if len(values) == 0:
            return 0
        
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def calculate_business_metrics(
        self, 
        features_df: pd.DataFrame,
        products_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Calculate business-relevant metrics.
        
        Args:
            features_df: DataFrame with features
            products_df: DataFrame with product information
            
        Returns:
            Dictionary of business metrics
        """
        metrics = {}
        
        # Actionability score
        actionable_features = features_df[
            (features_df['sentiment'] < -0.2) | (features_df['sentiment'] > 0.2)
        ]
        metrics['actionability_score'] = len(actionable_features) / len(features_df)
        
        # Insight quality
        high_frequency_features = features_df[features_df['frequency'] > features_df['frequency'].quantile(0.8)]
        metrics['insight_quality'] = len(high_frequency_features) / len(features_df)
        
        # Product coverage
        if products_df is not None:
            products_with_features = features_df['product_id'].nunique()
            total_products = len(products_df)
            metrics['product_coverage'] = products_with_features / total_products
        
        # Category insights
        category_insights = features_df.groupby('product_category').agg({
            'feature': 'nunique',
            'sentiment': ['mean', 'std']
        })
        metrics['category_insight_depth'] = category_insights['feature']['nunique'].mean()
        
        return metrics
    
    def evaluate_model_performance(
        self,
        features_df: pd.DataFrame,
        products_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation.
        
        Args:
            features_df: DataFrame with extracted features
            products_df: DataFrame with product information
            
        Returns:
            Dictionary with all evaluation results
        """
        results = {}
        
        # Sentiment metrics
        if 'sentiment' in features_df.columns and 'overall_sentiment' in features_df.columns:
            results['sentiment_metrics'] = self.calculate_sentiment_metrics(
                features_df['sentiment'].values,
                features_df['overall_sentiment'].values
            )
        
        # Feature metrics
        results['feature_metrics'] = self.calculate_feature_metrics(features_df)
        
        # Business metrics
        results['business_metrics'] = self.calculate_business_metrics(features_df, products_df)
        
        # Top features analysis
        results['top_features'] = self._analyze_top_features(features_df)
        
        # Sentiment distribution analysis
        results['sentiment_analysis'] = self._analyze_sentiment_distribution(features_df)
        
        return results
    
    def _analyze_top_features(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze top features by importance.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Analysis of top features
        """
        # Calculate feature importance
        feature_stats = features_df.groupby('feature').agg({
            'sentiment': ['count', 'mean', 'std'],
            'rating': 'mean'
        }).round(3)
        
        feature_stats.columns = ['frequency', 'avg_sentiment', 'sentiment_std', 'avg_rating']
        feature_stats['importance'] = (
            feature_stats['frequency'] * 0.4 +
            abs(feature_stats['avg_sentiment']) * 0.3 +
            (1 - feature_stats['sentiment_std'].fillna(0)) * 0.3
        )
        
        top_features = feature_stats.sort_values('importance', ascending=False).head(10)
        
        return {
            'top_positive': top_features[top_features['avg_sentiment'] > 0.2].head(5).to_dict(),
            'top_negative': top_features[top_features['avg_sentiment'] < -0.2].head(5).to_dict(),
            'most_frequent': top_features.sort_values('frequency', ascending=False).head(5).to_dict()
        }
    
    def _analyze_sentiment_distribution(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sentiment distribution.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Sentiment distribution analysis
        """
        sentiment_counts = features_df['sentiment'].apply(
            lambda x: 'positive' if x > 0.2 else 'negative' if x < -0.2 else 'neutral'
        ).value_counts()
        
        return {
            'distribution': sentiment_counts.to_dict(),
            'overall_sentiment': features_df['sentiment'].mean(),
            'sentiment_volatility': features_df['sentiment'].std()
        }
    
    def create_leaderboard(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create a leaderboard of results.
        
        Args:
            results: Evaluation results
            
        Returns:
            Leaderboard DataFrame
        """
        leaderboard_data = []
        
        # Add sentiment metrics
        if 'sentiment_metrics' in results:
            for metric, value in results['sentiment_metrics'].items():
                leaderboard_data.append({
                    'Metric': f"Sentiment_{metric}",
                    'Value': value,
                    'Category': 'Sentiment Analysis',
                    'Higher_is_Better': metric in ['accuracy', 'precision', 'recall', 'f1_score', 'correlation', 'r2_score']
                })
        
        # Add feature metrics
        if 'feature_metrics' in results:
            for metric, value in results['feature_metrics'].items():
                leaderboard_data.append({
                    'Metric': f"Feature_{metric}",
                    'Value': value,
                    'Category': 'Feature Extraction',
                    'Higher_is_Better': True
                })
        
        # Add business metrics
        if 'business_metrics' in results:
            for metric, value in results['business_metrics'].items():
                leaderboard_data.append({
                    'Metric': f"Business_{metric}",
                    'Value': value,
                    'Category': 'Business Impact',
                    'Higher_is_Better': True
                })
        
        leaderboard_df = pd.DataFrame(leaderboard_data)
        
        # Sort by category and value
        leaderboard_df = leaderboard_df.sort_values(['Category', 'Value'], ascending=[True, False])
        
        return leaderboard_df
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a text report of results.
        
        Args:
            results: Evaluation results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("PRODUCT FEATURE ANALYSIS EVALUATION REPORT")
        report.append("=" * 60)
        
        # Sentiment Analysis Results
        if 'sentiment_metrics' in results:
            report.append("\nSENTIMENT ANALYSIS METRICS:")
            report.append("-" * 30)
            for metric, value in results['sentiment_metrics'].items():
                report.append(f"{metric.upper()}: {value:.4f}")
        
        # Feature Extraction Results
        if 'feature_metrics' in results:
            report.append("\nFEATURE EXTRACTION METRICS:")
            report.append("-" * 30)
            for metric, value in results['feature_metrics'].items():
                report.append(f"{metric.upper()}: {value:.4f}")
        
        # Business Impact Results
        if 'business_metrics' in results:
            report.append("\nBUSINESS IMPACT METRICS:")
            report.append("-" * 30)
            for metric, value in results['business_metrics'].items():
                report.append(f"{metric.upper()}: {value:.4f}")
        
        # Top Features
        if 'top_features' in results:
            report.append("\nTOP FEATURES ANALYSIS:")
            report.append("-" * 30)
            
            if 'top_positive' in results['top_features']:
                report.append("Most Positive Features:")
                for feature, stats in list(results['top_features']['top_positive'].items())[:3]:
                    report.append(f"  - {feature}: {stats['avg_sentiment']:.3f}")
            
            if 'top_negative' in results['top_features']:
                report.append("Most Negative Features:")
                for feature, stats in list(results['top_features']['top_negative'].items())[:3]:
                    report.append(f"  - {feature}: {stats['avg_sentiment']:.3f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
