"""Compliance and bias detection utilities for Product Feature Analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import confusion_matrix
from collections import Counter
import warnings

warnings.filterwarnings("ignore")


class BiasDetector:
    """Detect bias in feature analysis results."""
    
    def __init__(self):
        """Initialize bias detector."""
        self.bias_metrics = {}
    
    def detect_sentiment_bias(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """Detect sentiment bias in features.
        
        Args:
            features_df: DataFrame with features and sentiment
            
        Returns:
            Dictionary of bias metrics
        """
        bias_metrics = {}
        
        # Overall sentiment distribution
        sentiment_dist = features_df['sentiment'].value_counts()
        total_sentiments = len(features_df)
        
        # Check for extreme sentiment skew
        positive_ratio = len(features_df[features_df['sentiment'] > 0.2]) / total_sentiments
        negative_ratio = len(features_df[features_df['sentiment'] < -0.2]) / total_sentiments
        neutral_ratio = len(features_df[(features_df['sentiment'] >= -0.2) & (features_df['sentiment'] <= 0.2)]) / total_sentiments
        
        bias_metrics['sentiment_balance'] = 1 - abs(positive_ratio - negative_ratio)
        bias_metrics['positive_skew'] = abs(positive_ratio - 0.33)  # Expected balanced distribution
        bias_metrics['negative_skew'] = abs(negative_ratio - 0.33)
        
        return bias_metrics
    
    def detect_feature_bias(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """Detect bias in feature representation.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Dictionary of bias metrics
        """
        bias_metrics = {}
        
        # Feature frequency distribution
        feature_counts = features_df['feature'].value_counts()
        
        # Check for feature dominance
        total_features = len(features_df)
        top_feature_ratio = feature_counts.iloc[0] / total_features
        bias_metrics['feature_dominance'] = top_feature_ratio
        
        # Check for long tail distribution
        n_unique_features = len(feature_counts)
        bias_metrics['feature_diversity'] = n_unique_features / total_features
        
        # Check for equal representation (Gini coefficient)
        bias_metrics['feature_inequality'] = self._calculate_gini(feature_counts.values)
        
        return bias_metrics
    
    def detect_category_bias(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """Detect bias across product categories.
        
        Args:
            features_df: DataFrame with features and categories
            
        Returns:
            Dictionary of bias metrics
        """
        bias_metrics = {}
        
        if 'product_category' not in features_df.columns:
            return bias_metrics
        
        # Category representation
        category_counts = features_df['product_category'].value_counts()
        total_categories = len(features_df)
        
        # Check for category dominance
        top_category_ratio = category_counts.iloc[0] / total_categories
        bias_metrics['category_dominance'] = top_category_ratio
        
        # Check for equal category representation
        n_categories = len(category_counts)
        expected_ratio = 1 / n_categories
        category_inequality = sum(abs(count/total_categories - expected_ratio) for count in category_counts.values)
        bias_metrics['category_inequality'] = category_inequality
        
        return bias_metrics
    
    def _calculate_gini(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for inequality measurement."""
        if len(values) == 0:
            return 0
        
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def comprehensive_bias_analysis(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive bias analysis.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Dictionary with comprehensive bias analysis
        """
        analysis = {}
        
        # Sentiment bias
        analysis['sentiment_bias'] = self.detect_sentiment_bias(features_df)
        
        # Feature bias
        analysis['feature_bias'] = self.detect_feature_bias(features_df)
        
        # Category bias
        analysis['category_bias'] = self.detect_category_bias(features_df)
        
        # Overall bias score
        bias_scores = []
        for category in analysis.values():
            if isinstance(category, dict):
                bias_scores.extend(category.values())
        
        analysis['overall_bias_score'] = np.mean(bias_scores) if bias_scores else 0
        
        # Bias interpretation
        analysis['bias_interpretation'] = self._interpret_bias_score(analysis['overall_bias_score'])
        
        return analysis
    
    def _interpret_bias_score(self, score: float) -> str:
        """Interpret bias score.
        
        Args:
            score: Bias score (0-1)
            
        Returns:
            Interpretation string
        """
        if score < 0.1:
            return "Low bias - Results appear balanced"
        elif score < 0.3:
            return "Moderate bias - Some imbalance detected"
        elif score < 0.5:
            return "High bias - Significant imbalance detected"
        else:
            return "Very high bias - Results may be unreliable"


class ComplianceChecker:
    """Check compliance with data protection and fairness requirements."""
    
    def __init__(self):
        """Initialize compliance checker."""
        self.compliance_results = {}
    
    def check_data_privacy(self, text_data: List[str]) -> Dict[str, Any]:
        """Check for potential privacy violations.
        
        Args:
            text_data: List of text strings to check
            
        Returns:
            Dictionary with privacy check results
        """
        privacy_results = {
            'pii_detected': False,
            'pii_types': [],
            'risk_level': 'low',
            'recommendations': []
        }
        
        # Check for common PII patterns
        import re
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = [re.findall(email_pattern, text) for text in text_data]
        emails = [email for email_list in emails for email in email_list]
        
        # Phone pattern
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phones = [re.findall(phone_pattern, text) for text in text_data]
        phones = [phone for phone_list in phones for phone in phone_list]
        
        # Name pattern (simple heuristic)
        name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        names = [re.findall(name_pattern, text) for text in text_data]
        names = [name for name_list in names for name in name_list]
        
        if emails:
            privacy_results['pii_detected'] = True
            privacy_results['pii_types'].append('email')
            privacy_results['recommendations'].append('Remove or anonymize email addresses')
        
        if phones:
            privacy_results['pii_detected'] = True
            privacy_results['pii_types'].append('phone')
            privacy_results['recommendations'].append('Remove or anonymize phone numbers')
        
        if names:
            privacy_results['pii_detected'] = True
            privacy_results['pii_types'].append('name')
            privacy_results['recommendations'].append('Review potential names for anonymization')
        
        # Determine risk level
        if len(privacy_results['pii_types']) > 2:
            privacy_results['risk_level'] = 'high'
        elif len(privacy_results['pii_types']) > 0:
            privacy_results['risk_level'] = 'medium'
        
        return privacy_results
    
    def check_fairness_metrics(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Check fairness metrics across different groups.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Dictionary with fairness metrics
        """
        fairness_results = {
            'demographic_parity': {},
            'equalized_odds': {},
            'equal_opportunity': {},
            'overall_fairness': 'unknown'
        }
        
        # Check sentiment distribution across categories
        if 'product_category' in features_df.columns:
            category_sentiment = features_df.groupby('product_category')['sentiment'].agg(['mean', 'std', 'count'])
            
            # Demographic parity (similar sentiment across categories)
            sentiment_means = category_sentiment['mean']
            max_sentiment = sentiment_means.max()
            min_sentiment = sentiment_means.min()
            
            fairness_results['demographic_parity'] = {
                'sentiment_range': max_sentiment - min_sentiment,
                'max_sentiment': max_sentiment,
                'min_sentiment': min_sentiment,
                'categories_analyzed': len(sentiment_means)
            }
        
        # Check rating distribution
        if 'rating' in features_df.columns:
            rating_dist = features_df['rating'].value_counts(normalize=True)
            
            # Equal opportunity (fair representation across ratings)
            fairness_results['equal_opportunity'] = {
                'rating_distribution': rating_dist.to_dict(),
                'distribution_entropy': -sum(p * np.log2(p) for p in rating_dist if p > 0)
            }
        
        return fairness_results
    
    def generate_compliance_report(self, features_df: pd.DataFrame, text_data: List[str]) -> Dict[str, Any]:
        """Generate comprehensive compliance report.
        
        Args:
            features_df: DataFrame with features
            text_data: List of text data for privacy check
            
        Returns:
            Dictionary with compliance report
        """
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'privacy_check': self.check_data_privacy(text_data),
            'fairness_check': self.check_fairness_metrics(features_df),
            'bias_analysis': BiasDetector().comprehensive_bias_analysis(features_df),
            'compliance_status': 'unknown',
            'recommendations': []
        }
        
        # Determine overall compliance status
        privacy_risk = report['privacy_check']['risk_level']
        bias_score = report['bias_analysis']['overall_bias_score']
        
        if privacy_risk == 'high' or bias_score > 0.5:
            report['compliance_status'] = 'non_compliant'
            report['recommendations'].append('Address privacy and bias issues before deployment')
        elif privacy_risk == 'medium' or bias_score > 0.3:
            report['compliance_status'] = 'needs_review'
            report['recommendations'].append('Review and mitigate identified issues')
        else:
            report['compliance_status'] = 'compliant'
            report['recommendations'].append('Continue monitoring for bias and privacy issues')
        
        return report


class AuditLogger:
    """Log audit trail for compliance."""
    
    def __init__(self, log_file: str = "logs/audit.log"):
        """Initialize audit logger.
        
        Args:
            log_file: Path to audit log file
        """
        self.log_file = log_file
        self.audit_entries = []
    
    def log_analysis_run(self, config: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Log analysis run for audit trail.
        
        Args:
            config: Configuration used
            results: Analysis results
        """
        entry = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'event_type': 'analysis_run',
            'config_hash': hash(str(config)),
            'data_hash': results.get('data_hash', 'unknown'),
            'results_summary': {
                'n_features': results.get('n_features', 0),
                'n_reviews': results.get('n_reviews', 0),
                'avg_sentiment': results.get('avg_sentiment', 0)
            }
        }
        
        self.audit_entries.append(entry)
        self._write_to_log(entry)
    
    def log_data_access(self, data_type: str, access_reason: str) -> None:
        """Log data access for audit trail.
        
        Args:
            data_type: Type of data accessed
            access_reason: Reason for access
        """
        entry = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'event_type': 'data_access',
            'data_type': data_type,
            'access_reason': access_reason
        }
        
        self.audit_entries.append(entry)
        self._write_to_log(entry)
    
    def _write_to_log(self, entry: Dict[str, Any]) -> None:
        """Write entry to audit log file.
        
        Args:
            entry: Audit log entry
        """
        import json
        from pathlib import Path
        
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get complete audit trail.
        
        Returns:
            List of audit entries
        """
        return self.audit_entries.copy()
