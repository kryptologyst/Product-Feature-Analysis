"""Main application for Product Feature Analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from datetime import datetime

from src.utils.config import AppConfig, load_config, save_config
from src.utils.logging import setup_logging, get_logger
from src.utils.utils import set_random_seeds, create_directory_structure, save_results
from src.utils.compliance import ComplianceChecker, AuditLogger
from src.data.generator import SyntheticReviewGenerator, load_sample_data, validate_review_data
from src.nlp.feature_extraction import FeatureAnalyzer
from src.eval.evaluator import FeatureAnalysisEvaluator
from src.viz.visualizer import FeatureVisualizer


class ProductFeatureAnalysisApp:
    """Main application class for Product Feature Analysis."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the application.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logging
        setup_logging(
            log_level=self.config.log_level,
            log_file=self.config.log_file
        )
        self.logger = get_logger("ProductFeatureAnalysis")
        
        # Set random seeds
        set_random_seeds(self.config.model_config.random_seed)
        
        # Create directory structure
        create_directory_structure(".")
        
        # Initialize components
        self.feature_analyzer = FeatureAnalyzer({
            'feature_extractor': self.config.model_config.feature_extractor,
            'sentiment_model': self.config.model_config.sentiment_model
        })
        
        self.evaluator = FeatureAnalysisEvaluator()
        
        self.visualizer = FeatureVisualizer({
            'figure_size': self.config.viz_config.figure_size,
            'dpi': self.config.viz_config.dpi,
            'save_plots': self.config.viz_config.save_plots,
            'plot_path': self.config.viz_config.plot_path
        })
        
        # Compliance components
        self.compliance_checker = ComplianceChecker()
        self.audit_logger = AuditLogger()
        
        # Data storage
        self.reviews_df: Optional[pd.DataFrame] = None
        self.products_df: Optional[pd.DataFrame] = None
        self.features_df: Optional[pd.DataFrame] = None
        self.results: Optional[Dict[str, Any]] = None
        self.compliance_report: Optional[Dict[str, Any]] = None
        
        self.logger.info("Product Feature Analysis App initialized")
    
    def generate_sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate sample data for analysis.
        
        Returns:
            Tuple of (reviews_df, products_df)
        """
        self.logger.info("Generating sample data...")
        
        generator = SyntheticReviewGenerator(seed=self.config.model_config.random_seed)
        
        # Generate data
        reviews_df = generator.generate_dataset(
            n_reviews=self.config.data_config.n_reviews,
            n_products=self.config.data_config.n_products
        )
        
        products_df = generator.generate_product_catalog(
            n_products=self.config.data_config.n_products
        )
        
        # Validate data
        validate_review_data(reviews_df)
        
        self.reviews_df = reviews_df
        self.products_df = products_df
        
        self.logger.info(f"Generated {len(reviews_df)} reviews for {len(products_df)} products")
        
        return reviews_df, products_df
    
    def load_data(self, reviews_path: str, products_path: Optional[str] = None) -> None:
        """Load data from files.
        
        Args:
            reviews_path: Path to reviews CSV file
            products_path: Path to products CSV file (optional)
        """
        self.logger.info(f"Loading data from {reviews_path}")
        
        # Load reviews
        self.reviews_df = pd.read_csv(reviews_path)
        validate_review_data(self.reviews_df)
        
        # Load products if provided
        if products_path and Path(products_path).exists():
            self.products_df = pd.read_csv(products_path)
        
        self.logger.info(f"Loaded {len(self.reviews_df)} reviews")
    
    def analyze_features(self) -> pd.DataFrame:
        """Analyze features from reviews.
        
        Returns:
            DataFrame with extracted features
        """
        if self.reviews_df is None:
            raise ValueError("No reviews data loaded. Call generate_sample_data() or load_data() first.")
        
        self.logger.info("Analyzing features from reviews...")
        
        # Extract features
        self.features_df = self.feature_analyzer.analyze_reviews(self.reviews_df)
        
        # Aggregate features
        aggregated_features = self.feature_analyzer.aggregate_features(self.features_df)
        
        # Cluster features if requested
        if self.config.model_config.clustering_method:
            self.features_df = self.feature_analyzer.cluster_features(
                self.features_df, 
                n_clusters=self.config.model_config.n_clusters
            )
        
        self.logger.info(f"Extracted {len(self.features_df)} feature mentions")
        
        return self.features_df
    
    def evaluate_results(self) -> Dict[str, Any]:
        """Evaluate analysis results.
        
        Returns:
            Dictionary with evaluation results
        """
        if self.features_df is None:
            raise ValueError("No features analyzed. Call analyze_features() first.")
        
        self.logger.info("Evaluating analysis results...")
        
        # Evaluate model performance
        self.results = self.evaluator.evaluate_model_performance(
            self.features_df, 
            self.products_df
        )
        
        # Create leaderboard
        leaderboard = self.evaluator.create_leaderboard(self.results)
        self.results['leaderboard'] = leaderboard
        
        # Generate report
        report = self.evaluator.generate_report(self.results)
        self.results['report'] = report
        
        # Run compliance checks
        if self.config.disclaimer_required:
            self._run_compliance_checks()
        
        self.logger.info("Evaluation completed")
        
        return self.results
    
    def create_visualizations(self) -> Dict[str, Any]:
        """Create visualizations.
        
        Returns:
            Dictionary with visualization objects
        """
        if self.features_df is None:
            raise ValueError("No features analyzed. Call analyze_features() first.")
        
        self.logger.info("Creating visualizations...")
        
        visualizations = {}
        
        # Feature frequency plot
        visualizations['feature_frequency'] = self.visualizer.plot_feature_frequency(
            self.features_df
        )
        
        # Sentiment distribution plot
        visualizations['sentiment_distribution'] = self.visualizer.plot_sentiment_distribution(
            self.features_df
        )
        
        # Feature sentiment heatmap
        visualizations['sentiment_heatmap'] = self.visualizer.plot_feature_sentiment_heatmap(
            self.features_df
        )
        
        # Feature importance plot
        visualizations['feature_importance'] = self.visualizer.plot_feature_importance(
            self.features_df
        )
        
        # Interactive dashboard
        visualizations['dashboard'] = self.visualizer.plot_interactive_dashboard(
            self.features_df, 
            self.products_df
        )
        
        # Word cloud
        visualizations['word_cloud'] = self.visualizer.create_word_cloud(
            self.features_df
        )
        
        # Evaluation metrics plot
        if self.results:
            visualizations['evaluation_metrics'] = self.visualizer.plot_evaluation_metrics(
                self.results
            )
        
        self.logger.info("Visualizations created")
        
        return visualizations
    
    def save_results(self, output_dir: str = "assets/reports") -> None:
        """Save analysis results.
        
        Args:
            output_dir: Output directory for results
        """
        if self.results is None:
            raise ValueError("No results to save. Call evaluate_results() first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results as JSON
        results_file = output_path / f"analysis_results_{timestamp}.json"
        save_results(self.results, str(results_file), format="json")
        
        # Save features DataFrame
        if self.features_df is not None:
            features_file = output_path / f"features_{timestamp}.csv"
            self.features_df.to_csv(features_file, index=False)
        
        # Save leaderboard
        if 'leaderboard' in self.results:
            leaderboard_file = output_path / f"leaderboard_{timestamp}.csv"
            self.results['leaderboard'].to_csv(leaderboard_file, index=False)
        
        # Save report
        if 'report' in self.results:
            report_file = output_path / f"report_{timestamp}.txt"
            with open(report_file, 'w') as f:
                f.write(self.results['report'])
        
        self.logger.info(f"Results saved to {output_path}")
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete analysis pipeline.
        
        Returns:
            Dictionary with all results
        """
        self.logger.info("Starting full analysis pipeline...")
        
        # Generate or load data
        if self.reviews_df is None:
            self.generate_sample_data()
        
        # Analyze features
        self.analyze_features()
        
        # Evaluate results
        self.evaluate_results()
        
        # Create visualizations
        visualizations = self.create_visualizations()
        
        # Save results
        self.save_results()
        
        self.logger.info("Full analysis pipeline completed")
        
        return {
            'features': self.features_df,
            'results': self.results,
            'visualizations': visualizations
        }
    
    def get_top_insights(self, n: int = 10) -> Dict[str, Any]:
        """Get top insights from analysis.
        
        Args:
            n: Number of insights to return
            
        Returns:
            Dictionary with top insights
        """
        if self.results is None:
            raise ValueError("No results available. Run analysis first.")
        
        insights = {}
        
        # Top positive features
        if 'top_features' in self.results and 'top_positive' in self.results['top_features']:
            positive_features = list(self.results['top_features']['top_positive'].items())[:n]
            insights['top_positive_features'] = [
                {'feature': feature, 'sentiment': stats['avg_sentiment'], 'frequency': stats['frequency']}
                for feature, stats in positive_features
            ]
        
        # Top negative features
        if 'top_features' in self.results and 'top_negative' in self.results['top_features']:
            negative_features = list(self.results['top_features']['top_negative'].items())[:n]
            insights['top_negative_features'] = [
                {'feature': feature, 'sentiment': stats['avg_sentiment'], 'frequency': stats['frequency']}
                for feature, stats in negative_features
            ]
        
        # Business insights
        if 'business_metrics' in self.results:
            insights['business_insights'] = self.results['business_metrics']
        
        return insights
    
    def _run_compliance_checks(self) -> None:
        """Run compliance and bias checks."""
        if self.features_df is None:
            return
        
        self.logger.info("Running compliance checks...")
        
        # Get text data for privacy check
        text_data = self.reviews_df['review_text'].tolist() if self.reviews_df is not None else []
        
        # Generate compliance report
        self.compliance_report = self.compliance_checker.generate_compliance_report(
            self.features_df, text_data
        )
        
        # Log analysis run
        self.audit_logger.log_analysis_run(
            config=self.config.__dict__,
            results={
                'n_features': len(self.features_df),
                'n_reviews': len(self.reviews_df) if self.reviews_df is not None else 0,
                'avg_sentiment': self.features_df['sentiment'].mean()
            }
        )
        
        # Add compliance results to main results
        if self.results:
            self.results['compliance_report'] = self.compliance_report
        
        self.logger.info("Compliance checks completed")
    
    def get_compliance_report(self) -> Optional[Dict[str, Any]]:
        """Get compliance report.
        
        Returns:
            Compliance report if available
        """
        return self.compliance_report
    
    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get audit trail.
        
        Returns:
            List of audit entries
        """
        return self.audit_logger.get_audit_trail()


def main():
    """Main function to run the application."""
    # Initialize app
    app = ProductFeatureAnalysisApp()
    
    # Run full analysis
    results = app.run_full_analysis()
    
    # Print summary
    print("\n" + "="*60)
    print("PRODUCT FEATURE ANALYSIS - SUMMARY")
    print("="*60)
    
    if 'results' in results and 'report' in results['results']:
        print(results['results']['report'])
    
    # Get top insights
    insights = app.get_top_insights(5)
    
    print("\nTOP INSIGHTS:")
    print("-" * 30)
    
    if 'top_positive_features' in insights:
        print("Most Positive Features:")
        for insight in insights['top_positive_features']:
            print(f"  - {insight['feature']}: {insight['sentiment']:.3f}")
    
    if 'top_negative_features' in insights:
        print("\nMost Negative Features:")
        for insight in insights['top_negative_features']:
            print(f"  - {insight['feature']}: {insight['sentiment']:.3f}")
    
    print(f"\nAnalysis completed. Results saved to assets/reports/")


if __name__ == "__main__":
    main()
