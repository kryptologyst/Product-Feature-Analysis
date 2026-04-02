"""Tests for Product Feature Analysis."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.generator import SyntheticReviewGenerator, validate_review_data
from src.nlp.feature_extraction import FeatureExtractor, SentimentAnalyzer, FeatureAnalyzer
from src.eval.evaluator import FeatureAnalysisEvaluator
from src.utils.utils import set_random_seeds, anonymize_text, clean_text
from src.main import ProductFeatureAnalysisApp


class TestSyntheticReviewGenerator:
    """Test synthetic review generator."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = SyntheticReviewGenerator(seed=42)
        assert generator.fake is not None
        assert len(generator.product_categories) > 0
    
    def test_generate_review(self):
        """Test review generation."""
        generator = SyntheticReviewGenerator(seed=42)
        features = generator.product_categories["smartphone"][:3]
        review = generator.generate_review("smartphone", features)
        
        assert isinstance(review, str)
        assert len(review) > 0
    
    def test_generate_dataset(self):
        """Test dataset generation."""
        generator = SyntheticReviewGenerator(seed=42)
        reviews_df = generator.generate_dataset(n_reviews=10, n_products=2)
        
        assert len(reviews_df) == 10
        assert 'review_text' in reviews_df.columns
        assert 'product_id' in reviews_df.columns
    
    def test_generate_product_catalog(self):
        """Test product catalog generation."""
        generator = SyntheticReviewGenerator(seed=42)
        products_df = generator.generate_product_catalog(n_products=5)
        
        assert len(products_df) == 5
        assert 'product_id' in products_df.columns
        assert 'product_name' in products_df.columns


class TestDataValidation:
    """Test data validation functions."""
    
    def test_validate_review_data_valid(self):
        """Test validation with valid data."""
        reviews_df = pd.DataFrame({
            'review_id': ['r1', 'r2'],
            'product_id': ['p1', 'p2'],
            'review_text': ['Great product!', 'Not bad.'],
            'rating': [5, 3],
            'customer_id': ['c1', 'c2'],
            'timestamp': ['2023-01-01', '2023-01-02']
        })
        
        assert validate_review_data(reviews_df) is True
    
    def test_validate_review_data_missing_column(self):
        """Test validation with missing column."""
        reviews_df = pd.DataFrame({
            'review_id': ['r1'],
            'review_text': ['Great product!']
        })
        
        with pytest.raises(ValueError):
            validate_review_data(reviews_df)
    
    def test_validate_review_data_invalid_rating(self):
        """Test validation with invalid rating."""
        reviews_df = pd.DataFrame({
            'review_id': ['r1'],
            'product_id': ['p1'],
            'review_text': ['Great product!'],
            'rating': [6],  # Invalid rating
            'customer_id': ['c1'],
            'timestamp': ['2023-01-01']
        })
        
        with pytest.raises(ValueError):
            validate_review_data(reviews_df)


class TestFeatureExtraction:
    """Test feature extraction functionality."""
    
    def test_text_preprocessor(self):
        """Test text preprocessing."""
        from src.nlp.feature_extraction import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        
        # Test clean_text
        dirty_text = "This is a TEST!!!   With extra spaces."
        clean = preprocessor.clean_text(dirty_text)
        assert clean == "this is a test with extra spaces"
        
        # Test tokenize
        tokens = preprocessor.tokenize("Hello world")
        assert len(tokens) == 2
    
    def test_feature_extractor(self):
        """Test feature extractor."""
        extractor = FeatureExtractor(method="noun_phrases")
        
        text = "The camera quality is excellent and battery life is good."
        features = extractor.extract_features(text)
        
        assert isinstance(features, list)
        assert len(features) > 0
    
    def test_sentiment_analyzer(self):
        """Test sentiment analyzer."""
        analyzer = SentimentAnalyzer(method="textblob")
        
        positive_text = "This is great!"
        negative_text = "This is terrible!"
        
        pos_sentiment = analyzer.analyze_sentiment(positive_text)
        neg_sentiment = analyzer.analyze_sentiment(negative_text)
        
        assert pos_sentiment > 0
        assert neg_sentiment < 0
    
    def test_feature_analyzer(self):
        """Test main feature analyzer."""
        analyzer = FeatureAnalyzer()
        
        # Create sample reviews
        reviews_df = pd.DataFrame({
            'review_id': ['r1', 'r2'],
            'product_id': ['p1', 'p1'],
            'review_text': ['Great camera quality!', 'Battery life is poor.'],
            'rating': [5, 2],
            'product_category': ['smartphone', 'smartphone']
        })
        
        features_df = analyzer.analyze_reviews(reviews_df)
        
        assert len(features_df) > 0
        assert 'feature' in features_df.columns
        assert 'sentiment' in features_df.columns


class TestEvaluation:
    """Test evaluation functionality."""
    
    def test_feature_analysis_evaluator(self):
        """Test feature analysis evaluator."""
        evaluator = FeatureAnalysisEvaluator()
        
        # Create sample data
        features_df = pd.DataFrame({
            'review_id': ['r1', 'r2', 'r3'],
            'feature': ['camera', 'battery', 'camera'],
            'sentiment': [0.8, -0.6, 0.7],
            'overall_sentiment': [0.8, -0.6, 0.7],
            'rating': [5, 2, 4]
        })
        
        # Test sentiment metrics
        metrics = evaluator.calculate_sentiment_metrics(
            features_df['sentiment'].values,
            features_df['overall_sentiment'].values
        )
        
        assert 'accuracy' in metrics
        assert 'mae' in metrics
        assert 'correlation' in metrics
        
        # Test feature metrics
        feature_metrics = evaluator.calculate_feature_metrics(features_df)
        
        assert 'feature_coverage' in feature_metrics
        assert 'feature_diversity' in feature_metrics


class TestUtilities:
    """Test utility functions."""
    
    def test_set_random_seeds(self):
        """Test random seed setting."""
        set_random_seeds(42)
        
        # Test numpy
        np.random.seed(42)
        val1 = np.random.random()
        
        set_random_seeds(42)
        val2 = np.random.random()
        
        assert val1 == val2
    
    def test_anonymize_text(self):
        """Test text anonymization."""
        text = "Contact me at john@example.com or call 555-1234"
        anonymized = anonymize_text(text)
        
        assert "john@example.com" not in anonymized
        assert "555-1234" not in anonymized
        assert "user_" in anonymized
        assert "phone_" in anonymized
    
    def test_clean_text(self):
        """Test text cleaning."""
        dirty_text = "This is a TEST!!!   With extra spaces."
        clean = clean_text(dirty_text)
        
        assert clean == "This is a TEST With extra spaces"


class TestMainApp:
    """Test main application."""
    
    def test_app_initialization(self):
        """Test app initialization."""
        app = ProductFeatureAnalysisApp()
        
        assert app.config is not None
        assert app.feature_analyzer is not None
        assert app.evaluator is not None
        assert app.visualizer is not None
    
    def test_generate_sample_data(self):
        """Test sample data generation."""
        app = ProductFeatureAnalysisApp()
        reviews_df, products_df = app.generate_sample_data()
        
        assert len(reviews_df) > 0
        assert len(products_df) > 0
        assert 'review_text' in reviews_df.columns
        assert 'product_id' in products_df.columns
    
    @patch('src.main.ProductFeatureAnalysisApp.analyze_features')
    @patch('src.main.ProductFeatureAnalysisApp.evaluate_results')
    def test_run_full_analysis(self, mock_evaluate, mock_analyze):
        """Test full analysis pipeline."""
        app = ProductFeatureAnalysisApp()
        
        # Mock the methods
        mock_analyze.return_value = pd.DataFrame({
            'feature': ['camera', 'battery'],
            'sentiment': [0.8, -0.6]
        })
        mock_evaluate.return_value = {'test': 'results'}
        
        # Generate sample data
        app.generate_sample_data()
        
        # Run analysis
        results = app.run_full_analysis()
        
        assert 'features' in results
        assert 'results' in results
        mock_analyze.assert_called_once()
        mock_evaluate.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
