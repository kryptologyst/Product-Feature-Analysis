#!/usr/bin/env python3
"""Simple test script for Product Feature Analysis."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("Testing Product Feature Analysis...")
    
    try:
        # Test configuration
        from src.utils.config import AppConfig
        config = AppConfig()
        print("✓ Configuration loaded successfully")
        
        # Test utilities
        from src.utils.utils import set_random_seeds, clean_text
        set_random_seeds(42)
        cleaned = clean_text("This is a test!!!")
        assert cleaned == "This is a test"
        print("✓ Utilities working correctly")
        
        # Test data generation (without external deps)
        from src.data.generator import SyntheticReviewGenerator
        generator = SyntheticReviewGenerator(seed=42)
        reviews_df = generator.generate_dataset(n_reviews=10, n_products=2)
        assert len(reviews_df) == 10
        print("✓ Data generation working")
        
        # Test feature extraction (basic)
        from src.nlp.feature_extraction import TextPreprocessor
        preprocessor = TextPreprocessor()
        cleaned_text = preprocessor.clean_text("The camera quality is excellent!")
        assert "camera quality" in cleaned_text.lower()
        print("✓ Feature extraction working")
        
        # Test evaluation
        from src.eval.evaluator import FeatureAnalysisEvaluator
        evaluator = FeatureAnalysisEvaluator()
        print("✓ Evaluation module loaded")
        
        # Test visualization
        from src.viz.visualizer import FeatureVisualizer
        visualizer = FeatureVisualizer()
        print("✓ Visualization module loaded")
        
        print("\n🎉 All basic tests passed!")
        print("The Product Feature Analysis tool is ready to use.")
        print("\nTo run the full analysis:")
        print("  python scripts/run_analysis.py --generate-data")
        print("\nTo run the interactive demo:")
        print("  streamlit run demo/app.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
