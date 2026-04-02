#!/usr/bin/env python3
"""Script to run Product Feature Analysis from command line."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.main import ProductFeatureAnalysisApp
from src.utils.config import load_config


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Product Feature Analysis - Extract and analyze product features from customer reviews"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--reviews", 
        type=str, 
        help="Path to reviews CSV file"
    )
    
    parser.add_argument(
        "--products", 
        type=str, 
        help="Path to products CSV file"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="assets/reports",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--n-reviews", 
        type=int, 
        default=1000,
        help="Number of reviews to generate (if using synthetic data)"
    )
    
    parser.add_argument(
        "--n-products", 
        type=int, 
        default=10,
        help="Number of products to generate (if using synthetic data)"
    )
    
    parser.add_argument(
        "--feature-extractor", 
        type=str, 
        choices=["noun_phrases", "ner", "keywords"],
        default="noun_phrases",
        help="Feature extraction method"
    )
    
    parser.add_argument(
        "--sentiment-model", 
        type=str, 
        choices=["textblob", "spacy"],
        default="textblob",
        help="Sentiment analysis model"
    )
    
    parser.add_argument(
        "--generate-data", 
        action="store_true",
        help="Generate synthetic data instead of loading from files"
    )
    
    parser.add_argument(
        "--demo", 
        action="store_true",
        help="Run interactive demo"
    )
    
    args = parser.parse_args()
    
    # Initialize app
    app = ProductFeatureAnalysisApp(args.config)
    
    # Update config with command line arguments
    app.config.data_config.n_reviews = args.n_reviews
    app.config.data_config.n_products = args.n_products
    app.config.model_config.feature_extractor = args.feature_extractor
    app.config.model_config.sentiment_model = args.sentiment_model
    
    if args.demo:
        # Run Streamlit demo
        import subprocess
        demo_path = Path(__file__).parent / "demo" / "app.py"
        subprocess.run(["streamlit", "run", str(demo_path)])
        return
    
    try:
        # Load or generate data
        if args.generate_data:
            print("Generating synthetic data...")
            app.generate_sample_data()
        elif args.reviews:
            print(f"Loading data from {args.reviews}")
            app.load_data(args.reviews, args.products)
        else:
            print("No data source specified. Generating synthetic data...")
            app.generate_sample_data()
        
        # Run analysis
        print("Running feature analysis...")
        results = app.run_full_analysis()
        
        # Save results
        print(f"Saving results to {args.output}")
        app.save_results(args.output)
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
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
        
        print(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
