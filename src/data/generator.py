"""Data generation and processing for Product Feature Analysis."""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import random
import re

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("Warning: TextBlob not available. Install with: pip install textblob")

try:
    from faker import Faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False
    print("Warning: Faker not available. Install with: pip install faker")


@dataclass
class ProductFeature:
    """Represents a product feature with sentiment."""
    
    name: str
    sentiment: float  # -1 to 1
    frequency: float  # 0 to 1
    importance: float  # 0 to 1


class SyntheticReviewGenerator:
    """Generate synthetic product reviews for analysis."""
    
    def __init__(self, seed: int = 42):
        """Initialize the generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        if FAKER_AVAILABLE:
            self.fake = Faker()
            Faker.seed(seed)
        else:
            self.fake = None
        random.seed(seed)
        np.random.seed(seed)
        
        # Define product categories and their features
        self.product_categories = {
            "smartphone": [
                ProductFeature("camera quality", 0.2, 0.8, 0.9),
                ProductFeature("battery life", -0.3, 0.9, 0.8),
                ProductFeature("screen resolution", 0.6, 0.6, 0.7),
                ProductFeature("performance", 0.4, 0.7, 0.8),
                ProductFeature("design", 0.5, 0.5, 0.6),
                ProductFeature("storage", -0.1, 0.4, 0.5),
                ProductFeature("audio quality", 0.3, 0.3, 0.4),
                ProductFeature("software", 0.1, 0.6, 0.7),
                ProductFeature("price", -0.2, 0.8, 0.9),
                ProductFeature("durability", 0.0, 0.3, 0.4),
            ],
            "laptop": [
                ProductFeature("performance", 0.5, 0.8, 0.9),
                ProductFeature("battery life", 0.1, 0.7, 0.8),
                ProductFeature("display quality", 0.4, 0.6, 0.7),
                ProductFeature("keyboard", 0.2, 0.5, 0.6),
                ProductFeature("portability", 0.3, 0.4, 0.5),
                ProductFeature("build quality", 0.4, 0.5, 0.6),
                ProductFeature("price", -0.1, 0.6, 0.7),
                ProductFeature("storage", 0.0, 0.4, 0.5),
                ProductFeature("connectivity", 0.1, 0.3, 0.4),
                ProductFeature("cooling", -0.2, 0.3, 0.4),
            ],
            "headphones": [
                ProductFeature("sound quality", 0.6, 0.9, 0.9),
                ProductFeature("comfort", 0.3, 0.7, 0.8),
                ProductFeature("noise cancellation", 0.4, 0.6, 0.7),
                ProductFeature("battery life", 0.2, 0.5, 0.6),
                ProductFeature("build quality", 0.3, 0.5, 0.6),
                ProductFeature("price", -0.1, 0.6, 0.7),
                ProductFeature("connectivity", 0.1, 0.4, 0.5),
                ProductFeature("design", 0.2, 0.4, 0.5),
                ProductFeature("portability", 0.1, 0.3, 0.4),
                ProductFeature("durability", 0.0, 0.3, 0.4),
            ]
        }
        
        # Review templates
        self.positive_templates = [
            "I love the {feature}. It's {adjective}.",
            "The {feature} is {adjective}. Highly recommend!",
            "Great {feature}. Very {adjective}.",
            "The {feature} exceeded my expectations. It's {adjective}.",
            "Amazing {feature}. So {adjective}!",
        ]
        
        self.negative_templates = [
            "The {feature} is {adjective}. Very disappointed.",
            "I hate the {feature}. It's {adjective}.",
            "Poor {feature}. Too {adjective}.",
            "The {feature} is {adjective}. Not worth it.",
            "Terrible {feature}. So {adjective}!",
        ]
        
        self.neutral_templates = [
            "The {feature} is {adjective}. It's okay.",
            "The {feature} is {adjective}. Nothing special.",
            "The {feature} is {adjective}. Average.",
            "The {feature} is {adjective}. It works.",
            "The {feature} is {adjective}. Decent.",
        ]
        
        # Adjectives for different sentiments
        self.adjectives = {
            "positive": ["excellent", "amazing", "fantastic", "outstanding", "brilliant", 
                        "superb", "wonderful", "perfect", "incredible", "awesome"],
            "negative": ["terrible", "awful", "horrible", "disappointing", "poor", 
                        "bad", "worst", "useless", "broken", "defective"],
            "neutral": ["okay", "average", "decent", "fine", "acceptable", 
                       "mediocre", "standard", "normal", "typical", "ordinary"]
        }
    
    def generate_review(self, product_category: str, features: List[ProductFeature]) -> str:
        """Generate a single review.
        
        Args:
            product_category: Category of the product
            features: List of features to mention
            
        Returns:
            Generated review text
        """
        review_parts = []
        
        # Select 2-4 features to mention
        n_features = random.randint(2, min(4, len(features)))
        selected_features = random.sample(features, n_features)
        
        for feature in selected_features:
            # Determine sentiment category
            if feature.sentiment > 0.2:
                sentiment_category = "positive"
                templates = self.positive_templates
            elif feature.sentiment < -0.2:
                sentiment_category = "negative"
                templates = self.negative_templates
            else:
                sentiment_category = "neutral"
                templates = self.neutral_templates
            
            # Select template and adjective
            template = random.choice(templates)
            adjective = random.choice(self.adjectives[sentiment_category])
            
            # Generate sentence
            sentence = template.format(feature=feature.name, adjective=adjective)
            review_parts.append(sentence)
        
        # Add some variation
        if random.random() < 0.3:
            review_parts.append("Overall, I'm satisfied with this product.")
        elif random.random() < 0.3:
            review_parts.append("Would recommend to others.")
        
        return " ".join(review_parts)
    
    def generate_dataset(
        self, 
        n_reviews: int = 1000,
        n_products: int = 10,
        categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate a complete dataset of reviews.
        
        Args:
            n_reviews: Number of reviews to generate
            n_products: Number of products
            categories: Product categories to use
            
        Returns:
            DataFrame with reviews and metadata
        """
        if categories is None:
            categories = list(self.product_categories.keys())
        
        reviews = []
        
        for i in range(n_reviews):
            # Select random category
            category = random.choice(categories)
            features = self.product_categories[category]
            
            # Generate review
            review_text = self.generate_review(category, features)
            
            # Add some noise to make it more realistic
            if random.random() < 0.1 and self.fake is not None:
                review_text += f" {self.fake.sentence()}"
            
            # Create review record
            review = {
                "review_id": f"review_{i:06d}",
                "product_id": f"product_{random.randint(1, n_products):03d}",
                "product_category": category,
                "review_text": review_text,
                "review_length": len(review_text.split()),
                "timestamp": self.fake.date_time_between(start_date="-1y", end_date="now") if self.fake else "2023-01-01",
                "customer_id": f"customer_{random.randint(1, 500):06d}",
                "rating": random.randint(1, 5),
            }
            
            reviews.append(review)
        
        return pd.DataFrame(reviews)
    
    def generate_product_catalog(self, n_products: int = 10) -> pd.DataFrame:
        """Generate product catalog.
        
        Args:
            n_products: Number of products to generate
            
        Returns:
            DataFrame with product information
        """
        products = []
        categories = list(self.product_categories.keys())
        
        for i in range(n_products):
            category = random.choice(categories)
            
            product = {
                "product_id": f"product_{i+1:03d}",
                "product_name": self.fake.catch_phrase() if self.fake else f"Product {i+1}",
                "category": category,
                "price": round(random.uniform(50, 2000), 2),
                "brand": self.fake.company() if self.fake else f"Brand {i+1}",
                "release_date": self.fake.date_between(start_date="-2y", end_date="now") if self.fake else "2023-01-01",
                "description": self.fake.text(max_nb_chars=200) if self.fake else f"Description for product {i+1}",
            }
            
            products.append(product)
        
        return pd.DataFrame(products)


def load_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load sample data for demonstration.
    
    Returns:
        Tuple of (reviews_df, products_df)
    """
    generator = SyntheticReviewGenerator()
    
    # Generate sample data
    reviews_df = generator.generate_dataset(n_reviews=500, n_products=5)
    products_df = generator.generate_product_catalog(n_products=5)
    
    return reviews_df, products_df


def validate_review_data(df: pd.DataFrame) -> bool:
    """Validate review dataset.
    
    Args:
        df: Review DataFrame
        
    Returns:
        True if valid
    """
    required_columns = [
        "review_id", "product_id", "review_text", 
        "rating", "customer_id", "timestamp"
    ]
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Check for empty reviews
    if df["review_text"].isna().any():
        raise ValueError("Found empty review texts")
    
    # Check rating range
    if not df["rating"].between(1, 5).all():
        raise ValueError("Ratings must be between 1 and 5")
    
    return True
