"""Utility functions for Product Feature Analysis."""

import random
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import hashlib
import re
from datetime import datetime


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Set seeds for other libraries if available
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def anonymize_text(text: str, salt: str = "default_salt") -> str:
    """Anonymize text by hashing sensitive information.
    
    Args:
        text: Input text to anonymize
        salt: Salt for hashing
        
    Returns:
        Anonymized text
    """
    # Simple email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    def hash_email(match):
        email = match.group()
        hashed = hashlib.sha256((email + salt).encode()).hexdigest()[:8]
        return f"user_{hashed}"
    
    # Simple phone pattern
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    
    def hash_phone(match):
        phone = match.group()
        hashed = hashlib.sha256((phone + salt).encode()).hexdigest()[:8]
        return f"phone_{hashed}"
    
    # Apply anonymization
    text = re.sub(email_pattern, hash_email, text)
    text = re.sub(phone_pattern, hash_phone, text)
    
    return text


def clean_text(text: str) -> str:
    """Clean text for processing.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\-()]', '', text)
    
    # Strip whitespace
    text = text.strip()
    
    return text


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if valid, False otherwise
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True


def create_directory_structure(base_path: str) -> None:
    """Create standard directory structure.
    
    Args:
        base_path: Base directory path
    """
    directories = [
        "data/raw",
        "data/processed", 
        "data/external",
        "models",
        "logs",
        "assets/plots",
        "assets/reports",
        "checkpoints"
    ]
    
    base_path = Path(base_path)
    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)


def save_results(
    results: Dict[str, Any], 
    filepath: str, 
    format: str = "json"
) -> None:
    """Save results to file.
    
    Args:
        results: Results dictionary
        filepath: Output file path
        format: File format (json, pickle, csv)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    elif format == "pickle":
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    elif format == "csv" and isinstance(results, pd.DataFrame):
        results.to_csv(filepath, index=False)
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_results(filepath: str, format: str = "json") -> Dict[str, Any]:
    """Load results from file.
    
    Args:
        filepath: Input file path
        format: File format (json, pickle, csv)
        
    Returns:
        Loaded results
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if format == "json":
        import json
        with open(filepath, 'r') as f:
            return json.load(f)
    
    elif format == "pickle":
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    elif format == "csv":
        return pd.read_csv(filepath)
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def calculate_business_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    feature_names: List[str]
) -> Dict[str, float]:
    """Calculate business-relevant metrics.
    
    Args:
        predictions: Model predictions
        targets: True targets
        feature_names: List of feature names
        
    Returns:
        Dictionary of business metrics
    """
    metrics = {}
    
    # Feature coverage (how many features are identified)
    metrics["feature_coverage"] = len(feature_names) / 100  # Normalized
    
    # Sentiment consistency (how consistent are predictions)
    if len(predictions) > 1:
        metrics["sentiment_consistency"] = 1 - np.std(predictions)
    else:
        metrics["sentiment_consistency"] = 1.0
    
    # Actionability score (how actionable are the insights)
    metrics["actionability_score"] = min(1.0, len(feature_names) / 10)
    
    # Insight quality (combination of coverage and consistency)
    metrics["insight_quality"] = (
        metrics["feature_coverage"] * 0.4 + 
        metrics["sentiment_consistency"] * 0.6
    )
    
    return metrics


def format_timestamp() -> str:
    """Get formatted timestamp string.
    
    Returns:
        Formatted timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")
