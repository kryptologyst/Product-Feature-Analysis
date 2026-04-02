"""Configuration management for Product Feature Analysis."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from omegaconf import OmegaConf
import yaml
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Data paths
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    
    # Dataset parameters
    n_reviews: int = 1000
    n_products: int = 10
    min_review_length: int = 10
    max_review_length: int = 500
    
    # Text preprocessing
    remove_stopwords: bool = True
    min_word_length: int = 2
    max_word_length: int = 20
    lemmatize: bool = True
    
    # Feature extraction
    min_feature_frequency: int = 3
    max_features: int = 1000
    ngram_range: tuple = (1, 3)


@dataclass
class ModelConfig:
    """Configuration for models."""
    
    # Sentiment analysis
    sentiment_model: str = "textblob"  # textblob, transformers, spacy
    sentiment_threshold: float = 0.1
    
    # Feature extraction
    feature_extractor: str = "noun_phrases"  # noun_phrases, ner, keywords
    use_aspect_sentiment: bool = True
    
    # Clustering
    clustering_method: str = "kmeans"  # kmeans, dbscan, hierarchical
    n_clusters: int = 5
    clustering_features: List[str] = None
    
    # Random seed
    random_seed: int = 42


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    
    # Metrics
    metrics: List[str] = None
    
    # Cross-validation
    cv_folds: int = 5
    test_size: float = 0.2
    
    # Business metrics
    business_metrics: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                "accuracy", "precision", "recall", "f1", 
                "auc_roc", "auc_pr", "calibration_error"
            ]
        
        if self.business_metrics is None:
            self.business_metrics = [
                "feature_coverage", "sentiment_consistency", 
                "actionability_score", "insight_quality"
            ]


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    
    # Plot settings
    figure_size: tuple = (12, 8)
    dpi: int = 300
    style: str = "seaborn-v0_8"
    
    # Colors
    color_palette: str = "viridis"
    sentiment_colors: Dict[str, str] = None
    
    # Output
    save_plots: bool = True
    plot_format: str = "png"
    plot_path: str = "assets/plots"
    
    def __post_init__(self):
        if self.sentiment_colors is None:
            self.sentiment_colors = {
                "positive": "#2E8B57",
                "negative": "#DC143C", 
                "neutral": "#808080"
            }


@dataclass
class AppConfig:
    """Main application configuration."""
    
    # Project info
    project_name: str = "Product Feature Analysis"
    version: str = "1.0.0"
    description: str = "Extract and analyze product features from customer reviews"
    
    # Paths
    data_config: DataConfig = None
    model_config: ModelConfig = None
    eval_config: EvaluationConfig = None
    viz_config: VisualizationConfig = None
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    
    # Compliance
    anonymize_data: bool = True
    remove_pii: bool = True
    disclaimer_required: bool = True
    
    def __post_init__(self):
        if self.data_config is None:
            self.data_config = DataConfig()
        if self.model_config is None:
            self.model_config = ModelConfig()
        if self.eval_config is None:
            self.eval_config = EvaluationConfig()
        if self.viz_config is None:
            self.viz_config = VisualizationConfig()


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from YAML file or use defaults."""
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return OmegaConf.structured(AppConfig(**config_dict))
    
    return AppConfig()


def save_config(config: AppConfig, config_path: str) -> None:
    """Save configuration to YAML file."""
    
    config_dict = OmegaConf.structured(config)
    OmegaConf.save(config_dict, config_path)


# Default configuration
DEFAULT_CONFIG = AppConfig()
