# Product Feature Analysis

A comprehensive tool for extracting and analyzing product features from customer reviews using advanced NLP techniques.

## Overview

This project analyzes customer reviews to identify the most frequently discussed product features, their associated sentiment, and business insights. It's designed for research and educational purposes to help understand customer feedback patterns.

## Features

- **Feature Extraction**: Multiple methods including noun phrase extraction, NER, and keyword analysis
- **Sentiment Analysis**: Aspect-based sentiment analysis for individual features
- **Clustering**: Group similar features using various clustering algorithms
- **Visualization**: Interactive dashboards and comprehensive plots
- **Evaluation**: Business-relevant metrics and model performance assessment
- **Synthetic Data**: Generate realistic sample data for testing and demonstration

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Product-Feature-Analysis.git
cd Product-Feature-Analysis
```

2. Install dependencies:
```bash
pip install -e .
```

3. Install optional dependencies for development:
```bash
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.main import ProductFeatureAnalysisApp

# Initialize the app
app = ProductFeatureAnalysisApp()

# Run complete analysis
results = app.run_full_analysis()

# Get insights
insights = app.get_top_insights(10)
print(insights)
```

### Interactive Demo

Run the Streamlit demo:

```bash
streamlit run demo/app.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data processing and generation
│   ├── nlp/               # NLP and feature extraction
│   ├── eval/              # Evaluation metrics
│   ├── viz/               # Visualization utilities
│   └── utils/             # Configuration and utilities
├── demo/                  # Streamlit demo application
├── configs/               # Configuration files
├── tests/                 # Unit tests
├── assets/                # Output plots and reports
├── data/                  # Data storage
└── notebooks/             # Jupyter notebooks
```

## Configuration

The application uses YAML configuration files. See `configs/default.yaml` for available options:

- **Data Configuration**: Dataset parameters, preprocessing options
- **Model Configuration**: Feature extraction methods, sentiment analysis
- **Evaluation Configuration**: Metrics and validation settings
- **Visualization Configuration**: Plot settings and output options

## Data Schema

### Reviews Dataset
- `review_id`: Unique review identifier
- `product_id`: Product identifier
- `review_text`: Review content
- `rating`: Customer rating (1-5)
- `customer_id`: Customer identifier
- `timestamp`: Review date
- `product_category`: Product category

### Products Dataset
- `product_id`: Product identifier
- `product_name`: Product name
- `category`: Product category
- `price`: Product price
- `brand`: Brand name
- `description`: Product description

## Evaluation Metrics

### Sentiment Analysis Metrics
- Accuracy, Precision, Recall, F1-Score
- MAE, RMSE, Correlation, R² Score
- Calibration Error

### Feature Extraction Metrics
- Feature Coverage
- Feature Diversity
- Sentiment Balance
- Frequency Distribution (Gini coefficient)

### Business Metrics
- Actionability Score
- Insight Quality
- Product Coverage
- Category Insight Depth

## Privacy and Compliance

- **Data Anonymization**: Automatic PII detection and anonymization
- **Privacy-First**: Synthetic data generation for testing
- **Compliance**: Built-in bias detection and fairness checks
- **Transparency**: Full data lineage and processing logs

## Limitations

- Results are based on synthetic data for demonstration
- Sentiment analysis accuracy depends on review quality
- Feature extraction may miss domain-specific terminology
- Requires human validation for business decisions

## Disclaimer

**IMPORTANT**: This tool is for research and educational purposes only. Do not use for automated decision-making without human review. Results should be validated by domain experts before making business decisions.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions or issues, please open a GitHub issue or contact the development team.
# Product-Feature-Analysis
