"""Visualization utilities for Product Feature Analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")


class FeatureVisualizer:
    """Create visualizations for feature analysis."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or {}
        self.figure_size = self.config.get('figure_size', (12, 8))
        self.dpi = self.config.get('dpi', 300)
        self.save_plots = self.config.get('save_plots', True)
        self.plot_path = Path(self.config.get('plot_path', 'assets/plots'))
        
        # Create output directory
        if self.save_plots:
            self.plot_path.mkdir(parents=True, exist_ok=True)
    
    def plot_feature_frequency(
        self, 
        features_df: pd.DataFrame, 
        top_n: int = 20,
        title: str = "Top Product Features by Frequency"
    ) -> plt.Figure:
        """Plot feature frequency distribution.
        
        Args:
            features_df: DataFrame with features
            top_n: Number of top features to show
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Aggregate features
        feature_counts = features_df['feature'].value_counts().head(top_n)
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(feature_counts)), feature_counts.values)
        ax.set_yticks(range(len(feature_counts)))
        ax.set_yticklabels(feature_counts.index)
        ax.set_xlabel('Frequency')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Color bars by frequency
        colors = plt.cm.viridis(np.linspace(0, 1, len(feature_counts)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_path / 'feature_frequency.png', dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_sentiment_distribution(
        self, 
        features_df: pd.DataFrame,
        title: str = "Feature Sentiment Distribution"
    ) -> plt.Figure:
        """Plot sentiment distribution.
        
        Args:
            features_df: DataFrame with features
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram of sentiment scores
        ax1.hist(features_df['sentiment'], bins=30, alpha=0.7, color='steelblue')
        ax1.set_xlabel('Sentiment Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Sentiment Score Distribution')
        ax1.axvline(0, color='red', linestyle='--', alpha=0.7, label='Neutral')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Categorical sentiment distribution
        sentiment_categories = features_df['sentiment'].apply(
            lambda x: 'Positive' if x > 0.2 else 'Negative' if x < -0.2 else 'Neutral'
        )
        
        sentiment_counts = sentiment_categories.value_counts()
        colors = ['#2E8B57', '#DC143C', '#808080']  # Green, Red, Gray
        
        ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, 
                autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Sentiment Category Distribution')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_path / 'sentiment_distribution.png', dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_feature_sentiment_heatmap(
        self, 
        features_df: pd.DataFrame,
        top_n: int = 15,
        title: str = "Feature Sentiment Heatmap"
    ) -> plt.Figure:
        """Plot feature sentiment heatmap.
        
        Args:
            features_df: DataFrame with features
            top_n: Number of top features to show
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Get top features by frequency
        top_features = features_df['feature'].value_counts().head(top_n).index
        
        # Create pivot table
        pivot_data = features_df[features_df['feature'].isin(top_features)].pivot_table(
            values='sentiment', 
            index='feature', 
            columns='product_category', 
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Create heatmap
        sns.heatmap(
            pivot_data, 
            annot=True, 
            cmap='RdBu_r', 
            center=0,
            fmt='.2f',
            ax=ax
        )
        
        ax.set_title(title)
        ax.set_xlabel('Product Category')
        ax.set_ylabel('Feature')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_path / 'feature_sentiment_heatmap.png', dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(
        self, 
        features_df: pd.DataFrame,
        top_n: int = 15,
        title: str = "Feature Importance Analysis"
    ) -> plt.Figure:
        """Plot feature importance analysis.
        
        Args:
            features_df: DataFrame with features
            top_n: Number of top features to show
            title: Plot title
            
        Returns:
            Matplotlib figure
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
        
        top_features = feature_stats.sort_values('importance', ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Create scatter plot
        scatter = ax.scatter(
            top_features['frequency'],
            top_features['avg_sentiment'],
            s=top_features['importance'] * 1000,
            c=top_features['avg_rating'],
            cmap='viridis',
            alpha=0.7
        )
        
        # Add feature labels
        for idx, row in top_features.iterrows():
            ax.annotate(idx, (row['frequency'], row['avg_sentiment']), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Average Sentiment')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Average Rating')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_path / 'feature_importance.png', dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_interactive_dashboard(
        self, 
        features_df: pd.DataFrame,
        products_df: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """Create interactive dashboard.
        
        Args:
            features_df: DataFrame with features
            products_df: DataFrame with product information
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Frequency', 'Sentiment Distribution', 
                          'Feature Sentiment Heatmap', 'Feature Importance'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # Feature frequency
        feature_counts = features_df['feature'].value_counts().head(10)
        fig.add_trace(
            go.Bar(x=feature_counts.values, y=feature_counts.index, 
                   orientation='h', name='Frequency'),
            row=1, col=1
        )
        
        # Sentiment distribution
        sentiment_categories = features_df['sentiment'].apply(
            lambda x: 'Positive' if x > 0.2 else 'Negative' if x < -0.2 else 'Neutral'
        )
        sentiment_counts = sentiment_categories.value_counts()
        
        fig.add_trace(
            go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values,
                   name='Sentiment'),
            row=1, col=2
        )
        
        # Feature sentiment heatmap
        top_features = features_df['feature'].value_counts().head(10).index
        pivot_data = features_df[features_df['feature'].isin(top_features)].pivot_table(
            values='sentiment', 
            index='feature', 
            columns='product_category', 
            aggfunc='mean'
        )
        
        fig.add_trace(
            go.Heatmap(z=pivot_data.values, 
                      x=pivot_data.columns, 
                      y=pivot_data.index,
                      colorscale='RdBu',
                      name='Sentiment Heatmap'),
            row=2, col=1
        )
        
        # Feature importance scatter
        feature_stats = features_df.groupby('feature').agg({
            'sentiment': ['count', 'mean'],
            'rating': 'mean'
        })
        feature_stats.columns = ['frequency', 'avg_sentiment', 'avg_rating']
        feature_stats['importance'] = (
            feature_stats['frequency'] * 0.4 +
            abs(feature_stats['avg_sentiment']) * 0.6
        )
        
        fig.add_trace(
            go.Scatter(
                x=feature_stats['frequency'],
                y=feature_stats['avg_sentiment'],
                mode='markers+text',
                text=feature_stats.index,
                textposition='top center',
                marker=dict(
                    size=feature_stats['importance'] * 20,
                    color=feature_stats['avg_rating'],
                    colorscale='viridis',
                    showscale=True
                ),
                name='Importance'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Product Feature Analysis Dashboard",
            showlegend=False,
            height=800
        )
        
        return fig
    
    def plot_evaluation_metrics(
        self, 
        results: Dict[str, Any],
        title: str = "Model Evaluation Metrics"
    ) -> plt.Figure:
        """Plot evaluation metrics.
        
        Args:
            results: Evaluation results
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        metrics_data = []
        
        # Collect metrics
        if 'sentiment_metrics' in results:
            for metric, value in results['sentiment_metrics'].items():
                metrics_data.append({'Category': 'Sentiment', 'Metric': metric, 'Value': value})
        
        if 'feature_metrics' in results:
            for metric, value in results['feature_metrics'].items():
                metrics_data.append({'Category': 'Feature', 'Metric': metric, 'Value': value})
        
        if 'business_metrics' in results:
            for metric, value in results['business_metrics'].items():
                metrics_data.append({'Category': 'Business', 'Metric': metric, 'Value': value})
        
        metrics_df = pd.DataFrame(metrics_data)
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Create grouped bar plot
        categories = metrics_df['Category'].unique()
        x = np.arange(len(categories))
        width = 0.8 / len(metrics_df['Metric'].unique())
        
        for i, metric in enumerate(metrics_df['Metric'].unique()):
            metric_values = metrics_df[metrics_df['Metric'] == metric]['Value'].values
            ax.bar(x + i * width, metric_values, width, label=metric)
        
        ax.set_xlabel('Category')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.set_xticks(x + width * (len(metrics_df['Metric'].unique()) - 1) / 2)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_path / 'evaluation_metrics.png', dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def create_word_cloud(
        self, 
        features_df: pd.DataFrame,
        title: str = "Feature Word Cloud"
    ) -> plt.Figure:
        """Create word cloud of features.
        
        Args:
            features_df: DataFrame with features
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        try:
            from wordcloud import WordCloud
        except ImportError:
            print("WordCloud not available. Install with: pip install wordcloud")
            return None
        
        # Prepare text data
        feature_text = ' '.join(features_df['feature'].astype(str))
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(feature_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title)
        
        if self.save_plots:
            plt.savefig(self.plot_path / 'word_cloud.png', dpi=self.dpi, bbox_inches='tight')
        
        return fig
