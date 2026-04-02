"""Streamlit demo for Product Feature Analysis."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.main import ProductFeatureAnalysisApp
from src.data.generator import SyntheticReviewGenerator
from src.utils.config import AppConfig


def main():
    """Main Streamlit app."""
    
    # Page configuration
    st.set_page_config(
        page_title="Product Feature Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and disclaimer
    st.title("📊 Product Feature Analysis")
    st.markdown("**Extract and analyze product features from customer reviews**")
    
    # Important disclaimer
    st.warning("""
    ⚠️ **IMPORTANT DISCLAIMER**: This is a research and educational tool. 
    Do not use for automated decision-making without human review. 
    Results should be validated by domain experts before making business decisions.
    """)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Data generation parameters
    st.sidebar.subheader("Data Parameters")
    n_reviews = st.sidebar.slider("Number of Reviews", 100, 2000, 500)
    n_products = st.sidebar.slider("Number of Products", 3, 20, 5)
    
    # Analysis parameters
    st.sidebar.subheader("Analysis Parameters")
    feature_extractor = st.sidebar.selectbox(
        "Feature Extractor", 
        ["noun_phrases", "ner", "keywords"],
        index=0
    )
    
    sentiment_model = st.sidebar.selectbox(
        "Sentiment Model",
        ["textblob", "spacy"],
        index=0
    )
    
    n_clusters = st.sidebar.slider("Number of Clusters", 3, 10, 5)
    
    # Initialize app
    if 'app' not in st.session_state:
        config = AppConfig()
        config.data_config.n_reviews = n_reviews
        config.data_config.n_products = n_products
        config.model_config.feature_extractor = feature_extractor
        config.model_config.sentiment_model = sentiment_model
        config.model_config.n_clusters = n_clusters
        
        st.session_state.app = ProductFeatureAnalysisApp()
        st.session_state.config = config
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "🔍 Feature Analysis", 
        "📊 Visualizations", 
        "📋 Evaluation", 
        "💡 Insights"
    ])
    
    with tab1:
        st.header("Analysis Overview")
        
        if st.button("🔄 Run Analysis", type="primary"):
            with st.spinner("Running analysis..."):
                # Update config
                st.session_state.config.data_config.n_reviews = n_reviews
                st.session_state.config.data_config.n_products = n_products
                st.session_state.config.model_config.feature_extractor = feature_extractor
                st.session_state.config.model_config.sentiment_model = sentiment_model
                st.session_state.config.model_config.n_clusters = n_clusters
                
                # Run analysis
                results = st.session_state.app.run_full_analysis()
                st.session_state.results = results
                
                st.success("Analysis completed successfully!")
        
        if 'results' in st.session_state:
            results = st.session_state.results
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Reviews", 
                    len(results['features']['review_id'].unique())
                )
            
            with col2:
                st.metric(
                    "Unique Features", 
                    results['features']['feature'].nunique()
                )
            
            with col3:
                avg_sentiment = results['features']['sentiment'].mean()
                st.metric(
                    "Avg Sentiment", 
                    f"{avg_sentiment:.3f}"
                )
            
            with col4:
                st.metric(
                    "Products Analyzed", 
                    results['features']['product_id'].nunique()
                )
    
    with tab2:
        st.header("Feature Analysis")
        
        if 'results' in st.session_state:
            features_df = st.session_state.results['features']
            
            # Feature frequency table
            st.subheader("Top Features by Frequency")
            feature_counts = features_df['feature'].value_counts().head(20)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(
                    feature_counts.reset_index().rename(columns={'index': 'Feature', 'feature': 'Count'}),
                    use_container_width=True
                )
            
            with col2:
                # Feature sentiment summary
                sentiment_summary = features_df.groupby('feature').agg({
                    'sentiment': ['count', 'mean', 'std']
                }).round(3)
                sentiment_summary.columns = ['Frequency', 'Avg_Sentiment', 'Sentiment_Std']
                sentiment_summary = sentiment_summary.sort_values('Frequency', ascending=False).head(10)
                
                st.dataframe(sentiment_summary, use_container_width=True)
            
            # Feature details
            st.subheader("Feature Details")
            selected_feature = st.selectbox(
                "Select Feature to Analyze",
                features_df['feature'].unique()
            )
            
            if selected_feature:
                feature_data = features_df[features_df['feature'] == selected_feature]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Sentiment Distribution for '{selected_feature}'**")
                    fig = px.histogram(
                        feature_data, 
                        x='sentiment', 
                        nbins=20,
                        title=f"Sentiment Distribution: {selected_feature}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write(f"**Reviews mentioning '{selected_feature}'**")
                    sample_reviews = feature_data[['review_id', 'sentiment', 'rating']].head(10)
                    st.dataframe(sample_reviews, use_container_width=True)
    
    with tab3:
        st.header("Visualizations")
        
        if 'results' in st.session_state:
            features_df = st.session_state.results['features']
            
            # Interactive dashboard
            st.subheader("Interactive Dashboard")
            
            # Feature frequency plot
            feature_counts = features_df['feature'].value_counts().head(15)
            fig1 = px.bar(
                x=feature_counts.values,
                y=feature_counts.index,
                orientation='h',
                title="Top Features by Frequency",
                labels={'x': 'Frequency', 'y': 'Feature'}
            )
            fig1.update_layout(height=500)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Sentiment distribution
            col1, col2 = st.columns(2)
            
            with col1:
                sentiment_categories = features_df['sentiment'].apply(
                    lambda x: 'Positive' if x > 0.2 else 'Negative' if x < -0.2 else 'Neutral'
                )
                sentiment_counts = sentiment_categories.value_counts()
                
                fig2 = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution"
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                fig3 = px.histogram(
                    features_df,
                    x='sentiment',
                    nbins=30,
                    title="Sentiment Score Distribution"
                )
                st.plotly_chart(fig3, use_container_width=True)
            
            # Feature importance scatter plot
            st.subheader("Feature Importance Analysis")
            
            feature_stats = features_df.groupby('feature').agg({
                'sentiment': ['count', 'mean'],
                'rating': 'mean'
            })
            feature_stats.columns = ['frequency', 'avg_sentiment', 'avg_rating']
            feature_stats['importance'] = (
                feature_stats['frequency'] * 0.4 +
                abs(feature_stats['avg_sentiment']) * 0.6
            )
            
            fig4 = px.scatter(
                feature_stats,
                x='frequency',
                y='avg_sentiment',
                size='importance',
                color='avg_rating',
                hover_data=['frequency', 'avg_sentiment', 'avg_rating'],
                title="Feature Importance Scatter Plot",
                labels={'frequency': 'Frequency', 'avg_sentiment': 'Average Sentiment'}
            )
            st.plotly_chart(fig4, use_container_width=True)
    
    with tab4:
        st.header("Evaluation Metrics")
        
        if 'results' in st.session_state and 'results' in st.session_state.results:
            eval_results = st.session_state.results['results']
            
            # Metrics tables
            col1, col2 = st.columns(2)
            
            with col1:
                if 'sentiment_metrics' in eval_results:
                    st.subheader("Sentiment Analysis Metrics")
                    sentiment_df = pd.DataFrame([
                        {'Metric': k, 'Value': v} 
                        for k, v in eval_results['sentiment_metrics'].items()
                    ])
                    st.dataframe(sentiment_df, use_container_width=True)
            
            with col2:
                if 'feature_metrics' in eval_results:
                    st.subheader("Feature Extraction Metrics")
                    feature_df = pd.DataFrame([
                        {'Metric': k, 'Value': v} 
                        for k, v in eval_results['feature_metrics'].items()
                    ])
                    st.dataframe(feature_df, use_container_width=True)
            
            # Business metrics
            if 'business_metrics' in eval_results:
                st.subheader("Business Impact Metrics")
                business_df = pd.DataFrame([
                    {'Metric': k, 'Value': v} 
                    for k, v in eval_results['business_metrics'].items()
                ])
                st.dataframe(business_df, use_container_width=True)
            
            # Leaderboard
            if 'leaderboard' in eval_results:
                st.subheader("Model Performance Leaderboard")
                st.dataframe(eval_results['leaderboard'], use_container_width=True)
    
    with tab5:
        st.header("Business Insights")
        
        if 'results' in st.session_state:
            insights = st.session_state.app.get_top_insights(10)
            
            # Top positive features
            if 'top_positive_features' in insights:
                st.subheader("🌟 Top Positive Features")
                positive_df = pd.DataFrame(insights['top_positive_features'])
                st.dataframe(positive_df, use_container_width=True)
            
            # Top negative features
            if 'top_negative_features' in insights:
                st.subheader("⚠️ Top Negative Features")
                negative_df = pd.DataFrame(insights['top_negative_features'])
                st.dataframe(negative_df, use_container_width=True)
            
            # Business insights
            if 'business_insights' in insights:
                st.subheader("📈 Business Impact")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Actionability Score",
                        f"{insights['business_insights'].get('actionability_score', 0):.3f}"
                    )
                
                with col2:
                    st.metric(
                        "Insight Quality",
                        f"{insights['business_insights'].get('insight_quality', 0):.3f}"
                    )
                
                with col3:
                    st.metric(
                        "Feature Coverage",
                        f"{insights['business_insights'].get('feature_coverage', 0):.3f}"
                    )
            
            # Recommendations
            st.subheader("💡 Recommendations")
            
            if 'top_positive_features' in insights and 'top_negative_features' in insights:
                st.write("**Strengths to Leverage:**")
                for feature in insights['top_positive_features'][:3]:
                    st.write(f"• Emphasize {feature['feature']} in marketing (sentiment: {feature['sentiment']:.3f})")
                
                st.write("**Areas for Improvement:**")
                for feature in insights['top_negative_features'][:3]:
                    st.write(f"• Address {feature['feature']} issues (sentiment: {feature['sentiment']:.3f})")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        Product Feature Analysis Tool | Research & Educational Use Only
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
