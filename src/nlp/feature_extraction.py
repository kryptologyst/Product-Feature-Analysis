"""Natural Language Processing utilities for Product Feature Analysis."""

import re
import string
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np
from textblob import TextBlob
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from collections import Counter, defaultdict
import warnings

warnings.filterwarnings("ignore")


class TextPreprocessor:
    """Text preprocessing utilities."""
    
    def __init__(self, language: str = "en"):
        """Initialize preprocessor.
        
        Args:
            language: Language code
        """
        self.language = language
        self._setup_nltk()
    
    def _setup_nltk(self) -> None:
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
    
    def clean_text(self, text: str) -> str:
        """Clean text for processing.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation except basic ones
        text = re.sub(r'[^\w\s.,!?;:\-()]', '', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        blob = TextBlob(text)
        return blob.words
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered tokens
        """
        stop_words = set(nltk.corpus.stopwords.words('english'))
        return [token for token in tokens if token not in stop_words]


class FeatureExtractor:
    """Extract product features from reviews."""
    
    def __init__(self, method: str = "noun_phrases"):
        """Initialize feature extractor.
        
        Args:
            method: Extraction method (noun_phrases, ner, keywords)
        """
        self.method = method
        self.preprocessor = TextPreprocessor()
        
        # Initialize spaCy if using NER
        if method == "ner":
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
    
    def extract_noun_phrases(self, text: str) -> List[str]:
        """Extract noun phrases using TextBlob.
        
        Args:
            text: Input text
            
        Returns:
            List of noun phrases
        """
        blob = TextBlob(text)
        phrases = []
        
        for phrase in blob.noun_phrases:
            # Clean and filter phrases
            phrase = phrase.lower().strip()
            if len(phrase) > 2 and len(phrase.split()) <= 3:
                phrases.append(phrase)
        
        return phrases
    
    def extract_ner_features(self, text: str) -> List[str]:
        """Extract features using Named Entity Recognition.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted features
        """
        if self.nlp is None:
            return []
        
        doc = self.nlp(text)
        features = []
        
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "ORG", "PERSON"]:
                features.append(ent.text.lower())
        
        return features
    
    def extract_keywords(self, texts: List[str], max_features: int = 100) -> List[str]:
        """Extract keywords using TF-IDF.
        
        Args:
            texts: List of texts
            max_features: Maximum number of features
            
        Returns:
            List of keywords
        """
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top features
        scores = tfidf_matrix.sum(axis=0).A1
        top_indices = np.argsort(scores)[-max_features:]
        
        return [feature_names[i] for i in top_indices]
    
    def extract_features(self, text: str) -> List[str]:
        """Extract features from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted features
        """
        if self.method == "noun_phrases":
            return self.extract_noun_phrases(text)
        elif self.method == "ner":
            return self.extract_ner_features(text)
        else:
            return []


class SentimentAnalyzer:
    """Analyze sentiment of text and features."""
    
    def __init__(self, method: str = "textblob"):
        """Initialize sentiment analyzer.
        
        Args:
            method: Analysis method (textblob, transformers, spacy)
        """
        self.method = method
        
        if method == "spacy":
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("spaCy model not found. Falling back to TextBlob.")
                self.method = "textblob"
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text.
        
        Args:
            text: Input text
            
        Returns:
            Sentiment score (-1 to 1)
        """
        if self.method == "textblob":
            blob = TextBlob(text)
            return blob.sentiment.polarity
        
        elif self.method == "spacy" and hasattr(self, 'nlp'):
            doc = self.nlp(text)
            # Simple sentiment based on positive/negative words
            positive_words = ['good', 'great', 'excellent', 'amazing', 'fantastic', 'love', 'best']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'disappointed', 'poor']
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count + neg_count == 0:
                return 0.0
            
            return (pos_count - neg_count) / (pos_count + neg_count)
        
        else:
            return 0.0
    
    def analyze_aspect_sentiment(self, text: str, features: List[str]) -> Dict[str, float]:
        """Analyze sentiment for specific features.
        
        Args:
            text: Input text
            features: List of features to analyze
            
        Returns:
            Dictionary mapping features to sentiment scores
        """
        aspect_sentiments = {}
        
        for feature in features:
            # Find sentences containing the feature
            sentences = text.split('.')
            feature_sentences = [s for s in sentences if feature.lower() in s.lower()]
            
            if feature_sentences:
                # Analyze sentiment of sentences containing the feature
                sentiments = [self.analyze_sentiment(s) for s in feature_sentences]
                aspect_sentiments[feature] = np.mean(sentiments)
            else:
                aspect_sentiments[feature] = 0.0
        
        return aspect_sentiments


class FeatureAnalyzer:
    """Main feature analysis class."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize feature analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.feature_extractor = FeatureExtractor(
            method=self.config.get("feature_extractor", "noun_phrases")
        )
        self.sentiment_analyzer = SentimentAnalyzer(
            method=self.config.get("sentiment_model", "textblob")
        )
        self.preprocessor = TextPreprocessor()
    
    def analyze_reviews(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze reviews and extract features with sentiment.
        
        Args:
            reviews_df: DataFrame with reviews
            
        Returns:
            DataFrame with extracted features and sentiment
        """
        results = []
        
        for _, row in reviews_df.iterrows():
            review_text = row['review_text']
            
            # Clean text
            clean_text = self.preprocessor.clean_text(review_text)
            
            # Extract features
            features = self.feature_extractor.extract_features(clean_text)
            
            # Analyze sentiment for each feature
            aspect_sentiments = self.sentiment_analyzer.analyze_aspect_sentiment(
                clean_text, features
            )
            
            # Overall sentiment
            overall_sentiment = self.sentiment_analyzer.analyze_sentiment(clean_text)
            
            # Store results
            for feature in features:
                results.append({
                    'review_id': row['review_id'],
                    'product_id': row['product_id'],
                    'product_category': row.get('product_category', 'unknown'),
                    'feature': feature,
                    'sentiment': aspect_sentiments.get(feature, overall_sentiment),
                    'overall_sentiment': overall_sentiment,
                    'rating': row.get('rating', 0),
                    'review_length': len(clean_text.split())
                })
        
        return pd.DataFrame(results)
    
    def aggregate_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate features across reviews.
        
        Args:
            features_df: DataFrame with extracted features
            
        Returns:
            Aggregated feature analysis
        """
        aggregation = features_df.groupby('feature').agg({
            'sentiment': ['count', 'mean', 'std'],
            'overall_sentiment': 'mean',
            'rating': 'mean',
            'product_category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'
        }).round(3)
        
        # Flatten column names
        aggregation.columns = [
            'frequency', 'avg_sentiment', 'sentiment_std',
            'overall_sentiment', 'avg_rating', 'top_category'
        ]
        
        # Calculate additional metrics
        aggregation['sentiment_consistency'] = 1 - aggregation['sentiment_std'].fillna(0)
        aggregation['importance_score'] = (
            aggregation['frequency'] * 0.4 + 
            abs(aggregation['avg_sentiment']) * 0.3 +
            aggregation['sentiment_consistency'] * 0.3
        )
        
        # Sort by importance
        aggregation = aggregation.sort_values('importance_score', ascending=False)
        
        return aggregation.reset_index()
    
    def cluster_features(self, features_df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        """Cluster similar features.
        
        Args:
            features_df: DataFrame with features
            n_clusters: Number of clusters
            
        Returns:
            DataFrame with cluster assignments
        """
        # Create feature vectors using TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        feature_matrix = vectorizer.fit_transform(features_df['feature'])
        
        # Cluster features
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(feature_matrix)
        
        features_df['cluster'] = clusters
        
        return features_df
