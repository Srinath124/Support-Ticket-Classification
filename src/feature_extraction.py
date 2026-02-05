"""
Feature extraction module.
"""
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import re
import pandas as pd
import numpy as np

def extract_text_features(df, text_col):
    """
    Extracts text features from a dataframe column.
    Returns a DataFrame with features:
    - char_count
    - word_count
    - avg_word_length
    - uppercase_ratio
    - exclamation_count
    - question_count
    """
    # Initialize features dataframe
    features = pd.DataFrame(index=df.index)
    
    # Ensure text column is string
    texts = df[text_col].astype(str)
    
    # Calculate simple features
    features['char_count'] = texts.str.len()
    features['word_count'] = texts.apply(lambda x: len(x.split()))
    
    # Avoid division by zero
    features['avg_word_length'] = features['char_count'] / features['word_count']
    features['avg_word_length'] = features['avg_word_length'].fillna(0)
    
    # Uppercase ratio
    def get_uppercase_ratio(text):
        if len(text) == 0:
            return 0.0
        return sum(1 for c in text if c.isupper()) / len(text)
    
    features['uppercase_ratio'] = texts.apply(get_uppercase_ratio)
    
    # Punctuation counts
    features['exclamation_count'] = texts.str.count('!')
    features['question_count'] = texts.str.count('\?')
    
    return features

def encode_labels(df, column=None):
    """
    Encodes labels using LabelEncoder.
    Can accept (df, column) or just a series/list.
    Returns (encoded_values, encoder).
    """
    le = LabelEncoder()
    if column is not None and isinstance(df, pd.DataFrame):
        encoded = le.fit_transform(df[column])
        return encoded, le
    else:
        # Assume input is array-like
        encoded = le.fit_transform(df)
        return encoded, le

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return [self._preprocess(text) for text in X]
        
    def _preprocess(self, text):
        if not isinstance(text, str):
            return str(text)
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

# Restore original class hierarchy for pickles
class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Wrapper around TfidfVectorizer/CountVectorizer to match pickled object structure.
    The pickled object contains a 'vectorizer' attribute.
    """
    def __init__(self, method='tfidf', max_features=None, ngram_range=(1, 1)):
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        
        # Initialize vectorizer based on method
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        elif method == 'bow':
            self.vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)

    def fit(self, X, y=None):
        if self.vectorizer:
            self.vectorizer.fit(X, y)
        return self

    def transform(self, X):
        if self.vectorizer:
            return self.vectorizer.transform(X)
        raise ValueError("FeatureExtractor: Internal vectorizer is missing or not fitted!")
    
    def fit_transform(self, X, y=None):
        if self.vectorizer:
            return self.vectorizer.fit_transform(X, y)
        return X

    def get_feature_names(self):
        if self.vectorizer:
            if hasattr(self.vectorizer, 'get_feature_names_out'):
                return self.vectorizer.get_feature_names_out()
            elif hasattr(self.vectorizer, 'get_feature_names'):
                return self.vectorizer.get_feature_names()
        return []

class TfidfExtractor(FeatureExtractor): pass
class TfidfWrapper(FeatureExtractor): pass
class Preprocessor(TextPreprocessor): pass
