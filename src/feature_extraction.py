
"""
Mock feature extraction module to satisfy pickle dependencies.
"""
from sklearn.base import BaseEstimator, TransformerMixin
import re

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

# Aliases for likely class names
class FeatureExtractor(TextPreprocessor): pass
class TfidfExtractor(TextPreprocessor): pass
class TfidfWrapper(TextPreprocessor): pass
class Preprocessor(TextPreprocessor): pass
