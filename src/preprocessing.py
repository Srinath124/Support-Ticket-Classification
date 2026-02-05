import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sys

def download_nltk_data():
    """Download necessary NLTK data."""
    resources = ['stopwords', 'punkt', 'wordnet', 'omw-1.4']
    for r in resources:
        try:
            nltk.data.find(f'corpora/{r}')
        except LookupError:
            try:
                nltk.data.find(f'tokenizers/{r}')
            except LookupError:
                nltk.download(r, quiet=True)
    print("✓ NLTK data downloaded")

def clean_text(text):
    """
    Clean and preprocess text.
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text) or text == '':
        return ''
        
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove special characters and digits (keep spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_dataframe(df, text_column='Ticket Description', remove_stopwords=True, lemmatize=True):
    """
    Apply preprocessing to the entire dataframe.
    
    Args:
        df: Input DataFrame
        text_column: Name of the text column to clean
        remove_stopwords: Whether to remove stopwords
        lemmatize: Whether to lemmatize tokens
        
    Returns:
        DataFrame with new column '{text_column}_cleaned'
    """
    df = df.copy()
    
    # Check if text column exists
    if text_column not in df.columns:
        # Try to find a suitable column if not found (fallback)
        potential_cols = [c for c in df.columns if 'description' in c.lower() or 'text' in c.lower()]
        if potential_cols:
            text_column = potential_cols[0]
            print(f"Warning: {text_column} not found. Using {text_column} instead.")
        else:
             raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    # Basic cleaning
    df[f'{text_column}_cleaned'] = df[text_column].apply(clean_text)
    
    # Setup NLTK tools
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
    
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        
    def advanced_processing(text):
        if not text:
            return ""
        tokens = text.split()
        
        if remove_stopwords:
            tokens = [t for t in tokens if t not in stop_words]
            
        if lemmatize:
            tokens = [lemmatizer.lemmatize(t) for t in tokens]
            
        return ' '.join(tokens)

    # Apply advanced processing if needed
    if remove_stopwords or lemmatize:
        df[f'{text_column}_cleaned'] = df[f'{text_column}_cleaned'].apply(advanced_processing)
        
    return df

def create_priority_labels(df, category_column=None, urgency_keywords=None):
    """
    Create priority labels based on urgency keywords in text and category rules.
    
    Args:
        df: DataFrame containing ticket data.
        category_column: Name of the column containing ticket category.
        urgency_keywords: List of keywords indicating high priority.
        
    Returns:
        DataFrame with a new 'priority' column.
    """
    df = df.copy()
    
    if urgency_keywords is None:
        urgency_keywords = ['urgent', 'critical', 'emergency', 'asap', 'immediate', 'crash', 'security']

    # Identify text column for keyword search (using the cleaned one if available)
    text_cols = [c for c in df.columns if '_cleaned' in c]
    if text_cols:
        search_col = text_cols[0]
    else:
        # Fallback to description or similar
        potential_cols = [c for c in df.columns if 'description' in c.lower()]
        search_col = potential_cols[0] if potential_cols else None

    def get_priority(row):
        # Default priority
        priority = 'Low'
        
        # 1. Check keywords in text
        if search_col and pd.notna(row[search_col]):
            text = str(row[search_col]).lower()
            if any(keyword in text for keyword in urgency_keywords):
                return 'Critical'
        
        # 2. Check category rules
        if category_column and category_column in row and pd.notna(row[category_column]):
            cat = str(row[category_column]).lower()
            if 'critical' in cat or 'security' in cat or 'outage' in cat:
                return 'High'
            elif 'billing' in cat or 'account' in cat:
                return 'Medium'
                
        return priority

    df['priority'] = df.apply(get_priority, axis=1)
    
    # Ensure there's a mapping to numerical label if needed later, but the notebook seems to plot 'High', 'Low' etc.
    # We can add 'priority_label' map just in case it's expected by later steps not yet visible.
    priority_map = {'Critical': 3, 'High': 2, 'Medium': 1, 'Low': 0}
    df['priority_label'] = df['priority'].map(priority_map)
    
    return df
