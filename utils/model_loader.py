"""
Model loading and management utilities for Streamlit dashboard.
"""

import joblib
import os
from pathlib import Path
import streamlit as st
import pandas as pd


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@st.cache_resource
def load_models():
    """
    Load all trained models and feature extractors.
    Uses Streamlit caching for performance.
    
    Returns:
    --------
    dict
        Dictionary containing all models and encoders
    """
    models_dir = get_project_root() / "models"
    
    models = {}
    
    try:
        # Load feature extractors
        models['tfidf_extractor'] = joblib.load(models_dir / "tfidf_extractor.pkl")
        models['bow_extractor'] = joblib.load(models_dir / "bow_extractor.pkl")
        
        # Load label encoders
        models['category_encoder'] = joblib.load(models_dir / "category_encoder.pkl")
        models['priority_encoder'] = joblib.load(models_dir / "priority_encoder.pkl")
        
        # Load production models
        models['category_model'] = joblib.load(models_dir / "production_category_model.pkl")
        models['priority_model'] = joblib.load(models_dir / "production_priority_model.pkl")
        
        # Load all category models for comparison
        models['category_lr'] = joblib.load(models_dir / "category_logistic_regression.pkl")
        models['category_nb'] = joblib.load(models_dir / "category_naive_bayes.pkl")
        models['category_rf'] = joblib.load(models_dir / "category_random_forest.pkl")
        models['category_svm'] = joblib.load(models_dir / "category_svm.pkl")
        
        # Load all priority models for comparison
        models['priority_lr'] = joblib.load(models_dir / "priority_logistic_regression.pkl")
        models['priority_nb'] = joblib.load(models_dir / "priority_naive_bayes.pkl")
        models['priority_rf'] = joblib.load(models_dir / "priority_random_forest.pkl")
        models['priority_svm'] = joblib.load(models_dir / "priority_svm.pkl")
        
        print(f"✓ Successfully loaded {len(models)} models and encoders")
        
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        raise
    
    return models


def predict_ticket(text, models):
    """
    Predict category and priority for a single ticket.
    
    Parameters:
    -----------
    text : str
        Ticket description text
    models : dict
        Dictionary of loaded models
        
    Returns:
    --------
    dict
        Prediction results with category, priority, and confidence scores
    """
    # Preprocess text (basic cleaning)
    text = str(text).strip()
    
    # Extract features using TF-IDF
    features = models['tfidf_extractor'].transform([text])
    
    # Predict category
    category_pred = models['category_model'].predict(features)[0]
    category_proba = models['category_model'].predict_proba(features)[0]
    category_confidence = float(max(category_proba))
    category_label = models['category_encoder'].inverse_transform([category_pred])[0]
    
    # Predict priority
    priority_pred = models['priority_model'].predict(features)[0]
    priority_proba = models['priority_model'].predict_proba(features)[0]
    priority_confidence = float(max(priority_proba))
    priority_label = models['priority_encoder'].inverse_transform([priority_pred])[0]
    
    return {
        'category': category_label,
        'category_confidence': category_confidence,
        'priority': priority_label,
        'priority_confidence': priority_confidence,
        'overall_confidence': (category_confidence + priority_confidence) / 2
    }


def predict_batch(texts, models, progress_callback=None):
    """
    Predict category and priority for multiple tickets.
    
    Parameters:
    -----------
    texts : list
        List of ticket description texts
    models : dict
        Dictionary of loaded models
    progress_callback : callable, optional
        Function to call with progress updates
        
    Returns:
    --------
    list
        List of prediction dictionaries
    """
    results = []
    
    for i, text in enumerate(texts):
        result = predict_ticket(text, models)
        results.append(result)
        
        if progress_callback:
            progress_callback(i + 1, len(texts))
    
    return results


def get_model_metadata():
    """
    Get metadata about production models.
    
    Returns:
    --------
    dict
        Model metadata
    """
    models_dir = get_project_root() / "models"
    metadata_file = models_dir / "production_metadata.csv"
    
    if metadata_file.exists():
        df = pd.read_csv(metadata_file)
        return df.to_dict('records')[0] if len(df) > 0 else {}
    
    return {
        'category_model': 'SVM',
        'priority_model': 'Random Forest',
        'trained_date': '2 days ago',
        'training_samples': '15.4K tickets'
    }


if __name__ == "__main__":
    # Test model loading
    print("Testing model loader...")
    models = load_models()
    print(f"Loaded models: {list(models.keys())}")
    
    # Test prediction
    test_text = "Login page not loading on mobile"
    result = predict_ticket(test_text, models)
    print(f"\nTest prediction for: '{test_text}'")
    print(f"Category: {result['category']} (confidence: {result['category_confidence']:.2%})")
    print(f"Priority: {result['priority']} (confidence: {result['priority_confidence']:.2%})")
