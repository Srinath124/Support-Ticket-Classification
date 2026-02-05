
import sys
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_loader import load_models

def inspect_models():
    with open('debug_output.txt', 'w') as f:
        f.write("Loading models...\n")
        try:
            models = load_models()
        except Exception as e:
            f.write(f"Failed to load models: {e}\n")
            return

        f.write("\nModel Inspection:\n")
        f.write("-" * 50 + "\n")
        
        # 1. Inspect TFIDF
        tfidf = models.get('tfidf_extractor')
        f.write(f"TFIDF Extractor Type: {type(tfidf)}\n")
        test_text = "Login page error"
        
        if hasattr(tfidf, 'transform'):
            f.write(f"Attributes: {dir(tfidf)}\n")
            f.write(f"Has vocabulary_: {hasattr(tfidf, 'vocabulary_')}\n")
            
            try:
                res = tfidf.transform([test_text])
                f.write(f"TFIDF transform output type: {type(res)}\n")
            except Exception as e:
                f.write(f"TFIDF transform FAILED: {e}\n")
        else:
            f.write("TFIDF does not have transform method!\n")

        # 1b. Inspect BOW
        bow = models.get('bow_extractor')
        f.write(f"\nBOW Extractor Type: {type(bow)}\n")
        if hasattr(bow, 'transform'):
             try:
                res = bow.transform([test_text])
                f.write(f"BOW transform output type: {type(res)}\n")
                f.write(f"BOW Shape: {getattr(res, 'shape', 'N/A')}\n")
             except Exception as e:
                f.write(f"BOW transform FAILED: {e}\n")
        else:
            f.write("TFIDF does not have transform method!\n")

        # 2. Inspect Category Model
        cat_model = models.get('category_model')
        f.write(f"\nCategory Model Type: {type(cat_model)}\n")
        f.write(f"Category Model Steps (if Pipeline): {getattr(cat_model, 'steps', 'Not a Pipeline')}\n")
        
        # Test Prediction
        if tfidf and cat_model:
            f.write("\nTesting Prediction Flow:\n")
            try:
                # Replicate code in predict_ticket
                features = tfidf.transform([test_text])
                f.write("Features extracted.\n")
                pred = cat_model.predict(features)
                f.write(f"Prediction successful: {pred}\n")
            except Exception as e:
                f.write(f"Prediction FAILED with features input: {e}\n")
                
            f.write("\nTesting Direct Prediction (as if Pipeline):\n")
            try:
                pred = cat_model.predict([test_text])
                f.write(f"Direct Prediction successful: {pred}\n")
            except Exception as e:
                f.write(f"Direct Prediction FAILED: {e}\n")

if __name__ == "__main__":
    inspect_models()
