
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
import sys

class TicketClassifier:
    """
    Wrapper for various classification models for support tickets.
    """
    def __init__(self, model_type='logistic_regression', **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = self._get_model(model_type, **kwargs)
        self.best_params_ = None

    def _get_model(self, model_type, **kwargs):
        if model_type == 'logistic_regression':
            # Default max_iter to avoid convergence warnings
            if 'max_iter' not in kwargs:
                kwargs['max_iter'] = 1000
            return LogisticRegression(**kwargs)
        elif model_type == 'naive_bayes':
            return MultinomialNB(**kwargs)
        elif model_type == 'random_forest':
            # Add random_state for reproducibility if not present
            if 'random_state' not in kwargs:
                kwargs['random_state'] = 42
            return RandomForestClassifier(**kwargs)
        elif model_type == 'svm':
            # Helper to enable probability estimates if needed (though SVC(probability=True) is slow)
            # Notebook 05 tries predict_proba, so we might want probability=True if feasible,
            # or handle the exception graciously as 05 seems to do ("try... except... y_pred_proba = None")
            if 'probability' not in kwargs and model_type == 'svm':
                 kwargs['probability'] = True
            if 'random_state' not in kwargs:
                kwargs['random_state'] = 42
            return SVC(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise NotImplementedError("This model does not support predict_proba")

    def tune_hyperparameters(self, X, y, cv=5):
        """
        Tune hyperparameters using GridSearchCV.
        """
        print(f"Tuning hyperparameters for {self.model_type}...")
        
        param_grid = {}
        if self.model_type == 'logistic_regression':
            param_grid = {'C': [0.1, 1, 10]}
        elif self.model_type == 'naive_bayes':
            param_grid = {'alpha': [0.1, 0.5, 1.0]}
        elif self.model_type == 'random_forest':
            param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
        elif self.model_type == 'svm':
            # Simplify grid for speed as SVM is slow
            param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
            
        if not param_grid:
            print("No parameters to tune for this model type.")
            return {}

        grid_search = GridSearchCV(self.model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        
        self.best_params_ = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        print(f"✓ Best parameters: {self.best_params_}")
        print(f"✓ Best CV score: {grid_search.best_score_:.4f}")
        
        return self.best_params_

    def save(self, filepath):
        joblib.dump(self, filepath)
        print(f"✓ Model saved to {filepath}")

    @staticmethod
    def load(filepath):
        return joblib.load(filepath)

def train_all_models(X_train, y_train, model_types=None):
    """
    Train and evaluate multiple models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_types: List of model types string identifiers
        
    Returns:
        tuple: (models_dict, results_dict, comparison_df)
    """
    if model_types is None:
        model_types = ['logistic_regression', 'naive_bayes', 'random_forest', 'svm']
        
    models = {}
    results = {}
    
    for model_name in model_types:
        print(f"\nTraining {model_name}...")
        
        # Initialize
        clf = TicketClassifier(model_type=model_name)
        
        # Cross-validation
        # Note: Depending on dataset size, this might take time.
        # Notebook 04 output shows CV scores.
        cv_scores = cross_val_score(clf.model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        
        # Train on full set
        clf.fit(X_train, y_train)
        
        # Training accuracy
        train_pred = clf.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        
        # Store results
        models[model_name] = clf
        results[model_name] = {
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'train_accuracy': train_acc
        }
        
        print(f"✓ Cross-validation accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        print(f"✓ Training accuracy: {train_acc:.4f}")
        
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results).T.sort_values('cv_mean', ascending=False)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(comparison_df)
    
    return models, results, comparison_df
