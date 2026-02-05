
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import LabelBinarizer

def evaluate_model(y_true, y_pred, y_pred_proba=None, labels=None, model_name="Model"):
    """
    Evaluate model performance and return metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    roc_auc = None
    if y_pred_proba is not None and labels is not None:
        try:
            # Handle potential mismatch between integer y_true and string labels
            y_true_for_auc = y_true
            if isinstance(labels[0], str) and np.issubdtype(np.array(y_true).dtype, np.number):
                # Map integers to corresponding label strings
                y_true_for_auc = [labels[i] for i in y_true]

            lb = LabelBinarizer()
            lb.fit(labels)
            y_true_bin = lb.transform(y_true_for_auc)
            
            if y_true_bin.shape[1] == 1: # Binary case adjustment
                 y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
            
            # check if predict_proba shape matches label binarizer shape
            if y_pred_proba.shape[1] == y_true_bin.shape[1]:
                 roc_auc = roc_auc_score(y_true_bin, y_pred_proba, average='weighted', multi_class='ovr')
        except Exception as e:
            print(f"Warning: Could not calculate ROC AUC: {e}")

    print(f"\n{'='*60}")
    print(f"{model_name} - EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    if roc_auc:
        print(f"  ROC AUC:   {roc_auc:.4f}")

    print(f"\nDetailed Classification Report:")
    # If labels are strings and y_true/y_pred are ints, labels keyword acts as target_names
    # but we need to ensure y_true/y_pred match the indices implicitly.
    # classification_report(y_true, y_pred, target_names=labels) works if y values are 0..n_classes-1
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc if roc_auc else 0.0
    }

def plot_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix', save_path=None):
    """
    Plot and save confusion matrix.
    """
    # Auto-handle numeric vs string mismatch if possible without encoder
    # But sticking to what works in notebook (pre-inverse-transformed) is safer.
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels if labels else 'auto',
                yticklabels=labels if labels else 'auto')
    plt.title(title, fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"✓ Confusion matrix saved to {save_path}")
    plt.show()

def plot_model_comparison(results_df, metric='accuracy', title='Model Comparison', save_path=None):
    """
    Plot comparison of models key metric.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=results_df.index, y=metric, data=results_df, palette='viridis')
    plt.title(title, fontsize=14)
    plt.ylabel(metric.capitalize())
    plt.xlabel('Model')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(results_df[metric]):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
    plt.tight_layout()
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✓ Model comparison plot saved to {save_path}")
    plt.show()

def create_evaluation_report(eval_results, save_path=None):
    """
    Create a DataFrame from evaluation results dictionary and save to CSV.
    """
    df = pd.DataFrame(eval_results).T
    df.index.name = 'model_name'
    df = df.reset_index()
    # Add proper model name column for display if index is slug
    df['model_display_name'] = df['model_name'].apply(lambda x: x.replace('_', ' ').title())
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE EVALUATION REPORT")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"\n✓ Evaluation report saved to {save_path}")
        
    return df

def analyze_errors(y_true, y_pred, texts, labels=None, top_n=10):
    """
    Analyze misclassified examples.
    
    Args:
        y_true: True labels (indices or strings)
        y_pred: Predicted labels (indices or strings)
        texts: List/Array of text examples
        labels: List of label names (strings) corresponding to indices
        top_n: Number of examples to return
    """
    # Ensure inputs are lists/arrays
    X_test_df = pd.DataFrame({'text': texts})
    
    df = X_test_df.copy()
    
    # Map indices to labels if provided and inputs are numeric
    if labels is not None and np.issubdtype(np.array(y_true).dtype, np.number):
        df['true_label'] = [labels[i] for i in y_true]
        df['predicted_label'] = [labels[i] for i in y_pred]
    else:
        df['true_label'] = y_true
        df['predicted_label'] = y_pred
    
    errors = df[df['true_label'] != df['predicted_label']]
    print(f"Found {len(errors)} misclassified samples out of {len(y_true)}")
    print(f"Error rate: {len(errors)/len(y_true):.2%}")
    
    if len(errors) > 0:
        print(f"\nTop {top_n} Misclassified Examples:")
        print(errors.head(top_n).to_string())
        
    return errors
