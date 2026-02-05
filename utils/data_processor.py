"""
Data processing utilities for ticket upload and classification.
"""

import pandas as pd
import numpy as np
import json
from io import StringIO
import re


def parse_uploaded_file(uploaded_file):
    """
    Parse uploaded file (CSV, JSON, or Excel).
    
    Parameters:
    -----------
    uploaded_file : UploadedFile
        Streamlit uploaded file object
        
    Returns:
    --------
    pd.DataFrame
        Parsed ticket data
    """
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'json':
            data = json.load(uploaded_file)
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        return df
    
    except Exception as e:
        raise ValueError(f"Error parsing file: {str(e)}")


def parse_pasted_data(text_data, format='csv'):
    """
    Parse pasted text data.
    
    Parameters:
    -----------
    text_data : str
        Pasted text data
    format : str
        Format of the data ('csv' or 'json')
        
    Returns:
    --------
    pd.DataFrame
        Parsed ticket data
    """
    try:
        if format == 'csv':
            df = pd.read_csv(StringIO(text_data))
        elif format == 'json':
            data = json.loads(text_data)
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return df
    
    except Exception as e:
        raise ValueError(f"Error parsing pasted data: {str(e)}")


def validate_ticket_data(df):
    """
    Validate that the DataFrame has required columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Ticket data
        
    Returns:
    --------
    tuple
        (is_valid, error_message)
    """
    if len(df) == 0:
        return False, "No data rows found in the file"
    
    # Check for description column (various possible names)
    description_columns = ['description', 'ticket_description', 'text', 'content', 'message', 'issue', 'ticket description', 'body', 'details', 'problem']
    
    # Convert all column names to lowercase for comparison
    df_columns_lower = [col.lower().strip() for col in df.columns]
    
    # Check if any description column exists
    has_description = any(desc_col.lower().strip() in df_columns_lower for desc_col in description_columns)
    
    if not has_description:
        # Provide helpful error message
        found_cols = ', '.join(df.columns.tolist())
        
        # Check if this looks like ticket data at all
        ticket_related_keywords = ['ticket', 'issue', 'problem', 'request', 'support', 'help', 'complaint']
        has_ticket_keyword = any(keyword in col.lower() for col in df.columns for keyword in ticket_related_keywords)
        
        if not has_ticket_keyword:
            return False, (
                f"❌ This doesn't appear to be ticket data.\n\n"
                f"**Found columns:** {found_cols}\n\n"
                f"**Expected:** A file with support ticket descriptions.\n\n"
                f"💡 **Tip:** Your file should have a column containing ticket text, such as:\n"
                f"   • `description` - ticket description\n"
                f"   • `text` - ticket content\n"
                f"   • `message` - customer message\n"
                f"   • `issue` - reported issue\n\n"
                f"**Example CSV format:**\n"
                f"```\n"
                f"description,title\n"
                f"Login page not loading,Login Issue\n"
                f"Request for dark mode,Feature Request\n"
                f"```"
            )
        else:
            return False, (
                f"❌ Missing description column.\n\n"
                f"**Found columns:** {found_cols}\n\n"
                f"**Expected one of:** {', '.join(description_columns)}\n\n"
                f"💡 **Tip:** Please rename your column containing ticket descriptions to `description`"
            )
    
    return True, None


def suggest_columns(df):
    """
    Suggest description and title columns based on content and names.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw ticket data
        
    Returns:
    --------
    dict
        {'description': col_name, 'title': col_name}
    """
    suggestions = {'description': None, 'title': None}
    
    # helper to check if col looks like text
    def is_text_col(col_name):
        if len(df) == 0: return False
        # Check first few non-null values
        sample = df[col_name].dropna().head(5).astype(str)
        if len(sample) == 0: return False
        # Check avg length (descriptions are usually longer)
        avg_len = sample.apply(len).mean()
        return avg_len > 10
    
    columns = df.columns.tolist()
    
    # 1. Look for known names for Description
    desc_keywords = ['description', 'ticket_description', 'text', 'content', 'message', 'issue', 'body', 'details', 'problem', 'complaint']
    for col in columns:
        if any(k in col.lower() for k in desc_keywords) and is_text_col(col):
            suggestions['description'] = col
            break
            
    # 2. If no name match, look for longest text column
    if not suggestions['description']:
        text_cols = [c for c in columns if is_text_col(c)]
        if text_cols:
            # Pick validation based on avg length
            lens = {c: df[c].dropna().head(10).astype(str).str.len().mean() for c in text_cols}
            suggestions['description'] = max(lens, key=lens.get)
            
    # 3. Look for known names for Title
    title_keywords = ['title', 'subject', 'ticket_subject', 'summary', 'header', 'topic']
    for col in columns:
        if any(k in col.lower() for k in title_keywords) and col != suggestions['description']:
            suggestions['title'] = col
            break
            
    return suggestions


def normalize_ticket_data(df, text_column=None, title_column=None):
    """
    Normalize column names and extract ticket descriptions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw ticket data
    text_column : str, optional
        Name of the column containing ticket text
    title_column : str, optional
        Name of the column containing ticket title
        
    Returns:
    --------
    pd.DataFrame
        Normalized ticket data with standard column names
    """
    df = df.copy()
    
    # Use specified text column or try to auto-detect
    if text_column and text_column in df.columns:
        df['description'] = df[text_column]
    else:
        # Fallback to existing logic if simple usage
        suggestions = suggest_columns(df)
        if suggestions['description']:
            df['description'] = df[suggestions['description']]
        
    # Use specified title column or try to auto-detect
    if title_column and title_column in df.columns:
        df['title'] = df[title_column]
    else:
        suggestions = suggest_columns(df)
        if suggestions['title']:
            df['title'] = df[suggestions['title']]
    
    # If still no title but we have description, generate it
    if 'description' in df.columns and 'title' not in df.columns:
        # Generate title from first 50 chars of description
        df['title'] = df['description'].astype(str).str[:50].str.strip() + '...'
    
    # Add ID if not present
    if 'id' not in df.columns and 'ticket_id' not in df.columns:
        df['id'] = [f"TKT-{i+1:04d}" for i in range(len(df))]
    elif 'ticket_id' in df.columns and 'id' not in df.columns:
        df['id'] = df['ticket_id']
    
    return df


def preprocess_text(text):
    """
    Basic text preprocessing.
    
    Parameters:
    -----------
    text : str
        Raw text
        
    Returns:
    --------
    str
        Preprocessed text
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def generate_sentiment(text):
    """
    Generate sentiment score based on keywords.
    
    Parameters:
    -----------
    text : str
        Ticket text
        
    Returns:
    --------
    str
        Sentiment label ('positive', 'neutral', 'negative')
    """
    # Simple keyword-based sentiment
    text_lower = str(text).lower()
    
    negative_keywords = ['error', 'bug', 'broken', 'issue', 'problem', 'fail', 'crash', 'not working', 'cannot', 'unable']
    positive_keywords = ['thank', 'great', 'excellent', 'good', 'appreciate', 'works', 'resolved']
    
    negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
    positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
    
    if negative_count > positive_count:
        return 'negative'
    elif positive_count > negative_count:
        return 'positive'
    else:
        return 'neutral'


def add_predictions_to_dataframe(df, predictions):
    """
    Add prediction results to DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original ticket data
    predictions : list
        List of prediction dictionaries
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with predictions added
    """
    df = df.copy()
    
    df['predicted_category'] = [p['category'] for p in predictions]
    df['category_confidence'] = [p['category_confidence'] for p in predictions]
    df['predicted_priority'] = [p['priority'] for p in predictions]
    df['priority_confidence'] = [p['priority_confidence'] for p in predictions]
    df['overall_confidence'] = [p['overall_confidence'] for p in predictions]
    
    # Add sentiment
    df['sentiment'] = df['description'].apply(generate_sentiment)
    
    return df


if __name__ == "__main__":
    # Test data processing
    print("Testing data processor...")
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'description': ['Login page not loading', 'Request for dark mode feature'],
        'title': ['Login issue', 'Feature request']
    })
    
    is_valid, error = validate_ticket_data(sample_data)
    print(f"Validation: {is_valid}, Error: {error}")
    
    normalized = normalize_ticket_data(sample_data)
    print(f"\nNormalized data:\n{normalized}")
