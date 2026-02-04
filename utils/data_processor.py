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


def normalize_ticket_data(df):
    """
    Normalize column names and extract ticket descriptions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw ticket data
        
    Returns:
    --------
    pd.DataFrame
        Normalized ticket data with standard column names
    """
    df = df.copy()
    
    # Find description column
    description_columns = ['description', 'ticket_description', 'text', 'content', 'message', 'issue', 'ticket description']
    description_col = None
    
    for col in df.columns:
        if col.lower().strip() in [dc.lower().strip() for dc in description_columns]:
            description_col = col
            break
    
    if description_col and description_col != 'description':
        df['description'] = df[description_col]
    
    # Find title/subject column if exists
    title_columns = ['title', 'subject', 'ticket_subject', 'summary', 'ticket subject']
    title_col = None
    
    for col in df.columns:
        if col.lower().strip() in [tc.lower().strip() for tc in title_columns]:
            title_col = col
            break
    
    if title_col and title_col != 'title':
        df['title'] = df[title_col]
    elif 'description' in df.columns and 'title' not in df.columns:
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
