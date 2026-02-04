"""
Model Performance page - Model metrics and comparison.
"""

import streamlit as st
import pandas as pd
from utils import (
    get_model_metrics,
    get_model_status,
    get_model_versions,
    create_metric_bars,
    COLORS,
)
from utils.mock_data import get_model_comparison


def show():
    """Display the model performance page."""
    
    # Header
    st.title("Model Performance")
    st.markdown("**Track and compare model metrics**")
    st.markdown("---")
    
    # Model Status Overview
    col1, col2, col3 = st.columns(3)
    
    model_status = get_model_status()
    
    with col1:
        st.metric(
            "Confidence Score",
            f"{model_status['confidence_score']}%",
            delta="Current model"
        )
    
    with col2:
        st.metric(
            "Training Data",
            model_status['trained_on'],
            delta="Dataset size"
        )
    
    with col3:
        st.metric(
            "Last Updated",
            model_status['last_updated'],
            delta="Model refresh"
        )
    
    st.markdown("---")
    
    # Current Model Metrics
    st.subheader("🎯 Current Model Metrics")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        model_metrics = get_model_metrics()
        fig = create_metric_bars(model_metrics, "Performance Metrics")
        st.plotly_chart(fig, width='stretch', key='model_performance_chart_1')
    
    with col2:
        st.markdown("### Metric Definitions")
        
        st.markdown("""
        <div style="background: #1A1A1A; padding: 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 12px;">
            <div style="color: #FFFFFF; font-weight: 600; margin-bottom: 8px;">📊 Precision</div>
            <div style="color: #B0B0B0; font-size: 14px;">
                Percentage of predicted categories that are correct. High precision means fewer false positives.
            </div>
        </div>
        
        <div style="background: #1A1A1A; padding: 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 12px;">
            <div style="color: #8B5CF6; font-weight: 600; margin-bottom: 8px;">📊 Recall</div>
            <div style="color: #B0B0B0; font-size: 14px;">
                Percentage of actual categories that are correctly identified. High recall means fewer false negatives.
            </div>
        </div>
        
        <div style="background: #1A1A1A; padding: 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 12px;">
            <div style="color: #06B6D4; font-weight: 600; margin-bottom: 8px;">📊 F1-Score</div>
            <div style="color: #B0B0B0; font-size: 14px;">
                Harmonic mean of precision and recall. Provides a balanced measure of model performance.
            </div>
        </div>
        
        <div style="background: #1A1A1A; padding: 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
            <div style="color: #10B981; font-weight: 600; margin-bottom: 8px;">📊 Accuracy</div>
            <div style="color: #B0B0B0; font-size: 14px;">
                Overall percentage of correct predictions across all categories.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Comparison
    st.subheader("🔬 Model Comparison")
    
    comparison_df = get_model_comparison()
    
    # Display as styled table
    st.markdown("""
    <div style="background: #1A1A1A; padding: 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
    """, unsafe_allow_html=True)
    
    # Format the dataframe for better display
    display_df = comparison_df.copy()
    display_df['accuracy'] = display_df['accuracy'].apply(lambda x: f"{x:.1%}")
    display_df['precision'] = display_df['precision'].apply(lambda x: f"{x:.1%}")
    display_df['recall'] = display_df['recall'].apply(lambda x: f"{x:.1%}")
    display_df['f1_score'] = display_df['f1_score'].apply(lambda x: f"{x:.1%}")
    
    display_df.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    st.dataframe(
        display_df,
        width='stretch',
        hide_index=True
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Best model highlight
    best_model = comparison_df.loc[comparison_df['accuracy'].idxmax(), 'model']
    best_accuracy = comparison_df['accuracy'].max()
    
    st.success(f"**🏆 Best Performing Model:** {best_model} with {best_accuracy:.1%} accuracy")
    
    st.markdown("---")
    
    # Model Versions
    st.subheader("📦 Model Version History")
    
    versions = get_model_versions()
    
    col1, col2, col3 = st.columns(3)
    
    for i, version in enumerate(versions):
        col = [col1, col2, col3][i % 3]
        
        with col:
            status_color = "#10B981" if version['status'] == 'Current' else "#6B7280"
            border_color = "rgba(16, 185, 129, 0.3)" if version['status'] == 'Current' else "rgba(255,255,255,0.1)"
            
            st.markdown(f"""
            <div style="background: #1A1A1A; padding: 20px; border-radius: 12px; border: 2px solid {border_color}; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div style="color: #FFFFFF; font-size: 20px; font-weight: 700;">{version['version']}</div>
                    <div style="background: {status_color}20; color: {status_color}; padding: 6px 14px; border-radius: 12px; font-size: 11px; font-weight: 600; border: 1px solid {status_color};">
                        {version['status']}
                    </div>
                </div>
                <div style="color: #B0B0B0; font-size: 14px; margin-bottom: 8px;">
                    📅 {version['date']}
                </div>
                <div style="color: #64748B; font-size: 13px;">
                    {'Production model' if version['status'] == 'Current' else 'Legacy version'}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Training Information
    st.subheader("🔧 Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: #1A1A1A; padding: 20px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
            <h4 style="color: #FFFFFF; margin-top: 0;">Category Model</h4>
            <div style="color: #B0B0B0; margin-bottom: 12px;">
                <strong style="color: #FFFFFF;">Algorithm:</strong> Support Vector Machine (SVM)
            </div>
            <div style="color: #B0B0B0; margin-bottom: 12px;">
                <strong style="color: #FFFFFF;">Kernel:</strong> Linear
            </div>
            <div style="color: #B0B0B0; margin-bottom: 12px;">
                <strong style="color: #FFFFFF;">Features:</strong> TF-IDF Vectorization
            </div>
            <div style="color: #B0B0B0;">
                <strong style="color: #FFFFFF;">Classes:</strong> 6 categories
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #1A1A1A; padding: 20px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
            <h4 style="color: #FFFFFF; margin-top: 0;">Priority Model</h4>
            <div style="color: #B0B0B0; margin-bottom: 12px;">
                <strong style="color: #8B5CF6;">Algorithm:</strong> Random Forest
            </div>
            <div style="color: #B0B0B0; margin-bottom: 12px;">
                <strong style="color: #8B5CF6;">Estimators:</strong> 100 trees
            </div>
            <div style="color: #B0B0B0; margin-bottom: 12px;">
                <strong style="color: #8B5CF6;">Features:</strong> TF-IDF Vectorization
            </div>
            <div style="color: #B0B0B0;">
                <strong style="color: #8B5CF6;">Classes:</strong> 3 priority levels
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    show()
