"""
Overview page - Enhanced dashboard with modern visualizations.
"""

import streamlit as st
import pandas as pd
from utils import (
    get_dashboard_metrics,
    get_category_distribution,
    get_weekly_trends,
    get_accuracy_trend,
    get_sentiment_distribution,
    get_model_metrics,
    get_recent_tickets,
    get_model_versions,
    get_model_status,
    create_category_bar_chart,
    create_line_chart,
    create_pie_chart,
    create_metric_bars,
    create_gauge_chart,
    get_category_color,
    get_priority_color,
    get_sentiment_color,
)


def show():
    """Display the enhanced overview dashboard page."""
    
    # Header with gradient effect
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">Dashboard</h1>
        <p style="font-size: 1.2rem; color: #B0B0B0;">Real-time AI-powered ticket analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Metrics - Enhanced with better spacing
    metrics = get_dashboard_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Tickets Processed",
            value=f"{metrics['tickets_processed']:,}",
            delta="Last 30 days"
        )
    
    with col2:
        st.metric(
            label="🎯 Classification Accuracy",
            value=f"{metrics['classification_accuracy']}%",
            delta="Current model"
        )
    
    with col3:
        st.metric(
            label="⚡ Avg. Resolution Time",
            value=metrics['avg_resolution_time'],
            delta="Down from 2.9h",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="🔥 Critical Issues",
            value=metrics['critical_issues'],
            delta=metrics['critical_change'],
            delta_color="inverse"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main Charts Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 Tickets by Category")
        st.markdown("Distribution of support tickets across different categories")
        category_data = get_category_distribution()
        fig = create_category_bar_chart(category_data)
        st.plotly_chart(fig, width='stretch', key='overview_chart_1')
    
    with col2:
        st.markdown("### 🎯 Model Performance")
        st.markdown("Current ML model accuracy metrics")
        model_metrics = get_model_metrics()
        fig = create_metric_bars(model_metrics)
        st.plotly_chart(fig, width='stretch', key='overview_chart_2')
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Trends Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Weekly Ticket Trends")
        st.markdown("Track resolved, pending, and new tickets over the past week")
        trends_data = get_weekly_trends()
        fig = create_line_chart(
            trends_data,
            'day',
            ['resolved', 'pending', 'new'],
            "",
            colors=['#4ADE80', '#FBBF24', '#EF4444']
        )
        st.plotly_chart(fig, width='stretch', key='overview_chart_3')
    
    with col2:
        st.markdown("### 📈 Accuracy Improvement")
        st.markdown("Model accuracy trend over 6 weeks")
        accuracy_data = get_accuracy_trend()
        fig = create_line_chart(
            accuracy_data,
            'week',
            ['accuracy'],
            "",
            colors=['#FFFFFF']
        )
        st.plotly_chart(fig, width='stretch', key='overview_chart_4')
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Bottom Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Recent Tickets")
        st.markdown("Latest support tickets with AI classifications")
        recent_tickets = get_recent_tickets()
        
        # Create styled table
        for _, ticket in recent_tickets.iterrows():
            cat_color = get_category_color(ticket['category'])
            pri_color = get_priority_color(ticket['priority'])
            sent_color = get_sentiment_color(ticket['sentiment'])
            
            confidence_pct = int(ticket['confidence'] * 100)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1A1A1A 0%, #0F0F0F 100%); 
                        padding: 1.25rem; border-radius: 12px; margin-bottom: 1rem; 
                        border: 1px solid #2A2A2A; transition: all 0.3s ease;">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.75rem;">
                    <div style="flex: 1;">
                        <div style="color: #B0B0B0; font-size: 0.85rem; margin-bottom: 0.25rem;">{ticket['id']}</div>
                        <div style="color: #FFFFFF; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">{ticket['title']}</div>
                        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                            <span style="background: {cat_color}20; color: {cat_color}; padding: 0.25rem 0.75rem; 
                                       border-radius: 12px; font-size: 0.75rem; font-weight: 600; border: 1px solid {cat_color};">
                                {ticket['category']}
                            </span>
                            <span style="background: {pri_color}20; color: {pri_color}; padding: 0.25rem 0.75rem; 
                                       border-radius: 12px; font-size: 0.75rem; font-weight: 600; border: 1px solid {pri_color};">
                                {ticket['priority']}
                            </span>
                            <span style="background: {sent_color}20; color: {sent_color}; padding: 0.25rem 0.75rem; 
                                       border-radius: 12px; font-size: 0.75rem; font-weight: 600; border: 1px solid {sent_color};">
                                {ticket['sentiment']}
                            </span>
                        </div>
                    </div>
                    <div style="text-align: right; min-width: 120px;">
                        <div style="color: #FFFFFF; font-size: 1.5rem; font-weight: 700;">{confidence_pct}%</div>
                        <div style="color: #B0B0B0; font-size: 0.8rem;">Confidence</div>
                        <div style="color: #808080; font-size: 0.75rem; margin-top: 0.25rem;">{ticket['created']}</div>
                    </div>
                </div>
                <div style="background: #0A0A0A; height: 6px; border-radius: 3px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #FFFFFF 0%, #B0B0B0 100%); 
                               height: 100%; width: {confidence_pct}%; transition: width 0.5s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Sentiment Distribution
        st.markdown("### 😊 Sentiment Analysis")
        st.markdown("Customer emotional tone distribution")
        sentiment_data = get_sentiment_distribution()
        fig = create_pie_chart(
            list(sentiment_data.keys()),
            list(sentiment_data.values()),
            ""
        )
        st.plotly_chart(fig, width='stretch', key='overview_chart_5')
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Model Status Card
        st.markdown("### ⚙️ Model Status")
        model_status = get_model_status()
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1A1A1A 0%, #0F0F0F 100%); 
                    padding: 1.5rem; border-radius: 12px; border: 1px solid #2A2A2A;">
            <div style="margin-bottom: 1.25rem;">
                <div style="color: #B0B0B0; font-size: 0.8rem; text-transform: uppercase; 
                           letter-spacing: 0.1em; margin-bottom: 0.5rem;">Confidence Score</div>
                <div style="color: #FFFFFF; font-size: 2.5rem; font-weight: 800;">
                    {model_status['confidence_score']}%
                </div>
            </div>
            <div style="margin-bottom: 1rem;">
                <div style="color: #B0B0B0; font-size: 0.8rem; margin-bottom: 0.25rem;">Training Data</div>
                <div style="color: #FFFFFF; font-size: 1.1rem; font-weight: 600;">{model_status['trained_on']}</div>
            </div>
            <div>
                <div style="color: #B0B0B0; font-size: 0.8rem; margin-bottom: 0.25rem;">Last Updated</div>
                <div style="color: #FFFFFF; font-size: 1.1rem; font-weight: 600;">{model_status['last_updated']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Model Versions
        st.markdown("### 📦 Model Versions")
        versions = get_model_versions()
        
        for version in versions:
            status_color = "#4ADE80" if version['status'] == 'Current' else "#6B7280"
            border_style = "2px solid #4ADE80" if version['status'] == 'Current' else "1px solid #2A2A2A"
            
            st.markdown(f"""
            <div style="background: #1A1A1A; padding: 1rem; border-radius: 12px; 
                        border: {border_style}; margin-bottom: 0.75rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="color: #FFFFFF; font-size: 1.1rem; font-weight: 700;">{version['version']}</div>
                        <div style="color: #B0B0B0; font-size: 0.85rem;">{version['date']}</div>
                    </div>
                    <div style="background: {status_color}20; color: {status_color}; padding: 0.4rem 1rem; 
                               border-radius: 12px; font-size: 0.75rem; font-weight: 700; border: 1px solid {status_color};">
                        {version['status']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    show()
