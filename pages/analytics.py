"""
Analytics page - Detailed analytics and insights.
"""

import streamlit as st
import pandas as pd
from utils import (
    get_category_distribution,
    get_weekly_trends,
    get_sentiment_distribution,
    create_category_bar_chart,
    create_line_chart,
    create_pie_chart,
    COLORS,
)


def show():
    """Display the analytics page."""
    
    # Header
    st.title("Analytics")
    st.markdown("**Detailed insights and trends**")
    st.markdown("---")
    
    # Category Analysis
    st.subheader("📊 Category Distribution")
    st.markdown("**Breakdown of support tickets by category.** Understanding category distribution helps allocate resources effectively and identify which areas need the most support attention.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        category_data = get_category_distribution()
        fig = create_category_bar_chart(category_data)
        st.plotly_chart(fig, width='stretch', key='analytics_chart_1')
    
    with col2:
        st.markdown("### Top Categories")
        category_df = pd.DataFrame(list(category_data.items()), columns=['Category', 'Count'])
        category_df = category_df.sort_values('Count', ascending=False)
        
        for _, row in category_df.iterrows():
            percentage = (row['Count'] / category_df['Count'].sum()) * 100
            st.markdown(f"""
            <div style="margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: #FFFFFF; font-size: 14px;">{row['Category']}</span>
                    <span style="color: #B0B0B0; font-size: 14px;">{row['Count']} ({percentage:.1f}%)</span>
                </div>
                <div style="background: #1A1A1A; height: 6px; border-radius: 3px; overflow: hidden;">
                    <div style="background: #FFFFFF; height: 100%; width: {percentage}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Ticket Trends
    st.subheader("📈 Ticket Trends Over Time")
    st.markdown("**Weekly ticket volume analysis showing resolved, pending, and new tickets.** This trend helps identify peak support periods and measure team efficiency in handling customer issues.")
    
    trends_data = get_weekly_trends()
    
    # Add total column
    trends_data['total'] = trends_data['resolved'] + trends_data['pending'] + trends_data['new']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_resolved = trends_data['resolved'].sum()
        st.metric("Total Resolved", total_resolved)
    
    with col2:
        total_pending = trends_data['pending'].sum()
        st.metric("Total Pending", total_pending)
    
    with col3:
        total_new = trends_data['new'].sum()
        st.metric("Total New", total_new)
    
    with col4:
        resolution_rate = (total_resolved / (total_resolved + total_pending + total_new)) * 100
        st.metric("Resolution Rate", f"{resolution_rate:.1f}%")
    
    fig = create_line_chart(
        trends_data,
        'day',
        ['resolved', 'pending', 'new'],
        "Weekly Trends",
        colors=['#06B6D4', '#F59E0B', '#EC4899']
    )
    st.plotly_chart(fig, width='stretch', key='analytics_chart_2')
    
    st.markdown("---")
    
    # Sentiment Analysis
    st.subheader("😊 Sentiment Analysis")
    st.markdown("**Emotional tone of customer communications.** Sentiment analysis reveals customer satisfaction levels and helps prioritize tickets that may indicate frustrated or unhappy customers requiring immediate attention.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        sentiment_data = get_sentiment_distribution()
        fig = create_pie_chart(
            list(sentiment_data.keys()),
            list(sentiment_data.values()),
            ""
        )
        st.plotly_chart(fig, width='stretch', key='analytics_chart_3')
    
    with col2:
        st.markdown("### Sentiment Breakdown")
        
        total_sentiment = sum(sentiment_data.values())
        
        for sentiment, count in sentiment_data.items():
            percentage = (count / total_sentiment) * 100
            
            if sentiment == 'Positive':
                color = '#06B6D4'
                icon = '😊'
            elif sentiment == 'Neutral':
                color = '#8B5CF6'
                icon = '😐'
            else:
                color = '#EC4899'
                icon = '😞'
            
            st.markdown(f"""
            <div style="background: #1A1A1A; padding: 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 12px;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 32px;">{icon}</span>
                        <div>
                            <div style="color: #FFFFFF; font-size: 18px; font-weight: 600;">{sentiment}</div>
                            <div style="color: #B0B0B0; font-size: 14px;">{count} tickets</div>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: {color}; font-size: 24px; font-weight: 700;">{percentage:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Additional Insights
    st.subheader("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📌 Most Common Issue**
        
        Bug Reports account for the highest volume of tickets (342), suggesting a need for improved quality assurance.
        """)
        
        st.success("""
        **✅ Resolution Efficiency**
        
        Average resolution time has decreased from 2.9h to 2.4h, showing improved team efficiency.
        """)
    
    with col2:
        st.warning("""
        **⚠️ Priority Distribution**
        
        12 critical issues require immediate attention. Consider allocating additional resources.
        """)
        
        st.info("""
        **📊 Sentiment Trend**
        
        53% of tickets have neutral sentiment, indicating factual issue reporting without strong emotions.
        """)


if __name__ == "__main__":
    show()
