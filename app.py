"""
Support Ticket Classification Dashboard
Main Streamlit Application
"""

import streamlit as st
from pathlib import Path
from utils import get_dashboard_metrics

# Page configuration
st.set_page_config(
    page_title="AI Ticket Analysis",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    """Load custom CSS styling."""
    css_file = Path(__file__).parent / "assets" / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ========== SIDEBAR CONFIGURATION ==========
st.sidebar.title("🎫 Support Analytics")
st.sidebar.markdown("---")

# Navigation
st.sidebar.subheader("📍 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Overview", "📤 Upload Tickets", "📈 Analytics", "🎯 Model Performance"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Overall Info Section (Persistent Sidebar Metrics)
st.sidebar.subheader("⚡ System Status")

# Get real-time metrics
metrics = get_dashboard_metrics()

# Custom HTML for sidebar metrics to look professional
st.sidebar.markdown(f"""
<div style="background: rgba(255, 255, 255, 0.05); padding: 10px; border-radius: 8px; margin-bottom: 10px;">
    <div style="color: #B0B0B0; font-size: 0.8rem;">Total Tickets</div>
    <div style="color: #FFFFFF; font-size: 1.2rem; font-weight: bold;">{metrics['tickets_processed']:,}</div>
</div>
<div style="background: rgba(255, 255, 255, 0.05); padding: 10px; border-radius: 8px; margin-bottom: 10px;">
    <div style="color: #B0B0B0; font-size: 0.8rem;">Model Accuracy</div>
    <div style="color: #FFFFFF; font-size: 1.2rem; font-weight: bold;">{metrics['classification_accuracy']}%</div>
</div>
<div style="background: rgba(255, 255, 255, 0.05); padding: 10px; border-radius: 8px;">
    <div style="color: #B0B0B0; font-size: 0.8rem;">Critical Issues</div>
    <div style="color: #EF4444; font-size: 1.2rem; font-weight: bold;">{metrics['critical_issues']}</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.info("v2.4.0 • Production")

# ========== MAIN CONTENT RENDERING ==========
# This approach avoids DuplicateElementId errors by only rendering one page at a time

if page == "📊 Overview":
    from pages import overview
    overview.show()

elif page == "📤 Upload Tickets":
    from pages import upload_tickets
    upload_tickets.show()

elif page == "📈 Analytics":
    from pages import analytics
    analytics.show()

elif page == "🎯 Model Performance":
    from pages import model_performance
    model_performance.show()
