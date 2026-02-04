"""
Upload Tickets page - Enhanced file upload and classification with modern design.
"""

import streamlit as st
import pandas as pd
import traceback
from utils import (
    load_models,
    predict_batch,
    parse_uploaded_file,
    parse_pasted_data,
    validate_ticket_data,
    normalize_ticket_data,
    add_predictions_to_dataframe,
    create_category_bar_chart,
    create_pie_chart,
    get_category_color,
    get_priority_color,
    get_sentiment_color,
)


def show():
    """Display the enhanced upload tickets page."""
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">Upload & Classify Tickets</h1>
        <p style="font-size: 1.2rem; color: #B0B0B0;">Import and analyze support tickets with AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Initialize session state
    if 'processed_tickets' not in st.session_state:
        st.session_state.processed_tickets = None
    
    # Upload method selection with modern styling
    st.markdown("### 📥 Choose Upload Method")
    upload_method = st.radio(
        "Upload Method",
        ["📁 Upload File", "📝 Paste Data"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    uploaded_data = None
    
    if upload_method == "📁 Upload File":
        st.markdown("### Upload Your Tickets")
        st.markdown("Supported formats: **CSV**, **JSON**, **Excel** (xlsx, xls)")
        
        uploaded_file = st.file_uploader(
            "Drag and drop your file here or click to browse",
            type=['csv', 'json', 'xlsx', 'xls'],
            help="Upload a file containing ticket descriptions",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            try:
                with st.spinner("📂 Parsing file..."):
                    uploaded_data = parse_uploaded_file(uploaded_file)
                st.success(f"✅ Successfully loaded **{len(uploaded_data)}** tickets from `{uploaded_file.name}`")
            except Exception as e:
                st.error(f"❌ Error parsing file: {str(e)}")
                st.code(traceback.format_exc(), language="python")
    
    else:  # Paste Data
        st.markdown("### Paste Your Ticket Data")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            data_format = st.selectbox("Format:", ["CSV", "JSON"])
        
        with col1:
            pasted_text = st.text_area(
                "Paste your data here:",
                height=250,
                placeholder="Example CSV:\ndescription,title\nLogin page not loading,Login issue\nRequest for dark mode,Feature request",
                label_visibility="collapsed"
            )
        
        if st.button("🚀 Process Pasted Data", type="primary"):
            if pasted_text.strip():
                try:
                    with st.spinner("📝 Processing pasted data..."):
                        uploaded_data = parse_pasted_data(pasted_text, format=data_format.lower())
                    st.success(f"✅ Successfully parsed **{len(uploaded_data)}** tickets")
                except Exception as e:
                    st.error(f"❌ Error parsing data: {str(e)}")
                    st.code(traceback.format_exc(), language="python")
            else:
                st.warning("⚠️ Please paste some data first")
    
    # Process uploaded data
    if uploaded_data is not None:
        st.markdown("---")
        st.markdown("### 📊 Data Preview")
        
        # Validate data
        is_valid, error_msg = validate_ticket_data(uploaded_data)
        
        if not is_valid:
            st.markdown(error_msg)
            return
        
        # Normalize data
        normalized_data = normalize_ticket_data(uploaded_data)
        
        # Show preview in styled container
        st.markdown(f"""
        <div style="background: #1A1A1A; padding: 1rem; border-radius: 12px; border: 1px solid #2A2A2A; margin-bottom: 1rem;">
            <div style="color: #B0B0B0; font-size: 0.9rem; margin-bottom: 0.5rem;">
                Showing first 10 rows of {len(normalized_data)} total tickets
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(normalized_data.head(10), width='stretch')
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Classification button
        if st.button("🤖 Classify All Tickets with AI", type="primary"):
            with st.spinner("🔄 Loading ML models..."):
                try:
                    models = load_models()
                    st.success("✅ Models loaded successfully!")
                except Exception as e:
                    st.error(f"❌ **Error loading models:** {str(e)}")
                    st.info("💡 **Tip:** Please ensure all model files are present in the `models/` directory.")
                    st.code(traceback.format_exc(), language="python")
                    return
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total):
                progress = current / total
                progress_bar.progress(progress)
                status_text.markdown(f"**Classifying:** {current}/{total} tickets ({progress*100:.1f}%)")
            
            # Predict
            try:
                with st.spinner("🤖 AI is classifying your tickets..."):
                    predictions = predict_batch(
                        normalized_data['description'].tolist(),
                        models,
                        progress_callback=update_progress
                    )
                    
                    # Add predictions to dataframe
                    results_df = add_predictions_to_dataframe(normalized_data, predictions)
                    
                    # Store in session state
                    st.session_state.processed_tickets = results_df
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"🎉 Successfully classified **{len(results_df)}** tickets!")
                    
            except Exception as e:
                st.error(f"❌ **Error during classification:** {str(e)}")
                st.code(traceback.format_exc(), language="python")
                return
    
    # Display results if available
    if st.session_state.processed_tickets is not None:
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <h2 style="font-size: 2.5rem;">🎯 Classification Results</h2>
        </div>
        """, unsafe_allow_html=True)
        
        results_df = st.session_state.processed_tickets
        
        # Summary metrics with modern cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Tickets", len(results_df))
        
        with col2:
            avg_confidence = results_df['overall_confidence'].mean()
            st.metric("🎯 Avg. Confidence", f"{avg_confidence:.1%}")
        
        with col3:
            high_priority = (results_df['predicted_priority'] == 'High').sum()
            st.metric("🔥 High Priority", high_priority)
        
        with col4:
            unique_categories = results_df['predicted_category'].nunique()
            st.metric("📁 Categories", unique_categories)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Predicted Categories")
            category_counts = results_df['predicted_category'].value_counts()
            fig = create_category_bar_chart(category_counts)
            st.plotly_chart(fig, width='stretch', key='upload_tickets_chart_1')
        
        with col2:
            st.markdown("### 🎯 Priority Distribution")
            priority_counts = results_df['predicted_priority'].value_counts()
            fig = create_pie_chart(
                priority_counts.index.tolist(),
                priority_counts.values.tolist(),
                ""
            )
            st.plotly_chart(fig, width='stretch', key='upload_tickets_chart_2')
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Results table with filters
        st.markdown("### 📋 Detailed Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            category_filter = st.multiselect(
                "Filter by Category:",
                options=results_df['predicted_category'].unique().tolist(),
                default=[]
            )
        
        with col2:
            priority_filter = st.multiselect(
                "Filter by Priority:",
                options=results_df['predicted_priority'].unique().tolist(),
                default=[]
            )
        
        with col3:
            sentiment_filter = st.multiselect(
                "Filter by Sentiment:",
                options=results_df['sentiment'].unique().tolist(),
                default=[]
            )
        
        # Apply filters
        filtered_df = results_df.copy()
        
        if category_filter:
            filtered_df = filtered_df[filtered_df['predicted_category'].isin(category_filter)]
        
        if priority_filter:
            filtered_df = filtered_df[filtered_df['predicted_priority'].isin(priority_filter)]
        
        if sentiment_filter:
            filtered_df = filtered_df[filtered_df['sentiment'].isin(sentiment_filter)]
        
        st.info(f"📊 Showing **{len(filtered_df)}** of **{len(results_df)}** tickets")
        
        # Display styled results
        for _, row in filtered_df.iterrows():
            cat_color = get_category_color(row['predicted_category'])
            pri_color = get_priority_color(row['predicted_priority'])
            sent_color = get_sentiment_color(row['sentiment'])
            
            cat_conf = int(row['category_confidence'] * 100)
            pri_conf = int(row['priority_confidence'] * 100)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1A1A1A 0%, #0F0F0F 100%); 
                        padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; 
                        border: 1px solid #2A2A2A;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                    <div style="flex: 1;">
                        <div style="color: #FFFFFF; font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem;">
                            {row.get('title', row.get('description', '')[:50])}
                        </div>
                        <div style="color: #B0B0B0; font-size: 0.9rem; margin-bottom: 1rem;">
                            {row['description'][:150]}{'...' if len(row['description']) > 150 else ''}
                        </div>
                        <div style="display: flex; gap: 0.75rem; flex-wrap: wrap;">
                            <div style="text-align: center;">
                                <div style="background: {cat_color}20; color: {cat_color}; padding: 0.5rem 1rem; 
                                           border-radius: 12px; font-weight: 700; border: 1px solid {cat_color};">
                                    {row['predicted_category']}
                                </div>
                                <div style="color: #808080; font-size: 0.75rem; margin-top: 0.25rem;">{cat_conf}% confidence</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="background: {pri_color}20; color: {pri_color}; padding: 0.5rem 1rem; 
                                           border-radius: 12px; font-weight: 700; border: 1px solid {pri_color};">
                                    {row['predicted_priority']} Priority
                                </div>
                                <div style="color: #808080; font-size: 0.75rem; margin-top: 0.25rem;">{pri_conf}% confidence</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="background: {sent_color}20; color: {sent_color}; padding: 0.5rem 1rem; 
                                           border-radius: 12px; font-weight: 700; border: 1px solid {sent_color};">
                                    {row['sentiment']}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Export functionality
        st.markdown("---")
        st.markdown("### 💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="classified_tickets.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = results_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name="classified_tickets.json",
                mime="application/json"
            )


if __name__ == "__main__":
    show()
