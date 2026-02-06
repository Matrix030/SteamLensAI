#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import time
import streamlit as st
from typing import Dict, Any

from ..core.summarize_processor import summarize_report

def render_summarize_tab() -> None:
    
    st.header("Generate Sentiment Summaries")
    st.write("""
    This step uses GPU-accelerated summarization to generate separate summaries for positive and negative reviews.
    This gives you better insights into what players love and hate about each theme.
    """)
    
    # Check if we have a result from previous step
    if 'result' in st.session_state and st.session_state.result:
        # Start summarization button
        if st.button("🚀 Start Summarization", key="summarize_button"):
            with st.spinner("Initializing Dask cluster..."):
                # Record start time for performance comparison
                start_time = time.time()
                st.session_state.timing_data['summarize_start_time'] = start_time
                
                # Inform user that summarization is starting
                st.info("Dask cluster is being initialized. Summarization will start shortly...")
            
            # Run summarization (note: moved outside the spinner to allow dashboard to show)
            summarized_report = summarize_report(st.session_state.result['final_report'])
            
            # Calculate elapsed time
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.session_state.timing_data['summarize_end_time'] = end_time
            
            if summarized_report is not None:
                st.session_state.summarized_report = summarized_report
                
                # Display execution time in a prominent way
                st.success(f"✅ Sentiment summarization completed in {elapsed_time:.2f} seconds!")
                
                # Create a dedicated section for timing metrics
                st.subheader("Execution Time Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Time", f"{elapsed_time:.2f} seconds")
                with col2:
                    per_item = elapsed_time/len(summarized_report) if len(summarized_report) > 0 else 0
                    st.metric("Average Per Item", f"{per_item:.4f} seconds")
                
                # Show sample of summarized data
                with st.expander("Show sample of sentiment summaries"):
                    sample_columns = ['steam_appid', 'Theme', 'Positive_Summary', 'Negative_Summary']
                    st.dataframe(summarized_report[sample_columns].head(5))
                
                # Switch to results tab
                st.info("👉 Go to the 'Results' tab to view the complete sentiment analysis")
    else:
        st.info("Please complete the 'Upload & Process' step first") 