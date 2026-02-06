#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import time
import streamlit as st
import os
import sys

# Add the src directory to the path if running directly
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Now use absolute imports
from steamlensai.config.app_config import STREAMLIT_PAGE_CONFIG
from steamlensai.ui.upload_tab import render_upload_tab
from steamlensai.ui.summarize_tab import render_summarize_tab
from steamlensai.ui.results_tab import render_results_tab
from steamlensai.ui.sidebar import render_sidebar

def main():
    """Main Streamlit UI with comprehensive timing and sentiment analysis"""
    # Configure the Streamlit page
    st.set_page_config(**STREAMLIT_PAGE_CONFIG)
    
    if "total_rows" not in st.session_state:
        st.session_state["total_rows"] = None

    # Start the global execution timer
    global_start_time = time.time()
    
    # Create a session state variable to store timing data if it doesn't exist
    if 'timing_data' not in st.session_state:
        st.session_state.timing_data = {
            'global_start_time': global_start_time,
            'process_start_time': None,
            'process_end_time': None,
            'summarize_start_time': None,
            'summarize_end_time': None,
            'global_end_time': None
        }
    
    # Render the title and description
    st.title("steamLensAI")
    st.write("This tool separates positive and negative reviews for deeper insights into what players love and hate.")
    
    # Render the sidebar
    render_sidebar()
    
    # Main content - Tabs
    tab1, tab2, tab3 = st.tabs(["Upload & Process", "Summarize", "Results"])
    
    # Render the tabs
    with tab1:
        render_upload_tab()
    
    with tab2:
        render_summarize_tab()
    
    with tab3:
        render_results_tab()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # If there's an unhandled exception, record the end time to still get timing data
        if 'timing_data' in st.session_state:
            st.session_state.timing_data['global_end_time'] = time.time()
        st.error(f"An error occurred: {str(e)}")
        raise e 