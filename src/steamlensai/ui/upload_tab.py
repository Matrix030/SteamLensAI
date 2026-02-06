#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import time
import streamlit as st
from typing import Dict, Any, Callable

from ..config.app_config import DEFAULT_THEME_FILE
from ..core.process_files import process_uploaded_files
from ..utils.system_utils import format_time

def render_upload_tab() -> None:
    
    st.header("Upload and Process Parquet Files")
    st.write("""
    Upload your Steam reviews Parquet files for processing. The app will:
    1. Check if each file's app ID exists in the theme dictionary
    2. Process only the files with matching app IDs
    3. Filter for English reviews and perform topic assignment
    4. Separate positive and negative reviews for each theme
    """)
    
    # Parquet file uploader
    uploaded_files = st.file_uploader(
        "Upload Parquet Files",
        type=["parquet"],
        accept_multiple_files=True,
        help="Upload one or more Parquet files containing Steam reviews data"
    )
    
    # Check if theme file exists
    theme_file_exists = check_theme_file_exists()
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} files")
        
        # Start processing button
        if st.button("Start Processing", key="process_button", disabled=not theme_file_exists):
           # Record start time for performance comparison
            start_time = time.time()
            st.session_state.timing_data['process_start_time'] = start_time
            # Inform user that processing is starting
            st.info("Dask cluster is being initialized. Processing will start shortly...")

            # Process the files (note: moved outside the spinner to allow dashboard to show)
            result = process_uploaded_files(uploaded_files, themes_file=DEFAULT_THEME_FILE)
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            st.session_state.timing_data['process_end_time'] = time.time()
            
            if result:
                st.session_state.result = result
                
                # Display summary
                st.subheader("Processing Summary")
                st.write(f"Processed {len(result['valid_files'])} files successfully")
                st.write(f"Skipped {len(result['skipped_files'])} files (app IDs not in theme dictionary)")
                
                # Show skipped files
                if result['skipped_files']:
                    with st.expander("Show skipped files"):
                        for file_name, app_id in result['skipped_files']:
                            st.write(f"- {file_name} (App ID: {app_id if app_id else 'Unknown'})")
                
                # Show sample of processed data
                with st.expander("Show sample of processed data"):
                    sample_df = result['final_report'][['steam_appid', 'Theme', '#Reviews', 'Positive', 'Negative', 'LikeRatio', 'DislikeRatio']].head(10)
                    st.dataframe(sample_df)
                
                # Switch to the summarize tab
                st.info("Go to the 'Summarize' tab to generate sentiment-based summaries")
    else:
        st.info("Please upload one or more Parquet files containing Steam reviews data.")

def check_theme_file_exists() -> bool:
    
    import os
    return os.path.exists(DEFAULT_THEME_FILE) 