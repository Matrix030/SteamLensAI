#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import streamlit as st
import pandas as pd
from typing import Dict, List, Any

def render_results_tab() -> None:
    
    st.header("Sentiment Analysis Results")
    
    # Check if we have summarized results
    if 'summarized_report' in st.session_state and not st.session_state.summarized_report.empty:
        show_summarized_results(st.session_state.summarized_report)
    elif 'result' in st.session_state and st.session_state.result:
        # We have processed data but no summarization
        show_unsummarized_results(st.session_state.result['final_report'])
    else:
        st.info("Please complete the 'Upload & Process' step first")

def show_summarized_results(summarized_report: pd.DataFrame) -> None:
    
    # Get game name mapping if available
    game_name_mapping = {}
    if 'result' in st.session_state and 'game_name_mapping' in st.session_state.result:
        game_name_mapping = st.session_state.result['game_name_mapping']
    
    # Display filters
    st.subheader("Filter Results")
    app_ids = sorted(summarized_report['steam_appid'].unique())
    
    # Create a list of options that include both app ID and game name
    app_id_options = []
    for app_id in app_ids:
        if app_id in game_name_mapping and game_name_mapping[app_id]:
            app_id_options.append(f"{app_id} - {game_name_mapping[app_id]}")
        else:
            app_id_options.append(f"{app_id}")
    
    selected_app_id_option = st.selectbox("Select Game", app_id_options)
    
    # Extract the app_id from the selected option
    selected_app_id = int(selected_app_id_option.split(' - ')[0])
    
    # Filter by app ID
    filtered_df = summarized_report[summarized_report['steam_appid'] == selected_app_id]
    
    # Get the game name for display
    game_name = game_name_mapping.get(selected_app_id, "Unknown Game")
    
    # Display the game name prominently
    if selected_app_id in game_name_mapping and game_name_mapping[selected_app_id]:
        st.subheader(f"Game: {game_name} (App ID: {selected_app_id})")
    else:
        st.subheader(f"App ID: {selected_app_id}")
    
    # Display themes for this app ID
    themes = filtered_df['Theme'].unique()
    selected_theme = st.selectbox("Select Theme", themes)
    
    # Display the results for the selected theme
    theme_data = filtered_df[filtered_df['Theme'] == selected_theme]
    
    if not theme_data.empty:
        st.subheader(f"Theme: {selected_theme}")
        
        # Basic metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Reviews", theme_data['#Reviews'].iloc[0])
        with col2:
            st.metric("Positive Reviews", f"{theme_data['Positive'].iloc[0]} ({theme_data['LikeRatio'].iloc[0]})")
        with col3:
            st.metric("Negative Reviews", f"{theme_data['Negative'].iloc[0]} ({theme_data['DislikeRatio'].iloc[0]})")
        
        # Sentiment summaries
        st.subheader("What Players Love ❤️")
        st.success(theme_data['Positive_Summary'].iloc[0])
        
        st.subheader("What Players Dislike 👎")
        st.error(theme_data['Negative_Summary'].iloc[0])
        
        # Display sample reviews by sentiment
        col1, col2 = st.columns(2)
        
        with col1:
            render_review_samples(theme_data, 'Positive_Reviews', "Sample Positive Reviews")
        
        with col2:
            render_review_samples(theme_data, 'Negative_Reviews', "Sample Negative Reviews")
    
    # Download buttons for the reports
    render_download_buttons(summarized_report, with_summaries=True)

def show_unsummarized_results(final_report: pd.DataFrame) -> None:
    
    # Get game name mapping if available
    game_name_mapping = {}
    if 'result' in st.session_state and 'game_name_mapping' in st.session_state.result:
        game_name_mapping = st.session_state.result['game_name_mapping']
    
    st.info("Summarization has not been performed yet. Only basic sentiment data is available.")
    
    # Display filters
    st.subheader("Filter Results")
    app_ids = sorted(final_report['steam_appid'].unique())
    
    # Create a list of options that include both app ID and game name
    app_id_options = []
    for app_id in app_ids:
        if app_id in game_name_mapping and game_name_mapping[app_id]:
            app_id_options.append(f"{app_id} - {game_name_mapping[app_id]}")
        else:
            app_id_options.append(f"{app_id}")
    
    selected_app_id_option = st.selectbox("Select Game", app_id_options)
    
    # Extract the app_id from the selected option
    selected_app_id = int(selected_app_id_option.split(' - ')[0])
    
    # Filter by app ID
    filtered_df = final_report[final_report['steam_appid'] == selected_app_id]
    
    # Get the game name for display
    game_name = game_name_mapping.get(selected_app_id, "Unknown Game")
    
    # Display the game name prominently
    if selected_app_id in game_name_mapping and game_name_mapping[selected_app_id]:
        st.subheader(f"Game: {game_name} (App ID: {selected_app_id})")
    else:
        st.subheader(f"App ID: {selected_app_id}")
    
    # Display themes for this app ID
    themes = filtered_df['Theme'].unique()
    selected_theme = st.selectbox("Select Theme", themes)
    
    # Display the results for the selected theme
    theme_data = filtered_df[filtered_df['Theme'] == selected_theme]
    
    if not theme_data.empty:
        st.subheader(f"Theme: {selected_theme}")
        
        # Basic metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Reviews", theme_data['#Reviews'].iloc[0])
        with col2:
            st.metric("Positive Reviews", f"{theme_data['Positive'].iloc[0]} ({theme_data['LikeRatio'].iloc[0]})")
        with col3:
            st.metric("Negative Reviews", f"{theme_data['Negative'].iloc[0]} ({theme_data['DislikeRatio'].iloc[0]})")
        
        # Display sample reviews by sentiment but without summaries
        col1, col2 = st.columns(2)
        
        with col1:
            render_review_samples(theme_data, 'Positive_Reviews', "Sample Positive Reviews")
        
        with col2:
            render_review_samples(theme_data, 'Negative_Reviews', "Sample Negative Reviews")
        
        st.subheader("Get Summaries")
        st.info("👉 Go to the 'Summarize' tab to generate sentiment-based summaries for what players love and dislike about each theme.")
    
    # Download button for the basic report
    render_download_buttons(final_report, with_summaries=False)
        
def render_review_samples(theme_data: pd.DataFrame, review_column: str, title: str) -> None:
    
    st.subheader(title)
    if (review_column in theme_data.columns and 
        isinstance(theme_data[review_column].iloc[0], list) and 
        theme_data[review_column].iloc[0]):
        
        reviews = theme_data[review_column].iloc[0]
        with st.expander(f"Show {min(5, len(reviews))} reviews"):
            for i, review in enumerate(reviews[:5]):  # Show first 5 reviews
                st.write(f"**Review {i+1}**")
                st.write(review[:300] + "..." if len(review) > 300 else review)
                st.write("---")
    else:
        st.write(f"No {review_column.lower()} available.")

def render_download_buttons(report: pd.DataFrame, with_summaries: bool = False) -> None:
    
    st.subheader("Download Reports")
    
    if with_summaries:
        col1, col2 = st.columns(2)
        
        with col1:
            csv = report.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Sentiment Report (CSV)",
                data=csv,
                file_name='steamLensAI_sentiment_report.csv',
                mime='text/csv',
            )
        
        with col2:
            # Create a simplified report for easier viewing
            simple_report = report[['steam_appid', 'Theme', '#Reviews', 'Positive', 'Negative', 
                                   'LikeRatio', 'DislikeRatio', 'Positive_Summary', 'Negative_Summary']]
            simple_csv = simple_report.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Simplified Report (CSV)",
                data=simple_csv,
                file_name='steamLensAI_sentiment_summary.csv',
                mime='text/csv',
            )
    else:
        # Just one download button for basic report
        csv = report.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Basic Sentiment Report (CSV)",
            data=csv,
            file_name='steamLensAI_basic_sentiment_report.csv',
            mime='text/csv',
        ) 