#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os
import json
import numpy as np
import pandas as pd
import psutil
import torch
import time
import tempfile
import datetime
import streamlit as st
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# Set page configuration
st.set_page_config(
    page_title="steamLensAI - Sequential Analysis",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create directories if they don't exist
os.makedirs('output_csvs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

def get_system_resources():
    """Get system resources info"""
    # Get available memory (in GB)
    total_memory = psutil.virtual_memory().total / (1024**3)
    # Get CPU count
    cpu_count = psutil.cpu_count(logical=False)  # Physical cores only
    if not cpu_count:
        cpu_count = psutil.cpu_count(logical=True)  # Logical if physical not available
    
    return {
        'total_memory': total_memory,
        'cpu_count': cpu_count
    }

def extract_appid_from_parquet(file_path):
    """Extract appid from a parquet file"""
    try:
        # Read just a few rows to extract the app ID
        df = pd.read_parquet(file_path, columns=['steam_appid'])
        if df.empty:
            return None
        
        # Get the most common app ID in case there are multiple
        app_id = df['steam_appid'].mode()[0]
        return int(app_id)
    except Exception as e:
        st.error(f"Error extracting app ID from file: {str(e)}")
        return None

def get_theme_embeddings(app_ids, GAME_THEMES, embedder):
    """Get theme embeddings for a specific set of app IDs"""
    embeddings = {}
    for appid in app_ids:
        if appid not in embeddings and appid in GAME_THEMES:
            emb_list = []
            for theme, seeds in GAME_THEMES[appid].items():
                seed_emb = embedder.encode(seeds, convert_to_numpy=True)
                emb_list.append(seed_emb.mean(axis=0))
            embeddings[appid] = np.vstack(emb_list)
    return embeddings

def assign_topic(df, GAME_THEMES, embedder):
    """Assign topics to reviews - sequential version"""
    # If no rows, return as-is
    if df.empty:
        df['topic_id'] = []
        return df
    
    # Get unique app IDs in this partition
    app_ids = df['steam_appid'].unique().tolist()
    app_ids = [int(appid) for appid in app_ids]
    
    # Get embeddings only for app IDs in this dataframe
    local_theme_embeddings = get_theme_embeddings(app_ids, GAME_THEMES, embedder)
    
    # Process in smaller batches to avoid memory issues
    batch_size = 100
    topic_ids = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        reviews = batch['review'].tolist()
        
        # Compute embeddings for this batch
        review_embeds = embedder.encode(reviews, convert_to_numpy=True, batch_size=16)
        
        batch_topic_ids = []
        for idx, appid in enumerate(batch['steam_appid']):
            appid = int(appid)
            if appid in local_theme_embeddings:
                theme_embs = local_theme_embeddings[appid]
                sims = cosine_similarity(review_embeds[idx:idx+1], theme_embs)
                batch_topic_ids.append(int(sims.argmax()))
            else:
                # Default topic if theme embeddings not available
                batch_topic_ids.append(0)
        
        topic_ids.extend(batch_topic_ids)
    
    df['topic_id'] = topic_ids
    return df

def hierarchical_summary(reviews, summarizer):
    """Create hierarchical summary sequentially"""
    # Handle edge cases
    if not reviews or not isinstance(reviews, list):
        return "No reviews available for summarization."
    
    # Fast path for small review sets
    if len(reviews) <= 20:
        doc = "\n\n".join(reviews)
        return summarizer(
            doc,
            max_length=60,
            min_length=20,
            truncation=True,
            do_sample=False
        )[0]['summary_text']
    
    # Process larger review sets with chunking
    chunk_size = 20
    all_chunks = []
    for i in range(0, len(reviews), chunk_size):
        batch = reviews[i:i+chunk_size]
        text = "\n\n".join(batch)
        all_chunks.append(text)
    
    # Process chunks sequentially
    intermediate_summaries = []
    for chunk in all_chunks:
        result = summarizer(
            chunk,
            max_length=60,
            min_length=20,
            truncation=True,
            do_sample=False
        )[0]['summary_text']
        intermediate_summaries.append(result)
        
        # Occasional cleanup for GPU memory
        if torch.cuda.is_available() and len(intermediate_summaries) % 5 == 0:
            torch.cuda.empty_cache()
    
    # Create final summary
    joined = " ".join(intermediate_summaries)
    return summarizer(
        joined,
        max_length=60,
        min_length=20,
        truncation=True,
        do_sample=False
    )[0]['summary_text']

def process_uploaded_files(uploaded_files, themes_file="game_themes.json"):
    """Process uploaded parquet files - sequential version with timing metrics"""
    # Start phase timer for more detailed timing
    phase_start_time = time.time()
    if not uploaded_files:
        st.warning("Please upload at least one Parquet file to begin processing.")
        return None
    
    # Create progress indicators
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    with progress_placeholder.container():
        progress_bar = st.progress(0.0)
        status_text = st.empty()
    
    # Load theme dictionary
    try:
        with open(themes_file, 'r') as f:
            raw = json.load(f)
        GAME_THEMES = {int(appid): themes for appid, themes in raw.items()}
        status_text.write(f"✅ Loaded theme dictionary with {len(GAME_THEMES)} games")
    except FileNotFoundError:
        st.error(f"Theme file '{themes_file}' not found. Please upload it first.")
        return None
    
    # Create a temporary directory to store uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        valid_files = []
        skipped_files = []
        
        # Check and save valid files
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            
            # Save the uploaded file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Extract app ID and check if it's in the theme dictionary
            app_id = extract_appid_from_parquet(file_path)
            
            if app_id and app_id in GAME_THEMES:
                valid_files.append((file_path, app_id))
                status_text.write(f"✅ File '{uploaded_file.name}' has app ID {app_id} - Processing")
            else:
                skipped_files.append((uploaded_file.name, app_id))
                status_text.write(f"⚠️ File '{uploaded_file.name}' has app ID {app_id} - Skipping (not in theme dictionary)")
        
        # Check if we have any valid files
        if not valid_files:
            st.error("No valid files to process. All uploaded files' app IDs were not found in the theme dictionary.")
            return None
        
        status_text.write(f"Starting processing with {len(valid_files)} valid files...")
        
        # Get system resource info
        resources = get_system_resources()
        status_text.write(f"System has {resources['total_memory']:.1f}GB memory and {resources['cpu_count']} CPU cores")
        status_text.write("Processing sequentially without parallelization...")
        
        # Initialize SBERT embedder
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        status_text.write("Initialized sentence embedder: all-MiniLM-L6-v2")
        
        # Process each valid file sequentially
        all_dfs = []
        file_count = len(valid_files)
        
        for i, (file_path, app_id) in enumerate(valid_files):
            file_name = os.path.basename(file_path)
            status_text.write(f"Processing file {i+1}/{file_count}: {file_name} (App ID: {app_id})")
            progress_bar.progress((i) / file_count)
            
            try:
                # Read the parquet file with pandas
                df = pd.read_parquet(
                    file_path,
                    columns=['steam_appid', 'review', 'review_language', 'voted_up']
                )
                
                # Filter & Clean Data
                df = df[df['review_language'] == 'english']
                df = df.dropna(subset=['review'])
                
                # Only include matching app_id
                df = df[df['steam_appid'] == app_id]
                
                # Assign topics sequentially
                status_text.write(f"Assigning topics to {len(df)} reviews in file {i+1}/{file_count}...")
                df = assign_topic(df, GAME_THEMES, embedder)
                
                all_dfs.append(df)
                status_text.write(f"Completed file {i+1}/{file_count}: {file_name}")
                
            except Exception as e:
                status_text.write(f"⚠️ Error processing file {file_name}: {str(e)}")
        
        # Combine all dataframes
        if not all_dfs:
            st.error("No data to process after filtering. Please check your files.")
            return None
        
        status_text.write("Combining all valid data...")
        combined_df = pd.concat(all_dfs)
        
        # Get unique app IDs
        unique_app_ids = combined_df['steam_appid'].unique()
        total_app_ids = len(unique_app_ids)
        
        status_text.write(f"Processing {total_app_ids} unique app IDs")
        
        # Create empty dataframes for results
        rows = []
        total_groups = 0
        
        # Process each app ID and topic ID sequentially
        status_text.write("Aggregating data by app ID and topic ID...")
        
        # Create a progress bar for aggregation
        agg_progress = st.progress(0.0)
        
        # Using a more memory-efficient approach for groupby operations
        for i, appid in enumerate(unique_app_ids):
            agg_progress.progress(i / total_app_ids)
            
            # Filter data for this app ID
            app_df = combined_df[combined_df['steam_appid'] == appid]
            
            # Get topic IDs for this app
            topic_ids = app_df['topic_id'].unique()
            
            for tid in topic_ids:
                # Filter for this topic
                topic_df = app_df[app_df['topic_id'] == tid]
                
                # Get theme name
                if appid in GAME_THEMES:
                    theme_keys = list(GAME_THEMES[appid].keys())
                    if tid < len(theme_keys):
                        theme_name = theme_keys[tid]
                    else:
                        theme_name = f"Unknown Theme {tid}"
                else:
                    theme_name = f"Unknown Theme {tid}"
                
                # Calculate metrics
                total = len(topic_df)
                likes = topic_df['voted_up'].sum()
                like_ratio = f"{(likes / total * 100):.1f}%" if total > 0 else '0%'
                
                # Get reviews
                reviews_list = topic_df['review'].tolist()
                
                # Add to results
                rows.append({
                    'steam_appid': appid,
                    'Theme': theme_name,
                    '#Reviews': total,
                    'LikeRatio': like_ratio,
                    'Reviews': reviews_list
                })
                
                total_groups += 1
        
        # Create final report
        final_report = pd.DataFrame(rows)
        
        # Save intermediate results
        csv_path = 'output_csvs/Sequential_report.csv'
        final_report.to_csv(csv_path, index=False)
        status_text.write(f"✅ Saved report to {csv_path}")
        
        # Complete the progress bar
        progress_bar.progress(1.0)
        
        # Calculate elapsed time for this phase
        phase_elapsed_time = time.time() - phase_start_time
        formatted_time = str(datetime.timedelta(seconds=int(phase_elapsed_time)))
        
        status_text.write(f"✅ Data processing complete! Total processing time: {formatted_time}")
        
        # Return results
        return {
            'final_report': final_report,
            'valid_files': valid_files,
            'skipped_files': skipped_files
        }

def summarize_report(final_report):
    """Sequential GPU summarization phase with timing metrics"""
    # Start phase timer for more detailed timing
    phase_start_time = time.time()
    if final_report is None or final_report.empty:
        st.error("No report to summarize.")
        return None
    
    # Create progress indicators
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    with progress_placeholder.container():
        status_text = st.empty()
        progress_bar = st.progress(0.0)
    
    # Initialize the model for summarization
    status_text.write("Loading summarization model...")
    model_name = 'sshleifer/distilbart-cnn-12-6'
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with optimal settings
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    
    # Create optimized pipeline
    summarizer = pipeline(
        task='summarization',
        model=model,
        tokenizer=tokenizer,
        framework='pt'
    )
    
    # Report hardware status
    if torch.cuda.is_available():
        status_text.write(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        status_text.write("GPU not available, using CPU (this will be slow)")
    
    # Start timing
    start_time = time.time()
    
    # Process each row sequentially
    status_text.write(f"Generating summaries for {len(final_report)} items sequentially...")
    summaries = []
    
    for i, row in tqdm(final_report.iterrows(), total=len(final_report)):
        # Update progress
        progress_bar.progress(i / len(final_report))
        
        # Generate summary
        if isinstance(row['Reviews'], list) and len(row['Reviews']) > 0:
            summary = hierarchical_summary(row['Reviews'], summarizer)
        else:
            summary = "No reviews available for summarization."
        
        summaries.append(summary)
        
        # Cleanup GPU memory periodically
        if torch.cuda.is_available() and i % 10 == 0:
            torch.cuda.empty_cache()
    
    # Store results
    final_report['QuickSummary'] = summaries
    
    # Report timing
    elapsed_time = time.time() - start_time
    status_text.write(f"✅ Sequential processing completed in {elapsed_time:.2f} seconds")
    status_text.write(f"Average time per item: {elapsed_time/len(final_report):.2f} seconds")
    
    # Save results
    output_path = 'output_csvs/sequential_summarized_report.csv'
    final_report.to_csv(output_path)
    status_text.write(f"✅ Results saved to {output_path}")
    
    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Complete the progress bar
    progress_bar.progress(1.0)
    
    # Calculate elapsed time for this phase
    phase_elapsed_time = time.time() - phase_start_time
    formatted_time = str(datetime.timedelta(seconds=int(phase_elapsed_time)))
    status_text.write(f"✅ Total summarization time: {formatted_time}")
    
    return final_report

def main():
    """Main Streamlit UI"""
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
    
    st.title("🎮 steamLensAI: Sequential Steam Reviews Analysis")
    st.warning("⚠️ This version uses sequential processing without Dask. It will be significantly slower for large datasets.")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Theme file uploader
    theme_file = st.sidebar.file_uploader(
        "Upload Theme File (game_themes.json)",
        type=["json"],
        help="This file maps game app IDs to their themes"
    )
    
    if theme_file:
        # Save the theme file
        with open("game_themes.json", "wb") as f:
            f.write(theme_file.getbuffer())
        st.sidebar.success("✅ Theme file uploaded successfully!")
    
    # Check if theme file exists
    theme_file_exists = os.path.exists("game_themes.json")
    if not theme_file_exists:
        st.sidebar.warning("⚠️ Theme file not found. Please upload a theme file first.")
    
    # Main content - Tabs
    tab1, tab2, tab3 = st.tabs(["Upload & Process", "Summarize", "Results"])
    
    with tab1:
        st.header("Upload and Process Parquet Files")
        st.write("""
        Upload your Steam reviews Parquet files for processing. The app will:
        1. Check if each file's app ID exists in the theme dictionary
        2. Process only the files with matching app IDs
        3. Filter for English reviews and perform topic assignment
        """)
        
        # Parquet file uploader
        uploaded_files = st.file_uploader(
            "Upload Parquet Files",
            type=["parquet"],
            accept_multiple_files=True,
            help="Upload one or more Parquet files containing Steam reviews data"
        )
        
        if uploaded_files:
            st.write(f"📂 Uploaded {len(uploaded_files)} files")
            
            # Start processing button
            if st.button("🚀 Start Processing", key="process_button", disabled=not theme_file_exists):
                with st.spinner("Processing files..."):
                    # Record start time for performance comparison
                    start_time = time.time()
                    st.session_state.timing_data['process_start_time'] = start_time
                    
                    result = process_uploaded_files(uploaded_files)
                    
                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time
                    st.session_state.timing_data['process_end_time'] = time.time()
                    
                    if result:
                        st.session_state.result = result
                        st.success(f"✅ Processing completed in {elapsed_time:.2f} seconds!")
                        
                        # Display summary
                        st.subheader("Processing Summary")
                        st.write(f"✅ Processed {len(result['valid_files'])} files successfully")
                        st.write(f"⚠️ Skipped {len(result['skipped_files'])} files (app IDs not in theme dictionary)")
                        
                        # Show skipped files
                        if result['skipped_files']:
                            with st.expander("Show skipped files"):
                                for file_name, app_id in result['skipped_files']:
                                    st.write(f"- {file_name} (App ID: {app_id if app_id else 'Unknown'})")
                        
                        # Show sample of processed data
                        with st.expander("Show sample of processed data"):
                            st.dataframe(result['final_report'][['steam_appid', 'Theme', '#Reviews', 'LikeRatio']].head(10))
                            
                        # Switch to the summarize tab
                        st.info("👉 Go to the 'Summarize' tab to generate review summaries")
    
    with tab2:
        st.header("Generate Review Summaries")
        st.write("""
        This step uses GPU-accelerated summarization to generate concise summaries for each theme's reviews.
        This is computationally intensive and will use your GPU if available.
        """)
        
        # Check if we have a result from previous step
        if 'result' in st.session_state and st.session_state.result:
            # Start summarization button
            if st.button("🚀 Start Summarization", key="summarize_button"):
                with st.spinner("Generating summaries..."):
                    # Record start time for performance comparison
                    start_time = time.time()
                    st.session_state.timing_data['summarize_start_time'] = start_time
                    
                    summarized_report = summarize_report(st.session_state.result['final_report'])
                    
                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time
                    st.session_state.timing_data['summarize_end_time'] = time.time()
                    
                    if summarized_report is not None:
                        st.session_state.summarized_report = summarized_report
                        st.success(f"✅ Summarization completed in {elapsed_time:.2f} seconds!")
                        
                        # Show sample of summarized data
                        with st.expander("Show sample of summarized data"):
                            st.dataframe(summarized_report[['steam_appid', 'Theme', 'QuickSummary']].head(5))
                        
                        # Switch to results tab
                        st.info("👉 Go to the 'Results' tab to view the complete results")
        else:
            st.info("Please complete the 'Upload & Process' step first")
    
    with tab3:
        st.header("Results and Visualization")
        
        # Check if we have summarized results
        if 'summarized_report' in st.session_state and not st.session_state.summarized_report.empty:
            summarized_report = st.session_state.summarized_report
            
            # Display filters
            st.subheader("Filter Results")
            app_ids = sorted(summarized_report['steam_appid'].unique())
            selected_app_id = st.selectbox("Select App ID", app_ids)
            
            # Filter by app ID
            filtered_df = summarized_report[summarized_report['steam_appid'] == selected_app_id]
            
            # Display themes for this app ID
            themes = filtered_df['Theme'].unique()
            selected_theme = st.selectbox("Select Theme", themes)
            
            # Display the results for the selected theme
            theme_data = filtered_df[filtered_df['Theme'] == selected_theme]
            
            if not theme_data.empty:
                st.subheader(f"Theme: {selected_theme}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Number of Reviews", theme_data['#Reviews'].iloc[0])
                with col2:
                    st.metric("Like Ratio", theme_data['LikeRatio'].iloc[0])
                
                st.subheader("Theme Summary")
                st.write(theme_data['QuickSummary'].iloc[0])
                
                # Display sample reviews
                if 'Reviews' in theme_data.columns and isinstance(theme_data['Reviews'].iloc[0], list):
                    reviews = theme_data['Reviews'].iloc[0]
                    
                    with st.expander("Show sample reviews"):
                        for i, review in enumerate(reviews[:5]):  # Show first 5 reviews
                            st.write(f"**Review {i+1}**")
                            st.write(review[:300] + "..." if len(review) > 300 else review)
                            st.write("---")
            
            # Download button for the full report
            csv = summarized_report.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Report as CSV",
                data=csv,
                file_name='steamLensAI_sequential_report.csv',
                mime='text/csv',
            )
        elif 'result' in st.session_state and st.session_state.result:
            # We have processed data but no summarization
            final_report = st.session_state.result['final_report']
            
            st.info("Summarization has not been performed yet. Only basic data is available.")
            
            # Display filters
            st.subheader("Filter Results")
            app_ids = sorted(final_report['steam_appid'].unique())
            selected_app_id = st.selectbox("Select App ID", app_ids)
            
            # Filter by app ID
            filtered_df = final_report[final_report['steam_appid'] == selected_app_id]
            
            # Display themes for this app ID
            themes = filtered_df['Theme'].unique()
            selected_theme = st.selectbox("Select Theme", themes)
            
            # Display the results for the selected theme
            theme_data = filtered_df[filtered_df['Theme'] == selected_theme]
            
            if not theme_data.empty:
                st.subheader(f"Theme: {selected_theme}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Number of Reviews", theme_data['#Reviews'].iloc[0])
                with col2:
                    st.metric("Like Ratio", theme_data['LikeRatio'].iloc[0])
                
                # Display sample reviews
                if 'Reviews' in theme_data.columns and isinstance(theme_data['Reviews'].iloc[0], list):
                    reviews = theme_data['Reviews'].iloc[0]
                    
                    with st.expander("Show sample reviews"):
                        for i, review in enumerate(reviews[:5]):  # Show first 5 reviews
                            st.write(f"**Review {i+1}**")
                            st.write(review[:300] + "..." if len(review) > 300 else review)
                            st.write("---")
            
            # Download button for the full report
            csv = final_report.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Basic Report as CSV",
                data=csv,
                file_name='steamLensAI_sequential_report.csv',
                mime='text/csv',
            )
        else:
            st.info("Please complete the 'Upload & Process' step first")

    # Update global end time
    current_time = time.time()
    st.session_state.timing_data['global_end_time'] = current_time
    
    # Calculate total execution time
    global_elapsed_time = current_time - st.session_state.timing_data['global_start_time']
    
    # Display timing information
    st.sidebar.markdown("---")
    st.sidebar.header("⏱️ Execution Time Metrics")
    
    # Format the display of times
    def format_time(seconds):
        if seconds is None:
            return "Not completed"
        
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = seconds // 60
            remaining_seconds = seconds % 60
            return f"{int(minutes)} minutes, {remaining_seconds:.2f} seconds"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            remaining_seconds = seconds % 60
            return f"{int(hours)} hours, {int(minutes)} minutes, {remaining_seconds:.2f} seconds"
    
    # Calculate timing data only if operations were performed
    process_time = None
    if st.session_state.timing_data['process_start_time'] and st.session_state.timing_data['process_end_time']:
        process_time = st.session_state.timing_data['process_end_time'] - st.session_state.timing_data['process_start_time']
    
    summarize_time = None
    if st.session_state.timing_data['summarize_start_time'] and st.session_state.timing_data['summarize_end_time']:
        summarize_time = st.session_state.timing_data['summarize_end_time'] - st.session_state.timing_data['summarize_start_time']
    
    # Display the timing metrics
    st.sidebar.markdown(f"**Total Session Time:** {format_time(global_elapsed_time)}")
    st.sidebar.markdown(f"**Processing Time:** {format_time(process_time)}")
    st.sidebar.markdown(f"**Summarization Time:** {format_time(summarize_time)}")
    
    # If both processing and summarization were completed, show total analytical time
    if process_time and summarize_time:
        total_analytical_time = process_time + summarize_time
        st.sidebar.markdown(f"**Total Analytical Time:** {format_time(total_analytical_time)}")
    
    # Add a timestamp of when the analysis was performed
    if st.session_state.timing_data['global_end_time']:
        timestamp = datetime.datetime.fromtimestamp(st.session_state.timing_data['global_end_time']).strftime('%Y-%m-%d %H:%M:%S')
        st.sidebar.markdown(f"**Analysis Timestamp:** {timestamp}")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **steamLensAI (Sequential Version)** processes Steam reviews to identify themes and summarize sentiment.
    This version uses sequential processing for performance comparison with the distributed version.
    """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # If there's an unhandled exception, record the end time to still get timing data
        if 'timing_data' in st.session_state:
            st.session_state.timing_data['global_end_time'] = time.time()
        st.error(f"An error occurred: {str(e)}")
        raise e