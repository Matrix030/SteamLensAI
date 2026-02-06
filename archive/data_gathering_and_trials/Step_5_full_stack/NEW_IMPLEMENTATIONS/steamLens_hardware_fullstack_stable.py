#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
steamLensAI_sentiment_analysis.py - Big Data Processing for Steam Reviews with Sentiment Separation
Splits summaries into positive and negative review analyses for better insights
Includes comprehensive timing functionality and dashboard monitoring
"""

import os
import json
import numpy as np
import pandas as pd
import psutil
import torch
import dask
import threading
import time
import tempfile
import datetime
import streamlit as st
from tqdm.auto import tqdm
from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
import dask.bag as db
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import pyarrow.parquet as pq

# Set page configuration
st.set_page_config(
    page_title="steamLensAI - Sentiment Analysis",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create directories if they don't exist
os.makedirs('output_csvs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

def get_system_resources():
    """Dynamically determine system resources"""
    # Get available memory (in GB)
    total_memory = psutil.virtual_memory().total / (1024**3)
    # Get CPU count
    cpu_count = psutil.cpu_count(logical=False)  # Physical cores only
    if not cpu_count:
        cpu_count = psutil.cpu_count(logical=True)  # Logical if physical not available
    
    # Use 70% of available memory for Dask, split across workers
    dask_memory = int(total_memory * 0.7)
    # Determine optimal worker count (leave at least 1 core for system)
    worker_count = max(1, cpu_count - 1)
    # Memory per worker
    memory_per_worker = int(dask_memory / worker_count)
    
    return {
        'worker_count': worker_count,
        'memory_per_worker': memory_per_worker,
        'total_memory': total_memory
    }

def extract_appid_and_name_from_parquet(file_path):
    """Extract appid and game name from a parquet file (Optimized with PyArrow)"""
    try:
        pf = pq.ParquetFile(file_path)
        
        potential_name_fields = ['name', 'game_name', 'title', 'short_description', 'about_the_game']
        # Read schema to find available columns among potential_name_fields
        available_schema_fields = pf.schema.names
        columns_to_read = ['steam_appid'] + [col for col in potential_name_fields if col in available_schema_fields]

        # Read a small sample (e.g., first 1000 rows, or first few row groups) of necessary columns
        # This example reads the first non-empty row group, up to 1000 rows from it.
        # A more robust version might try a few row groups or read a fixed number of rows across groups.
        sample_df = None
        for i in range(min(5, pf.num_row_groups)): # Try up to 5 row groups
            try:
                # Attempt to read a slice of rows from the row group
                # read_row_group might be more memory efficient if we know row group sizes.
                # For simplicity, reading first N rows from the file using iter_batches.
                # Read up to 1000 rows for sampling
                batches = []
                for batch in pf.iter_batches(batch_size=100, columns=columns_to_read):
                    batches.append(batch)
                    if sum(len(b) for b in batches) >= 100: # Limit sample size for initial check
                        break
                if not batches:
                    continue

                sample_df = pd.concat([b.to_pandas() for b in batches])
                if not sample_df.empty:
                    break
            except Exception: # pylint: disable=broad-except
                # If a row group is problematic, try the next one
                continue
        
        if sample_df is None or sample_df.empty:
            st.warning(f"Could not read sample data from {file_path} to extract metadata.")
            return None, None
        
        app_id_series = sample_df['steam_appid'].dropna()
        if app_id_series.empty:
            return None, None
        
        # Get the most common app ID in case there are multiple
        app_id = app_id_series.mode()[0]
        
        game_name = None
        for field in potential_name_fields:
            if field in sample_df.columns and not sample_df[field].isnull().all():
                non_null_values = sample_df[field].dropna()
                if not non_null_values.empty:
                    game_name = non_null_values.mode()[0]
                    break
                    
        return int(app_id) if pd.notna(app_id) else None, game_name
    except Exception as e:
        st.error(f"Error extracting app ID and name (optimized) from file {os.path.basename(file_path)}: {str(e)}")
        return None, None

# This function maintains compatibility with existing code
def extract_appid_from_parquet(file_path):
    """Extract appid from a parquet file"""
    app_id, _ = extract_appid_and_name_from_parquet(file_path)
    return app_id

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

def estimate_file_size(file):
    """Estimate size of file in GB"""
    return file.size / (1024**3)  # Convert to GB

def assign_topic(df_partition, GAME_THEMES, embedder):
    """Assign topics using only theme embeddings for app IDs in this partition"""
    # If no rows, return as-is
    if df_partition.empty:
        df_partition['topic_id'] = []
        return df_partition
    
    # Get unique app IDs in this partition
    app_ids = df_partition['steam_appid'].unique().tolist()
    app_ids = [int(appid) for appid in app_ids]
    
    # Get embeddings only for app IDs in this partition
    local_theme_embeddings = get_theme_embeddings(app_ids, GAME_THEMES, embedder)
    
    reviews = df_partition['review'].tolist()
    # Compute embeddings in one go with batching
    review_embeds = embedder.encode(reviews, convert_to_numpy=True, batch_size=64)
    
    # Assign each review to its game-specific theme
    topic_ids = []
    for idx, appid in enumerate(df_partition['steam_appid']):
        appid = int(appid)
        if appid in local_theme_embeddings:
            theme_embs = local_theme_embeddings[appid]
            sims = cosine_similarity(review_embeds[idx:idx+1], theme_embs)
            topic_ids.append(int(sims.argmax()))
        else:
            # Default topic if theme embeddings not available
            topic_ids.append(0)
    
    df_partition['topic_id'] = topic_ids
    return df_partition

def prepare_partition(start_idx, end_idx, final_report):
    """Prepare a partition optimized for high-end hardware"""
    return final_report.iloc[start_idx:end_idx].copy()

def process_partition(partition_df, worker_id, HARDWARE_CONFIG):
    """Optimized worker for GPU processing with sentiment separation"""
    # Import needed packages
    from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
    import torch
    
    # Load model components with optimal settings
    print(f"Worker {worker_id} initializing with optimized settings")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(HARDWARE_CONFIG['model_name'])
    
    # Load model with optimized settings
    model = AutoModelForSeq2SeqLM.from_pretrained(
        HARDWARE_CONFIG['model_name'],
        torch_dtype=torch.float16,        # Half precision for speed
        device_map="auto",                # Automatic device placement
        low_cpu_mem_usage=True            # Optimized memory usage
    )
    
    # Create optimized pipeline
    summarizer = pipeline(
        task='summarization',
        model=model,
        tokenizer=tokenizer,
        framework='pt',
        model_kwargs={
            "use_cache": True,            # Enable caching for speed
            "return_dict_in_generate": True  # More efficient generation
        }
    )
    
    # Report GPU status if available
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"Worker {worker_id}: GPU Memory: {gpu_mem:.2f}GB allocated")
    
    # Highly optimized batch processing function
    def process_chunks_batched(chunks):
        """Process chunks in large batches for GPU"""
        all_summaries = []
        
        # Use large batches for the GPU
        for i in range(0, len(chunks), HARDWARE_CONFIG['gpu_batch_size']):
            batch = chunks[i:i+HARDWARE_CONFIG['gpu_batch_size']]
            batch_summaries = summarizer(
                batch,
                max_length=60,
                min_length=20,
                truncation=True,
                do_sample=False,
                num_beams=2  # Use beam search for better quality with minimal speed impact
            )
            all_summaries.extend([s["summary_text"] for s in batch_summaries])
            
            # Minimal cleanup - only when really needed
            if i % (HARDWARE_CONFIG['gpu_batch_size'] * 3) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                    
        return all_summaries
    
    # Optimized hierarchical summary function
    def hierarchical_summary(reviews):
        """Create hierarchical summary with optimized chunk sizes"""
        # Handle edge cases efficiently
        if not reviews or not isinstance(reviews, list) or len(reviews) == 0:
            return "No reviews available for summarization."
        
        # Fast path for small review sets
        if len(reviews) <= HARDWARE_CONFIG['chunk_size']:
            doc = "\n\n".join(reviews)
            return summarizer(
                doc,
                max_length=60,
                min_length=20,
                truncation=True,
                do_sample=False
            )[0]['summary_text']
        
        # Process larger review sets with optimized chunking
        all_chunks = []
        for i in range(0, len(reviews), HARDWARE_CONFIG['chunk_size']):
            batch = reviews[i:i+HARDWARE_CONFIG['chunk_size']]
            text = "\n\n".join(batch)
            all_chunks.append(text)
        
        # Process chunks with optimized batching
        intermediate_summaries = process_chunks_batched(all_chunks)
        
        # Create final summary
        joined = " ".join(intermediate_summaries)
        return summarizer(
            joined,
            max_length=60,
            min_length=20,
            truncation=True,
            do_sample=False
        )[0]['summary_text']
    
    # Process the partition with minimal overhead
    results = []
    
    # Use tqdm for progress tracking
    with tqdm(total=len(partition_df), desc=f"Worker {worker_id}", position=worker_id) as pbar:
        for idx, row in partition_df.iterrows():
            # Process the reviews by sentiment
            positive_reviews = row['Positive_Reviews'] if isinstance(row['Positive_Reviews'], list) else []
            negative_reviews = row['Negative_Reviews'] if isinstance(row['Negative_Reviews'], list) else []
            
            # Generate summaries for both positive and negative reviews
            positive_summary = hierarchical_summary(positive_reviews) if positive_reviews else "No positive reviews available."
            negative_summary = hierarchical_summary(negative_reviews) if negative_reviews else "No negative reviews available."
            
            results.append((idx, positive_summary, negative_summary))
            
            # Minimal cleanup - only every N iterations
            if len(results) % HARDWARE_CONFIG['cleanup_frequency'] == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Update progress bar
            pbar.update(1)
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Worker {worker_id} completed successfully")
    return results

def update_main_progress(futures, progress_bar, stop_flag, final_report_length):
    """Thread function to update the main progress bar"""
    while not stop_flag[0]:
        # Count completed futures
        completed_count = sum(f.status == 'finished' for f in futures)
        completed_percentage = completed_count / len(futures)
        
        # Update progress bar
        progress_bar.progress(completed_percentage)
        
        # Only check every 2 seconds to reduce overhead
        time.sleep(2)

def process_uploaded_files(uploaded_files, themes_file="game_themes.json"):
    """Process uploaded parquet files with timing metrics and sentiment separation"""
    # Start phase timer for more detailed timing
    phase_start_time = time.time()
    
    if not uploaded_files:
        st.warning("Please upload at least one Parquet file to begin processing.")
        return None
    
    # Create progress indicators
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    dashboard_placeholder = st.empty()  # Add placeholder for dashboard link
    
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
        game_name_mapping = {}  # Store mapping of app_id to game_name
        
        # Check and save valid files
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            
            # Save the uploaded file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Extract app ID and game name and check if it's in the theme dictionary
            app_id, game_name = extract_appid_and_name_from_parquet(file_path)
            
            if app_id and app_id in GAME_THEMES:
                valid_files.append((file_path, app_id))
                if game_name:
                    game_name_mapping[app_id] = game_name
                status_text.write(f"✅ File '{uploaded_file.name}' has app ID {app_id} - Processing")
            else:
                skipped_files.append((uploaded_file.name, app_id))
                status_text.write(f"⚠️ File '{uploaded_file.name}' has app ID {app_id} - Skipping (not in theme dictionary)")
        
        # Check if we have any valid files
        if not valid_files:
            st.error("No valid files to process. All uploaded files' app IDs were not found in the theme dictionary.")
            return None
        
        status_text.write(f"Starting processing with {len(valid_files)} valid files...")
        
        # Dynamic resource allocation
        resources = get_system_resources()
        status_text.write(f"System has {resources['total_memory']:.1f}GB memory and {resources['worker_count']} CPU cores")
        status_text.write(f"Allocating {resources['worker_count']} workers with {resources['memory_per_worker']}GB each")

        # Start a local Dask cluster with dynamically determined resources
        cluster = LocalCluster(
            n_workers=resources['worker_count'],
            threads_per_worker=2,
            memory_limit=f"{resources['memory_per_worker']}GB"
        )
        client = Client(cluster)
        dashboard_link = client.dashboard_link
        status_text.write(f"Dask dashboard initialized")
        
        # Immediately display dashboard link for monitoring
        with dashboard_placeholder.container():
            st.success("✅ Dask Processing Cluster Ready")
            st.markdown(f"**[Open Dask Dashboard]({dashboard_link})** (opens in new tab)")
            st.info("👁️ You can monitor processing tasks in real-time using this dashboard")
        
        # Store dashboard link in session state for later reference
        if 'process_dashboard_link' not in st.session_state:
            st.session_state.process_dashboard_link = dashboard_link
        
        # Initialize SBERT embedder
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        status_text.write("Initialized sentence embedder: all-MiniLM-L6-v2")
        
        # Process each valid file
        all_ddfs = []
        
        for file_path, app_id in valid_files:
            file_name = os.path.basename(file_path)
            status_text.write(f"Processing file: {file_name} (App ID: {app_id})")
            
            # Determine blocksize based on file size
            file_size = os.path.getsize(file_path) / (1024**3)  # in GB
            if file_size > 1.0:
                blocksize = '16MB'
            elif file_size > 0.1:
                blocksize = '32MB'
            else:
                blocksize = '64MB'
            
            status_text.write(f"Using blocksize: {blocksize} for {file_size:.2f}GB file")
            
            # Read the parquet file with Dask
            try:
                ddf = dd.read_parquet(
                    file_path,
                    columns=['steam_appid', 'review', 'review_language', 'voted_up'],
                    blocksize=blocksize
                )
                
                # Filter & Clean Data
                ddf = ddf[ddf['review_language'] == 'english']
                ddf = ddf.dropna(subset=['review'])
                
                # Only include matching app_id
                ddf = ddf[ddf['steam_appid'] == app_id]
                
                all_ddfs.append(ddf)
                status_text.write(f"Added file to processing queue: {file_name}")
                
            except Exception as e:
                status_text.write(f"⚠️ Error processing file {file_name}: {str(e)}")
        
        # Combine all dataframes
        if not all_ddfs:
            st.error("No data to process after filtering. Please check your files.")
            client.close()
            cluster.close()
            return None
        
        status_text.write("Combining all valid data for processing...")
        combined_ddf = dd.concat(all_ddfs)
        
        # Apply topic assignment
        status_text.write("Assigning topics to reviews...")
        meta = combined_ddf._meta.assign(topic_id=np.int64())
        
        # Create a partial function that includes GAME_THEMES and embedder
        def assign_topic_with_context(df_partition):
            return assign_topic(df_partition, GAME_THEMES, embedder)
        
        ddf_with_topic = combined_ddf.map_partitions(assign_topic_with_context, meta=meta)
        
        # Get unique app IDs
        unique_app_ids = combined_ddf['steam_appid'].unique().compute()
        total_app_ids = len(unique_app_ids)
        
        # Dynamically determine batch size based on number of app IDs and memory
        if total_app_ids > 1000:  # Very large number of app IDs
            batch_size = 3
        elif total_app_ids > 500:  # Medium-large number
            batch_size = 5
        elif total_app_ids > 100:  # Medium number
            batch_size = 10
        else:  # Smaller number
            batch_size = 20
        
        status_text.write(f"Processing {total_app_ids} unique app IDs with batch size {batch_size}")
        
        # Initialize empty dataframes for results
        all_agg_dfs = []
        all_positive_review_dfs = []
        all_negative_review_dfs = []
        
        # Create a progress bar for batch processing
        batch_progress = st.progress(0.0)
        
        # Process in dynamically sized batches
        for i in range(0, len(unique_app_ids), batch_size):
            batch_progress.progress(i / len(unique_app_ids))
            batch_app_ids = unique_app_ids[i:i+batch_size]
            
            # Filter data for this batch of app IDs
            batch_ddf = ddf_with_topic[ddf_with_topic['steam_appid'].isin(batch_app_ids)]
            
            # Aggregate for this batch - Fixed version without lambda
            agg = batch_ddf.groupby(['steam_appid', 'topic_id']).agg(
                review_count=('review', 'count'),
                likes_sum=('voted_up', 'sum')
            )
            
            # Collect positive reviews for this batch
            positive_reviews_series = batch_ddf[batch_ddf['voted_up']].groupby(['steam_appid', 'topic_id'])['review'] \
                .apply(lambda x: list(x), meta=('review', object))
            
            # Collect negative reviews for this batch
            negative_reviews_series = batch_ddf[~batch_ddf['voted_up']].groupby(['steam_appid', 'topic_id'])['review'] \
                .apply(lambda x: list(x), meta=('review', object))
            
            # Compute all in parallel
            agg_df, positive_reviews_df, negative_reviews_df = dd.compute(
                agg, positive_reviews_series, negative_reviews_series
            )
            
            # Convert to DataFrames
            agg_df = agg_df.reset_index()
            
            # Calculate dislikes after computation (total reviews - likes)
            agg_df['dislikes_sum'] = agg_df['review_count'] - agg_df['likes_sum']
            
            # Process positive reviews
            if not positive_reviews_df.empty:
                positive_reviews_df = positive_reviews_df.reset_index().rename(columns={'review': 'Positive_Reviews'})
                all_positive_review_dfs.append(positive_reviews_df)
            
            # Process negative reviews
            if not negative_reviews_df.empty:
                negative_reviews_df = negative_reviews_df.reset_index().rename(columns={'review': 'Negative_Reviews'})
                all_negative_review_dfs.append(negative_reviews_df)
            
            # Append to aggregation results
            all_agg_dfs.append(agg_df)
            
            status_text.write(f"Processed batch {i//batch_size + 1}/{(len(unique_app_ids) + batch_size - 1)//batch_size}")
        
        # Complete the batch progress
        batch_progress.progress(1.0)
        
        # Combine results
        status_text.write("Combining results...")
        agg_df = pd.concat(all_agg_dfs) if all_agg_dfs else pd.DataFrame()
        
        # Check if we have any data
        if agg_df.empty:
            st.error("No data after processing. Please check your files and filters.")
            client.close()
            cluster.close()
            return None
        
        # Process positive reviews
        positive_reviews_df = pd.concat(all_positive_review_dfs) if all_positive_review_dfs else pd.DataFrame()
        
        # Process negative reviews
        negative_reviews_df = pd.concat(all_negative_review_dfs) if all_negative_review_dfs else pd.DataFrame()
        
        # Merge counts and sentiment-specific reviews
        # First, merge aggregation with positive reviews
        if not positive_reviews_df.empty:
            report_df = pd.merge(
                agg_df,
                positive_reviews_df,
                on=['steam_appid', 'topic_id'],
                how='left'
            )
        else:
            report_df = agg_df.copy()
            report_df['Positive_Reviews'] = None
        
        # Then, merge with negative reviews
        if not negative_reviews_df.empty:
            report_df = pd.merge(
                report_df,
                negative_reviews_df,
                on=['steam_appid', 'topic_id'],
                how='left'
            )
        else:
            report_df['Negative_Reviews'] = None
        
        # Build the final output structure
        status_text.write("Building final report with sentiment analysis...")

        # Helper function to get theme name safely
        def get_theme_name_safely(row, themes_dict):
            appid = int(row['steam_appid'])
            tid = int(row['topic_id'])
            if appid in themes_dict:
                theme_keys = list(themes_dict[appid].keys())
                if 0 <= tid < len(theme_keys):
                    return theme_keys[tid]
                else:
                    return f"Unknown Theme {tid} (Index out of bounds)"
            else:
                return f"Unknown Theme {tid} (AppID not in themes)"

        if not report_df.empty:
            # Apply the helper function to create the 'Theme' column
            report_df['Theme'] = report_df.apply(lambda row: get_theme_name_safely(row, GAME_THEMES), axis=1)

            # Vectorized calculations for ratios
            report_df['#Reviews'] = report_df['review_count'].astype(int)
            report_df['Positive'] = report_df['likes_sum'].astype(int)
            report_df['Negative'] = report_df['dislikes_sum'].astype(int)
            
            # Ensure '#Reviews' is not zero to avoid division by zero errors
            report_df['LikeRatio'] = '0.0%'
            report_df.loc[report_df['#Reviews'] > 0, 'LikeRatio'] = \
                (report_df['Positive'] / report_df['#Reviews'] * 100).round(1).astype(str) + '%'
            
            report_df['DislikeRatio'] = '0.0%'
            report_df.loc[report_df['#Reviews'] > 0, 'DislikeRatio'] = \
                (report_df['Negative'] / report_df['#Reviews'] * 100).round(1).astype(str) + '%'
            
            # Select and rename columns for the final report
            final_report = report_df[[
                'steam_appid',
                'Theme',
                '#Reviews',
                'Positive',
                'Negative',
                'LikeRatio',
                'DislikeRatio',
                'Positive_Reviews', # Already exists from merge, ensure it's correctly populated
                'Negative_Reviews'  # Already exists from merge, ensure it's correctly populated
            ]].copy()
            final_report['steam_appid'] = final_report['steam_appid'].astype(int)
        else:
            # If report_df is empty, create an empty final_report with correct columns
            final_report = pd.DataFrame(columns=[
                'steam_appid', 'Theme', '#Reviews', 'Positive', 'Negative',
                'LikeRatio', 'DislikeRatio', 'Positive_Reviews', 'Negative_Reviews'
            ])

        # Ensure Positive_Reviews and Negative_Reviews are lists, fillna with empty list if they are None
        # This was previously handled in the loop, ensure it here for consistency
        if 'Positive_Reviews' in final_report.columns:
            final_report['Positive_Reviews'] = final_report['Positive_Reviews'].apply(lambda x: x if isinstance(x, list) else [])
        else:
            final_report['Positive_Reviews'] = pd.Series([[] for _ in range(len(final_report))], index=final_report.index)
            
        if 'Negative_Reviews' in final_report.columns:
            final_report['Negative_Reviews'] = final_report['Negative_Reviews'].apply(lambda x: x if isinstance(x, list) else [])
        else:
            final_report['Negative_Reviews'] = pd.Series([[] for _ in range(len(final_report))], index=final_report.index)
        
        # Save intermediate results to avoid recomputation if summarization fails
        csv_path = 'output_csvs/sentiment_report.csv'
        final_report.to_csv(csv_path, index=False)
        status_text.write(f"✅ Saved sentiment report to {csv_path}")
        
        client.close()
        cluster.close()
        
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
            'skipped_files': skipped_files,
            'dashboard_link': dashboard_link,
            'processing_time': phase_elapsed_time,
            'game_name_mapping': game_name_mapping  # Add the game name mapping to the return value
        }

def summarize_report(final_report):
    """Hardware-optimized GPU summarization phase with sentiment separation"""
    # Start phase timer for detailed timing
    phase_start_time = time.time()
    
    if final_report is None or final_report.empty:
        st.error("No report to summarize.")
        return None
    
    # Create progress indicators
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    dashboard_placeholder = st.empty()  # Add placeholder for dashboard link
    
    with progress_placeholder.container():
        status_text = st.empty()
        progress_bar = st.progress(0.0)
    
    # Hardware-optimized configuration - this can be adjusted based on actual hardware
    HARDWARE_CONFIG = {
        'worker_count': 6,                # Default for high-end CPU
        'memory_per_worker': '3GB',       # Adjust based on available RAM
        'gpu_batch_size': 96,             # Adjust based on GPU VRAM
        'model_name': 'sshleifer/distilbart-cnn-12-6',  # Efficient summarization model
        'chunk_size': 400,                # Chunk size for processing reviews
        'checkpoint_frequency': 25,       # Frequency of checkpoints
        'cleanup_frequency': 10,          # Frequency of memory cleanup
    }
    
    status_text.write(f"Starting optimized Dask cluster for sentiment-based summarization")
    cluster = LocalCluster(
        n_workers=HARDWARE_CONFIG['worker_count'], 
        threads_per_worker=2,
        memory_limit=HARDWARE_CONFIG['memory_per_worker']
    )
    client = Client(cluster)
    status_text.write(f"Dask dashboard initialized")
    
    # Immediately display dashboard link for monitoring
    with dashboard_placeholder.container():
        st.success("✅ Dask Summarization Cluster Ready")
        st.markdown(f"**[Open Dask Dashboard]({client.dashboard_link})** (opens in new tab)")
        st.info("👁️ Monitor summarization tasks in real-time with this dashboard")
    
    # Store dashboard link in session state for later reference
    if 'summarize_dashboard_link' not in st.session_state:
        st.session_state.summarize_dashboard_link = client.dashboard_link
    
    # Determine optimal partition sizes - larger for better throughput
    n_workers = HARDWARE_CONFIG['worker_count']
    partition_size = len(final_report) // n_workers
    partitions = []
    for i in range(n_workers):
        start_idx = i * partition_size
        end_idx = (i + 1) * partition_size if i < n_workers - 1 else len(final_report)
        partitions.append(dask.delayed(prepare_partition)(start_idx, end_idx, final_report))
        status_text.write(f"Prepared partition {i+1} with {end_idx-start_idx} items")
    
    # Schedule tasks
    status_text.write(f"Scheduling {n_workers} optimized partitions for sentiment analysis...")
    delayed_results = []
    for i in range(n_workers):
        delayed_result = dask.delayed(process_partition)(partitions[i], i, HARDWARE_CONFIG)
        delayed_results.append(delayed_result)
    
    # Start timing
    start_time = time.time()
    
    # Submit tasks to cluster
    futures = client.compute(delayed_results)
    
    # Start progress monitor with minimal overhead
    stop_flag = [False]  # Use a list to make it mutable for the thread
    monitor_thread = threading.Thread(
        target=update_main_progress, 
        args=(futures, progress_bar, stop_flag, len(final_report))
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Wait for computation
    try:
        status_text.write("Computing sentiment-based summaries with optimal settings...")
        results = client.gather(futures)
    except Exception as e:
        status_text.write(f"Error with futures: {e}")
        status_text.write("Falling back to direct computation...")
        results = dask.compute(*delayed_results)
    
    # Stop progress monitor
    stop_flag[0] = True
    monitor_thread.join(timeout=3)
    
    # Update progress to completion
    progress_bar.progress(1.0)
    
    # Process results efficiently
    all_results = []
    for worker_results in results:
        all_results.extend(worker_results)
    
    # Sort results
    all_results.sort(key=lambda x: x[0])
    
    # Extract positive and negative summaries
    positive_summaries = [result[1] for result in all_results]
    negative_summaries = [result[2] for result in all_results]
    
    # Store results
    final_report['Positive_Summary'] = positive_summaries
    final_report['Negative_Summary'] = negative_summaries
    
    # Report timing
    elapsed_time = time.time() - start_time
    status_text.write(f"✅ Optimized sentiment analysis completed in {elapsed_time:.2f} seconds")
    status_text.write(f"Average time per item: {elapsed_time/len(final_report):.2f} seconds")
    
    # Save results
    output_path = 'output_csvs/sentiment_summaries.csv'
    final_report.to_csv(output_path)
    status_text.write(f"✅ Results saved to {output_path}")
    
    # Calculate elapsed time for this phase
    phase_elapsed_time = time.time() - phase_start_time
    formatted_time = str(datetime.timedelta(seconds=int(phase_elapsed_time)))
    status_text.write(f"✅ Total summarization time: {formatted_time}")
    
    # Clean up
    client.close()
    cluster.close()
    
    return final_report

def main():
    """Main Streamlit UI with comprehensive timing and sentiment analysis"""
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
    
    st.title("🎮 steamLensAI: Sentiment Analysis for Steam Reviews")
    st.write("This tool separates positive and negative reviews for deeper insights into what players love and hate.")
    
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
        4. Separate positive and negative reviews for each theme
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
                with st.spinner("Initializing Dask cluster..."):
                    # Record start time for performance comparison
                    start_time = time.time()
                    st.session_state.timing_data['process_start_time'] = start_time
                    
                    # Inform user that processing is starting
                    st.info("Dask cluster is being initialized. Processing will start shortly...")
                
                # Process the files (note: moved outside the spinner to allow dashboard to show)
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
                        sample_df = result['final_report'][['steam_appid', 'Theme', '#Reviews', 'Positive', 'Negative', 'LikeRatio', 'DislikeRatio']].head(10)
                        st.dataframe(sample_df)
                    
                    # Switch to the summarize tab
                    st.info("👉 Go to the 'Summarize' tab to generate sentiment-based summaries")
    
    with tab2:
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
                elapsed_time = time.time() - start_time
                st.session_state.timing_data['summarize_end_time'] = time.time()
                
                if summarized_report is not None:
                    st.session_state.summarized_report = summarized_report
                    st.success(f"✅ Sentiment summarization completed in {elapsed_time:.2f} seconds!")
                    
                    # Show sample of summarized data
                    with st.expander("Show sample of sentiment summaries"):
                        sample_columns = ['steam_appid', 'Theme', 'Positive_Summary', 'Negative_Summary']
                        st.dataframe(summarized_report[sample_columns].head(5))
                    
                    # Switch to results tab
                    st.info("👉 Go to the 'Results' tab to view the complete sentiment analysis")
        else:
            st.info("Please complete the 'Upload & Process' step first")
    
    with tab3:
        st.header("Sentiment Analysis Results")
        
        # Check if we have summarized results
        if 'summarized_report' in st.session_state and not st.session_state.summarized_report.empty:
            summarized_report = st.session_state.summarized_report
            
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
                    st.subheader("Sample Positive Reviews")
                    if 'Positive_Reviews' in theme_data.columns and isinstance(theme_data['Positive_Reviews'].iloc[0], list) and theme_data['Positive_Reviews'].iloc[0]:
                        positive_reviews = theme_data['Positive_Reviews'].iloc[0]
                        with st.expander(f"Show {min(5, len(positive_reviews))} positive reviews"):
                            for i, review in enumerate(positive_reviews[:5]):  # Show first 5 reviews
                                st.write(f"**Review {i+1}**")
                                st.write(review[:300] + "..." if len(review) > 300 else review)
                                st.write("---")
                    else:
                        st.write("No positive reviews available.")
                
                with col2:
                    st.subheader("Sample Negative Reviews")
                    if 'Negative_Reviews' in theme_data.columns and isinstance(theme_data['Negative_Reviews'].iloc[0], list) and theme_data['Negative_Reviews'].iloc[0]:
                        negative_reviews = theme_data['Negative_Reviews'].iloc[0]
                        with st.expander(f"Show {min(5, len(negative_reviews))} negative reviews"):
                            for i, review in enumerate(negative_reviews[:5]):  # Show first 5 reviews
                                st.write(f"**Review {i+1}**")
                                st.write(review[:300] + "..." if len(review) > 300 else review)
                                st.write("---")
                    else:
                        st.write("No negative reviews available.")
            
            # Download buttons for the reports
            st.subheader("Download Reports")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = summarized_report.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Sentiment Report (CSV)",
                    data=csv,
                    file_name='steamLensAI_sentiment_report.csv',
                    mime='text/csv',
                )
            
            with col2:
                # Create a simplified report for easier viewing
                simple_report = summarized_report[['steam_appid', 'Theme', '#Reviews', 'Positive', 'Negative', 
                                                  'LikeRatio', 'DislikeRatio', 'Positive_Summary', 'Negative_Summary']]
                simple_csv = simple_report.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Simplified Report (CSV)",
                    data=simple_csv,
                    file_name='steamLensAI_sentiment_summary.csv',
                    mime='text/csv',
                )
            
        elif 'result' in st.session_state and st.session_state.result:
            # We have processed data but no summarization
            final_report = st.session_state.result['final_report']
            
            # Get game name mapping if available
            game_name_mapping = {}
            if 'game_name_mapping' in st.session_state.result:
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
                    st.subheader("Sample Positive Reviews")
                    if 'Positive_Reviews' in theme_data.columns and isinstance(theme_data['Positive_Reviews'].iloc[0], list) and theme_data['Positive_Reviews'].iloc[0]:
                        positive_reviews = theme_data['Positive_Reviews'].iloc[0]
                        with st.expander(f"Show {min(5, len(positive_reviews))} positive reviews"):
                            for i, review in enumerate(positive_reviews[:5]):  # Show first 5 reviews
                                st.write(f"**Review {i+1}**")
                                st.write(review[:300] + "..." if len(review) > 300 else review)
                                st.write("---")
                    else:
                        st.write("No positive reviews available.")
                
                with col2:
                    st.subheader("Sample Negative Reviews")
                    if 'Negative_Reviews' in theme_data.columns and isinstance(theme_data['Negative_Reviews'].iloc[0], list) and theme_data['Negative_Reviews'].iloc[0]:
                        negative_reviews = theme_data['Negative_Reviews'].iloc[0]
                        with st.expander(f"Show {min(5, len(negative_reviews))} negative reviews"):
                            for i, review in enumerate(negative_reviews[:5]):  # Show first 5 reviews
                                st.write(f"**Review {i+1}**")
                                st.write(review[:300] + "..." if len(review) > 300 else review)
                                st.write("---")
                    else:
                        st.write("No negative reviews available.")
                
                st.subheader("Get Summaries")
                st.info("👉 Go to the 'Summarize' tab to generate sentiment-based summaries for what players love and dislike about each theme.")
            
            # Download button for the basic report
            csv = final_report.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Basic Sentiment Report (CSV)",
                data=csv,
                file_name='steamLensAI_basic_sentiment_report.csv',
                mime='text/csv',
            )
        else:
            st.info("Please complete the 'Upload & Process' step first")

    # Display dashboard links if available
    st.sidebar.markdown("---")
    st.sidebar.header("🔗 Dask Dashboards")
    
    if 'process_dashboard_link' in st.session_state:
        st.sidebar.markdown(f"**[Processing Dashboard]({st.session_state.process_dashboard_link})** ✨")
    
    if 'summarize_dashboard_link' in st.session_state:
        st.sidebar.markdown(f"**[Summarization Dashboard]({st.session_state.summarize_dashboard_link})** ✨")
    
    if 'process_dashboard_link' not in st.session_state and 'summarize_dashboard_link' not in st.session_state:
        st.sidebar.info("Dashboard links will appear here once you start processing or summarization.")
    
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
    **steamLensAI Sentiment Analysis** processes Steam reviews to identify what players love and hate about each theme.
    Separating positive and negative reviews provides clearer insights into player sentiment.
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