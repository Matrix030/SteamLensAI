#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
steamLensAI.py - Big Data Processing for Steam Reviews with Streamlit Frontend
Dynamic resource allocation and hardware-optimized processing with interactive UI
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
import streamlit as st
from tqdm.auto import tqdm
from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
import dask.bag as db
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# Set page configuration
st.set_page_config(
    page_title="steamLensAI - Big Data Analysis",
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
    """Optimized worker for GPU processing"""
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
        if not reviews or not isinstance(reviews, list):
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
            # Process the review
            summary = hierarchical_summary(row['Reviews'])
            results.append((idx, summary))
            
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
        
        # Only check every 5 seconds to reduce overhead
        time.sleep(2)

def process_uploaded_files(uploaded_files, themes_file="game_themes.json"):
    """Process uploaded parquet files"""
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
        all_review_dfs = []
        
        # Create a progress bar for batch processing
        batch_progress = st.progress(0.0)
        
        # Process in dynamically sized batches
        for i in range(0, len(unique_app_ids), batch_size):
            batch_progress.progress(i / len(unique_app_ids))
            batch_app_ids = unique_app_ids[i:i+batch_size]
            
            # Filter data for this batch of app IDs
            batch_ddf = ddf_with_topic[ddf_with_topic['steam_appid'].isin(batch_app_ids)]
            
            # Aggregate for this batch
            agg = batch_ddf.groupby(['steam_appid', 'topic_id']).agg(
                review_count=('review', 'count'),
                likes_sum=('voted_up', 'sum')
            )
            
            # Collect reviews for this batch
            reviews_series = batch_ddf.groupby(['steam_appid', 'topic_id'])['review'] \
                .apply(lambda x: list(x), meta=('review', object))
            
            # Compute both in parallel
            agg_df, reviews_df = dd.compute(agg, reviews_series)
            
            # Convert to DataFrames
            agg_df = agg_df.reset_index()
            reviews_df = reviews_df.reset_index().rename(columns={'review': 'Reviews'})
            
            # Append to results
            all_agg_dfs.append(agg_df)
            all_review_dfs.append(reviews_df)
            
            status_text.write(f"Processed batch {i//batch_size + 1}/{(len(unique_app_ids) + batch_size - 1)//batch_size}")
        
        # Complete the batch progress
        batch_progress.progress(1.0)
        
        # Combine results
        status_text.write("Combining results...")
        agg_df = pd.concat(all_agg_dfs) if all_agg_dfs else pd.DataFrame()
        reviews_df = pd.concat(all_review_dfs) if all_review_dfs else pd.DataFrame()
        
        if agg_df.empty or reviews_df.empty:
            st.error("No data after processing. Please check your files and filters.")
            client.close()
            cluster.close()
            return None
        
        # Merge counts, likes, and reviews
        report_df = pd.merge(
            agg_df,
            reviews_df,
            on=['steam_appid', 'topic_id'],
            how='left'
        )
        
        # Build the final output structure
        status_text.write("Building final report...")
        rows = []
        for _, row in report_df.iterrows():
            appid = int(row['steam_appid'])
            tid = int(row['topic_id'])
            
            # Check if appid exists in GAME_THEMES
            if appid in GAME_THEMES:
                theme_keys = list(GAME_THEMES[appid].keys())
                # Check if tid is a valid index
                if tid < len(theme_keys):
                    theme_name = theme_keys[tid]
                else:
                    theme_name = f"Unknown Theme {tid}"
            else:
                theme_name = f"Unknown Theme {tid}"
            
            total = int(row['review_count'])
            likes = int(row['likes_sum'])
            like_ratio = f"{(likes / total * 100):.1f}%" if total > 0 else '0%'
            rows.append({
                'steam_appid': appid,
                'Theme': theme_name,
                '#Reviews': total,
                'LikeRatio': like_ratio,
                'Reviews': row['Reviews']
            })
        
        final_report = pd.DataFrame(rows)
        
        # Save intermediate results to avoid recomputation if summarization fails
        csv_path = 'output_csvs/SBERT_DD_new_report.csv'
        final_report.to_csv(csv_path, index=False)
        status_text.write(f"✅ Saved report to {csv_path}")
        
        client.close()
        cluster.close()
        
        # Complete the progress bar
        progress_bar.progress(1.0)
        status_text.write("✅ Data processing complete!")
        
        # Return skipped files info and final report
        return {
            'final_report': final_report,
            'valid_files': valid_files,
            'skipped_files': skipped_files,
            'dashboard_link': dashboard_link
        }

def summarize_report(final_report):
    """Hardware-optimized GPU summarization phase"""
    if final_report is None or final_report.empty:
        st.error("No report to summarize.")
        return None
    
    # Create progress indicators
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
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
    
    status_text.write(f"Starting optimized Dask cluster for summarization")
    cluster = LocalCluster(
        n_workers=HARDWARE_CONFIG['worker_count'], 
        threads_per_worker=2,
        memory_limit=HARDWARE_CONFIG['memory_per_worker']
    )
    client = Client(cluster)
    status_text.write(f"Dask dashboard initialized")
    
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
    status_text.write(f"Scheduling {n_workers} optimized partitions...")
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
        status_text.write("Computing with optimal settings...")
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
    summaries = [result[1] for result in all_results]
    
    # Store results
    final_report['QuickSummary'] = summaries
    
    # Report timing
    elapsed_time = time.time() - start_time
    status_text.write(f"✅ Optimized processing completed in {elapsed_time:.2f} seconds")
    status_text.write(f"Average time per item: {elapsed_time/len(final_report):.2f} seconds")
    
    # Save results
    output_path = 'output_csvs/optimized_hardware_report.csv'
    final_report.to_csv(output_path)
    status_text.write(f"✅ Results saved to {output_path}")
    
    # Clean up
    client.close()
    cluster.close()
    
    return final_report

def main():
    """Main Streamlit UI"""
    st.title("🎮 steamLensAI: Steam Reviews Analysis Tool")
    
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
                    result = process_uploaded_files(uploaded_files)
                    
                    if result:
                        st.session_state.result = result
                        st.success("✅ Processing completed successfully!")
                        
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
                            
                        # Link to Dask dashboard
                        st.markdown(f"[Open Dask Dashboard]({result['dashboard_link']})")
                        
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
                    summarized_report = summarize_report(st.session_state.result['final_report'])
                    
                    if summarized_report is not None:
                        st.session_state.summarized_report = summarized_report
                        st.success("✅ Summarization completed successfully!")
                        
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
                file_name='steamLensAI_report.csv',
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
                file_name='steamLensAI_basic_report.csv',
                mime='text/csv',
            )
        else:
            st.info("Please complete the 'Upload & Process' step first")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **steamLensAI** processes Steam reviews to identify themes and summarize sentiment.
    Developed with dynamic resource allocation and hardware-optimized processing.
    """)

if __name__ == "__main__":
    main()