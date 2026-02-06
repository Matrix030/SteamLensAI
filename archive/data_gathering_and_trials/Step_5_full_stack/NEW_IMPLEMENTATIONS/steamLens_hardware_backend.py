#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
steamLensAI.py - Big Data Processing for Steam Reviews
Dynamic resource allocation and hardware-optimized processing
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
from tqdm.auto import tqdm
from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
import dask.bag as db
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

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

def estimate_dataset_size(path):
    """Estimate size of dataset in GB"""
    total_size = 0
    for file in os.listdir(path):
        if file.endswith('.parquet'):
            file_path = os.path.join(path, file)
            total_size += os.path.getsize(file_path)
    return total_size / (1024**3)  # Convert to GB

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

def update_main_progress(futures, main_progress, stop_flag, final_report_length):
    """Thread function to update the main progress bar"""
    while not stop_flag[0]:
        # Count completed futures
        completed_count = sum(f.status == 'finished' for f in futures)
        completed_percentage = completed_count / len(futures)
        
        # Update progress bar
        main_progress.n = int(final_report_length * completed_percentage)
        main_progress.refresh()
        
        # Only check every 5 seconds to reduce overhead
        time.sleep(5)

def data_processing_phase(dataset_path="parquet_output_theme_combo", themes_file="game_themes.json"):
    """Main data processing phase"""
    # Dynamic resource allocation
    resources = get_system_resources()
    print(f"System has {resources['total_memory']:.1f}GB memory and {resources['worker_count']} CPU cores")
    print(f"Allocating {resources['worker_count']} workers with {resources['memory_per_worker']}GB each")

    # Start a local Dask cluster with dynamically determined resources
    cluster = LocalCluster(
        n_workers=resources['worker_count'],
        threads_per_worker=2,
        memory_limit=f"{resources['memory_per_worker']}GB"
    )
    client = Client(cluster)
    print(f"Dashboard link: {client.dashboard_link}")
    
    # Load per-game theme keywords
    try:
        with open(themes_file, 'r') as f:
            raw = json.load(f)
        GAME_THEMES = {int(appid): themes for appid, themes in raw.items()}
        print(f"Loaded theme dictionary with {len(GAME_THEMES)} games")
    except FileNotFoundError:
        print(f"Warning: {themes_file} not found. Creating dummy theme dictionary.")
        GAME_THEMES = {}
    
    # Initialize SBERT embedder
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("Initialized sentence embedder: all-MiniLM-L6-v2")
    
    # Estimate dataset size
    try:
        estimated_size = estimate_dataset_size(dataset_path)
        print(f"Estimated dataset size: {estimated_size:.2f}GB")
    except FileNotFoundError:
        print(f"Warning: Dataset path '{dataset_path}' not found. Using default blocksize.")
        estimated_size = 0
        
    # Dynamically determine blocksize based on dataset and memory
    if estimated_size > 100:  # Very large dataset
        blocksize = '16MB'
    elif estimated_size > 10:  # Medium-large dataset
        blocksize = '32MB'
    else:  # Smaller dataset
        blocksize = '64MB'
    
    print(f"Using dynamic blocksize: {blocksize}")
    
    # Read with dynamic blocksize
    try:
        ddf = dd.read_parquet(
            f'{dataset_path}/*.parquet',
            columns=['steam_appid', 'review', 'review_language', 'voted_up'],
            blocksize=blocksize
        )
        
        # Filter & Clean Data
        ddf = ddf[ddf['review_language'] == 'english']
        ddf = ddf.dropna(subset=['review'])
        
        # Apply to each partition; specify output metadata
        meta = ddf._meta.assign(topic_id=np.int64())
        
        # Create a partial function that includes GAME_THEMES and embedder
        def assign_topic_with_context(df_partition):
            return assign_topic(df_partition, GAME_THEMES, embedder)
        
        ddf_with_topic = ddf.map_partitions(assign_topic_with_context, meta=meta)
        
        # Get unique app IDs
        unique_app_ids = ddf['steam_appid'].unique().compute()
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
        
        print(f"Processing {total_app_ids} unique app IDs with batch size {batch_size}")
        
        # Initialize empty dataframes for results
        all_agg_dfs = []
        all_review_dfs = []
        
        # Process in dynamically sized batches
        for i in tqdm(range(0, len(unique_app_ids), batch_size)):
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
        
        # Combine results
        agg_df = pd.concat(all_agg_dfs)
        reviews_df = pd.concat(all_review_dfs)
        
        # Merge counts, likes, and reviews
        report_df = pd.merge(
            agg_df,
            reviews_df,
            on=['steam_appid', 'topic_id'],
            how='left'
        )
        
        # Build the final output structure
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
        final_report.to_csv('output_csvs/SBERT_DD_new_report.csv', index=False)
        
        # Print preview of the DataFrame (excluding the Reviews column as it contains lists)
        print("Final report preview (Reviews column contains lists of review texts):")
        print(final_report[['steam_appid', 'Theme', '#Reviews', 'LikeRatio']].head())
        
        # Verify that Reviews column contains lists
        sample_reviews = final_report['Reviews'].iloc[0]
        print(f"\nSample from first Reviews entry (showing first review only):")
        if isinstance(sample_reviews, list) and len(sample_reviews) > 0:
            print(f"Number of reviews in list: {len(sample_reviews)}")
            print(f"First review (truncated): {sample_reviews[0][:100]}...")
        
        client.close()
        cluster.close()
        
        return final_report
    
    except Exception as e:
        print(f"Error in data processing phase: {str(e)}")
        client.close()
        cluster.close()
        return None

def summarization_phase(final_report=None):
    """Hardware-optimized GPU summarization phase"""
    if final_report is None:
        # Try to load from saved file if final_report not provided
        try:
            final_report = pd.read_csv('output_csvs/SBERT_DD_new_report.csv')
            # Convert string representation of lists back to actual lists
            # This is necessary because CSV can't store lists directly
            print("Loaded report from CSV file. Processing list conversions...")
        except FileNotFoundError:
            print("No report file found. Cannot proceed with summarization.")
            return
    
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
    
    print(f"Starting optimized Dask cluster for summarization")
    cluster = LocalCluster(
        n_workers=HARDWARE_CONFIG['worker_count'], 
        threads_per_worker=2,
        memory_limit=HARDWARE_CONFIG['memory_per_worker']
    )
    client = Client(cluster)
    print(f"Dask dashboard available at: {client.dashboard_link}")
    
    # Determine optimal partition sizes - larger for better throughput
    n_workers = HARDWARE_CONFIG['worker_count']
    partition_size = len(final_report) // n_workers
    partitions = []
    for i in range(n_workers):
        start_idx = i * partition_size
        end_idx = (i + 1) * partition_size if i < n_workers - 1 else len(final_report)
        partitions.append(dask.delayed(prepare_partition)(start_idx, end_idx, final_report))
        print(f"Prepared partition {i+1} with {end_idx-start_idx} items")
    
    # Schedule tasks
    print(f"Scheduling {n_workers} optimized partitions...")
    delayed_results = []
    for i in range(n_workers):
        delayed_result = dask.delayed(process_partition)(partitions[i], i, HARDWARE_CONFIG)
        delayed_results.append(delayed_result)
    
    # Streamlined progress tracking
    print("\nStarting optimized computation...")
    main_progress = tqdm(total=len(final_report), desc="Overall Progress")
    
    # Start timing
    start_time = time.time()
    
    # Submit tasks to cluster
    futures = client.compute(delayed_results)
    
    # Start progress monitor with minimal overhead
    stop_flag = [False]  # Use a list to make it mutable for the thread
    monitor_thread = threading.Thread(
        target=update_main_progress, 
        args=(futures, main_progress, stop_flag, len(final_report))
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Wait for computation
    try:
        print("Computing with optimal settings...")
        results = client.gather(futures)
    except Exception as e:
        print(f"Error with futures: {e}")
        print("Falling back to direct computation...")
        results = dask.compute(*delayed_results)
    
    # Stop progress monitor
    stop_flag[0] = True
    monitor_thread.join(timeout=3)
    
    # Update progress to completion
    main_progress.n = len(final_report)
    main_progress.refresh()
    main_progress.close()
    
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
    print(f"\nOptimized processing completed in {elapsed_time:.2f} seconds")
    print(f"Average time per item: {elapsed_time/len(final_report):.2f} seconds")
    
    # Display results
    print("\nResults sample:")
    print(final_report[['steam_appid', 'Theme', 'QuickSummary']].head())
    
    # Save results
    final_report.to_csv('output_csvs/optimized_hardware_report.csv')
    print("Results saved to output_csvs/optimized_hardware_report.csv")
    
    # Clean up
    client.close()
    cluster.close()
    
    return final_report

def main():
    """Main execution function"""
    print("=" * 80)
    print("steamLensAI: Big Data Processing for Steam Reviews")
    print("=" * 80)
    
    # Ask user if they want to run data processing
    run_data_processing = input("Run data processing phase? (y/n, default: y): ").strip().lower()
    if run_data_processing in ('', 'y', 'yes'):
        print("\nStarting data processing phase...")
        final_report = data_processing_phase()
    else:
        final_report = None
    
    # Ask user if they want to run summarization
    run_summarization = input("Run GPU summarization phase? (y/n, default: y): ").strip().lower()
    if run_summarization in ('', 'y', 'yes'):
        print("\nStarting hardware-optimized summarization phase...")
        summarization_phase(final_report)
    
    print("\nsteamLensAI processing complete!")

if __name__ == "__main__":
    main()