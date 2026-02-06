#!/usr/bin/env python3

"""
Steam Reviews Processing Module

This module handles the distributed processing of Steam review data using Dask.
It performs the following main tasks:
1. Loads and validates uploaded parquet files containing Steam reviews
2. Distributes processing across multiple workers for performance
3. Assigns topics/themes to reviews using semantic similarity
4. Separates positive and negative reviews by sentiment
5. Aggregates results and saves them for summarization

Key Technologies Used:
- Dask: For distributed/parallel processing across multiple CPU cores
- Sentence Transformers: For converting text to numerical embeddings
- Pandas: For data manipulation and analysis
- PyTorch: For GPU acceleration when available
"""

import os
import gc
import time
import numpy as np
import pandas as pd
import tempfile
import datetime
import streamlit as st
import dask
import dask.dataframe as dd
from typing import Dict, List, Tuple, Any, Optional
from dask.distributed import Client, LocalCluster, as_completed
from sentence_transformers import SentenceTransformer
from pathlib import Path
import shutil
import torch
import pyarrow.parquet as pq
from dask import delayed
from collections.abc import Iterable

# Import configuration and utility modules
from ..config.app_config import (SENTENCE_TRANSFORMER_MODEL, PARQUET_COLUMNS, 
                                DEFAULT_LANGUAGE, DEFAULT_INTERIM_PATH, PROCESSING_CONFIG)
from ..data.data_loader import extract_appid_and_name_from_parquet, determine_blocksize
from ..utils.system_utils import get_system_resources
from .topic_assignment import assign_topic

# Try to import monitoring utilities (optional dependency)
try:
    from ..utils.dask_monitor import DaskMonitor, create_progress_tracker, update_progress_tracker, display_performance_summary
    MONITORING_AVAILABLE = True
except ImportError:
    # If monitoring utilities aren't available, create dummy functions
    MONITORING_AVAILABLE = False
    
    class DaskMonitor:
        """Dummy monitoring class when real monitoring isn't available"""
        def __init__(self, client):
            self.client = client
            self.start_time = time.time()
        
        def log_task_completion(self, *args, **kwargs):
            """Dummy method - does nothing"""
            pass
        
        def display_cluster_metrics(self, *args, **kwargs):
            """Dummy method - does nothing"""
            pass
        
        def get_performance_summary(self):
            """Return empty performance summary"""
            return {}
        
        def estimate_completion_time(self, completed, total):
            """Return None since we can't estimate"""
            return None
    
    def create_progress_tracker(total_chunks, placeholder):
        """Create basic progress tracking when full monitoring isn't available"""
        progress_bar = placeholder.progress(0.0)
        status_text = placeholder.empty()
        return {
            'progress_bar': progress_bar,
            'status_text': status_text,
            'chunks_metric': None,
            'rows_metric': None,
            'time_metric': None,
            'eta_metric': None
        }
    
    def update_progress_tracker(tracker, completed, total_chunks, total_rows, monitor):
        """Update basic progress when full monitoring isn't available"""
        progress = completed / total_chunks if total_chunks > 0 else 0
        tracker['progress_bar'].progress(progress)
        tracker['status_text'].write(f"Processing chunk {completed}/{total_chunks} ({total_rows:,} rows)")
    
    def display_performance_summary(monitor, container):
        """Display basic completion message when full monitoring isn't available"""
        container.success("Processing complete!")


def process_uploaded_files(uploaded_files: List[Any], themes_file: str = "game_themes.json") -> Optional[Dict[str, Any]]:
    """
    Main function to process uploaded Steam review files using distributed computing.
    
    This function orchestrates the entire processing pipeline:
    1. Validates uploaded files and loads game themes
    2. Sets up a Dask cluster for parallel processing
    3. Breaks files into chunks for distributed processing
    4. Assigns topics to reviews using machine learning
    5. Separates positive/negative reviews and aggregates results
    
    Args:
        uploaded_files: List of uploaded parquet files from Streamlit
        themes_file: Path to JSON file containing game themes/topics
        
    Returns:
        Dictionary containing processing results and metadata, or None if failed
    """
    
    # Record when this processing phase started (for timing metrics)
    phase_start_time = time.time()
    
    # Validate inputs - make sure we have files to process
    if not uploaded_files:
        st.warning("Please upload at least one Parquet file to begin processing.")
        return None
    
    # Create the user interface layout for progress monitoring
    # Split into two columns: main progress on left, cluster info on right
    progress_column, cluster_info_column = st.columns([3, 1])
    
    with progress_column:
        progress_display_area = st.empty()  # Area for progress bars and status
        main_status_area = st.empty()       # Area for status messages
    
    with cluster_info_column:
        cluster_dashboard_area = st.empty()  # Area for cluster dashboard link
        system_metrics_area = st.empty()     # Area for system information
    
    # Create status text display in the main status area
    with main_status_area.container():
        status_display = st.empty()
    
    # Step 1: Load the game themes dictionary
    # This maps game app IDs to their themes (like "gameplay", "graphics", etc.)
    try:
        from ..data.data_loader import load_theme_dictionary
        game_themes_dict = load_theme_dictionary(themes_file)
        if not game_themes_dict:
            return None
        status_display.write(f"✅ Loaded theme dictionary with {len(game_themes_dict)} games")
    except Exception as e:
        st.error(f"Error loading theme dictionary: {str(e)}")
        return None
    
    # Step 2: Create temporary storage for intermediate processing results  after each worker is done processing a file, it is stored here
    temp_storage_paths = create_temp_storage()
    status_display.write("✅ Created temporary storage for chunk processing")
    
    # Initialize variables for the Dask cluster (distributed computing system)
    dask_cluster = None
    dask_client = None
    
    try:
        # Get the processing configuration (how many workers, memory limits, etc.)
        processing_config = PROCESSING_CONFIG
        
        # Display system configuration to the user
        with system_metrics_area.container():
            from ..config.app_config import display_system_config
            st.info(display_system_config())
        
        # Step 3: Create a Dask cluster for distributed processing
        # This sets up multiple worker processes to handle the data in parallel
        status_display.write(f"Starting Dask cluster with {processing_config['n_workers']} workers...")
        
        dask_cluster = LocalCluster(
            n_workers=processing_config['n_workers'],           # Number of worker processes
            threads_per_worker=processing_config['threads_per_worker'],  # Threads per worker
            memory_limit=processing_config['memory_per_worker'], # Memory limit per worker
            processes=True,           # Use separate processes (better memory isolation)
            dashboard_address=':8787' # Web dashboard for monitoring
        )
        dask_client = Client(dask_cluster)
        
        # Initialize performance monitoring
        performance_monitor = DaskMonitor(dask_client)
        
        # Display the dashboard link so users can monitor progress in real-time
        dashboard_web_link = dask_client.dashboard_link
        with cluster_dashboard_area.container():
            st.success("✅ Dask Cluster Ready")
            st.markdown(f"**[Open Dashboard]({dashboard_web_link})**")
            st.caption("Real-time monitoring")
        
        # Step 4: Initialize the machine learning model for topic assignment
        # This model converts text reviews into numerical vectors for comparison
        status_display.write(f"Initializing sentence embedder: {SENTENCE_TRANSFORMER_MODEL}")
        
        # Check if GPU is available for faster processing
        device_type = 'cuda' if torch.cuda.is_available() and processing_config['use_gpu'] else 'cpu'
        sentence_embedder = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, device=device_type)
        
        # Move model to CPU for distribution to workers (required for serialization)
        sentence_embedder.to('cpu')
        for model_parameter in sentence_embedder.parameters():
            if hasattr(model_parameter, 'data'):
                model_parameter.data = model_parameter.data.cpu()
        
        # Distribute the model to all worker processes so they can use it
        embedder_dataset_name = f"embedder_{int(time.time())}"
        dask_client.publish_dataset(sentence_embedder, name=embedder_dataset_name)
        status_display.write(f"✅ Published embedder to {processing_config['n_workers']} workers")
        
        # Step 5: Process and validate uploaded files/ to read the uploaded files we need them on disk and not in memory so we dump the uploaded files in the disk using this
        with tempfile.TemporaryDirectory() as temporary_upload_directory:
            valid_files_for_processing = []
            skipped_files_with_reasons = []
            game_id_to_name_mapping = {}
            
            # Check each uploaded file to see if we can process it
            for uploaded_file in uploaded_files:
                # Save the uploaded file to temporary storage
                temp_file_path = os.path.join(temporary_upload_directory, uploaded_file.name)
                
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                
                # Extract the Steam app ID and game name from the file
                steam_app_id, game_name = extract_appid_and_name_from_parquet(temp_file_path)
                
                # Check if this game's app ID is in our themes dictionary
                if steam_app_id and steam_app_id in game_themes_dict:
                    valid_files_for_processing.append((temp_file_path, steam_app_id))
                    if game_name:
                        game_id_to_name_mapping[steam_app_id] = game_name
                    status_display.write(f"✅ File '{uploaded_file.name}' has app ID {steam_app_id} - Processing")
                else:
                    skipped_files_with_reasons.append((uploaded_file.name, steam_app_id))
                    status_display.write(f"⚠️ File '{uploaded_file.name}' has app ID {steam_app_id} - Skipping")
            
            # Make sure we have at least one valid file to process
            if not valid_files_for_processing:
                st.error("No valid files to process.")
                return None
            
            # Step 6: Break files into chunks for parallel processing
            # Large files are split into smaller pieces so multiple workers can process them
            status_display.write("Creating processing chunks...")
            all_processing_chunks = []
            chunk_id_counter = 0
            
            for file_path, app_id in valid_files_for_processing:
                # Create chunks for this file
                file_chunks = create_file_chunks(file_path, app_id, processing_config['chunk_size'])
                
                # Add unique IDs to each chunk for tracking
                for file_path, app_id, row_group_slices in file_chunks:
                    chunk_info = (file_path, app_id, row_group_slices, f"chunk_{chunk_id_counter}")
                    all_processing_chunks.append(chunk_info)
                    chunk_id_counter += 1
            
            total_chunks_to_process = len(all_processing_chunks)
            status_display.write(f"Created {total_chunks_to_process} chunks for parallel processing")
            
            # Step 7: Set up progress tracking
            progress_tracker = create_progress_tracker(total_chunks_to_process, progress_display_area)
            
            # Step 8: Create delayed tasks for distributed processing
            # "Delayed" means the tasks are defined but not executed yet
            delayed_processing_tasks = []
            task_submission_times = {}  # Track when each task was submitted
            
            for file_path, app_id, row_group_slices, chunk_id in all_processing_chunks:
                # Create a delayed task to load the chunk data
                chunk_loading_task = delayed(load_chunk_data)((file_path, app_id, row_group_slices))
 
                # Create a delayed task to process the chunk (load + topic assignment)
                chunk_processing_task = process_chunk_delayed(
                    chunk_loading_task, 
                    chunk_id,
                    app_id,
                    game_themes_dict,
                    embedder_dataset_name,
                    temp_storage_paths
                )
                delayed_processing_tasks.append(chunk_processing_task)
                task_submission_times[chunk_id] = time.time()

            
            # Step 9: Submit all tasks to the Dask cluster for execution
            progress_tracker['status_text'].write(f"Submitting {len(delayed_processing_tasks)} tasks to Dask cluster...")
            computation_futures = dask_client.compute(delayed_processing_tasks)
            
            # Create a mapping from futures to chunk IDs for progress tracking
            future_to_chunk_mapping = {computation_futures[i]: all_processing_chunks[i][3] for i in range(len(computation_futures))}
            
            # Step 10: Track progress as tasks complete
            completed_chunks_count = 0
            total_rows_processed_count = 0
            
            # Process results as they finish (not necessarily in order)
            for completed_future in as_completed(computation_futures):
                chunk_id = future_to_chunk_mapping[completed_future]
                task_start_time = task_submission_times.get(chunk_id, time.time())
                
                # Get the result from the completed task
                chunk_result = completed_future.result()
                completed_chunks_count += 1
                total_rows_processed_count += chunk_result['processed_rows']
                
                # Log performance metrics
                task_duration = time.time() - task_start_time
                performance_monitor.log_task_completion(chunk_id, task_duration, chunk_result['processed_rows'])
                
                # Update the progress display
                update_progress_tracker(progress_tracker, completed_chunks_count, total_chunks_to_process, 
                                      total_rows_processed_count, performance_monitor)
                
                # Update cluster metrics display every 5 completed tasks
                if completed_chunks_count % 5 == 0:
                    performance_monitor.display_cluster_metrics(system_metrics_area)
                
                # Clean up memory every 10 completed tasks
                if completed_chunks_count % 10 == 0:
                    gc.collect()
            
            # Store the total number of rows processed for later use
            st.session_state["total_rows"] = total_rows_processed_count
            
            # Final progress update
            update_progress_tracker(progress_tracker, completed_chunks_count, total_chunks_to_process, 
                                  total_rows_processed_count, performance_monitor)
            
            # Display final cluster performance metrics
            performance_monitor.display_cluster_metrics(system_metrics_area)
            
            # Step 11: Combine all the temporary results into a final report
            progress_tracker['status_text'].write("Aggregating chunk results...")
            final_aggregated_report = aggregate_temp_results(temp_storage_paths, game_themes_dict, game_id_to_name_mapping)
            
            if final_aggregated_report.empty:
                st.error("No data after processing.")
                return None
            
            # Save the final report to a CSV file
            final_aggregated_report.to_csv(DEFAULT_INTERIM_PATH, index=False)
            progress_tracker['status_text'].write(f"✅ Saved sentiment report to {DEFAULT_INTERIM_PATH}")
            
            # Display performance summary
            display_performance_summary(performance_monitor, main_status_area)
            
            # Calculate total time taken for this processing phase
            phase_elapsed_time = time.time() - phase_start_time
            formatted_elapsed_time = str(datetime.timedelta(seconds=int(phase_elapsed_time)))
            
            # Show success message with summary statistics
            st.success(
                f"✅ **Data Processing Complete!**\n\n"
                f"- Total time: {formatted_elapsed_time}\n"
                f"- Processed: {total_rows_processed_count:,} reviews\n"
                f"- Parallel chunks: {total_chunks_to_process}\n"
                f"- Workers used: {processing_config['n_workers']}"
            )
            
            # Return all the results and metadata
            return {
                'final_report': final_aggregated_report,
                'valid_files': valid_files_for_processing,
                'skipped_files': skipped_files_with_reasons,
                'processing_time': phase_elapsed_time,
                'game_name_mapping': game_id_to_name_mapping,
                'total_rows': total_rows_processed_count,
                'dashboard_link': dashboard_web_link,
                'performance_summary': performance_monitor.get_performance_summary()
            }
            
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None
        
    finally:
        # Always clean up resources, even if an error occurred
        cleanup_temp_storage(temp_storage_paths)
        if dask_client:
            dask_client.close()
        if dask_cluster:
            dask_cluster.close()


def create_temp_storage() -> Dict[str, str]:
    """
    Create temporary directory structure for storing intermediate processing results.
    
    During processing, we need to store partial results from each worker.
    This function creates a temporary directory structure with folders for:
    - Aggregation data (counts + review text lists by topic)
    - Metadata
    
    Returns:
        Dictionary mapping folder names to their full paths
    """
    TEMP_DIR_PREFIX = "steamlens_temp_dir_"
    temporary_base_directory = tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX)
    
    temp_folder_paths = {
        'base': temporary_base_directory,
        'aggregations': os.path.join(temporary_base_directory, 'aggregations'),
        'metadata': os.path.join(temporary_base_directory, 'metadata')
    }
    
    # Create all the subdirectories
    for folder_path in temp_folder_paths.values():
        if folder_path != temporary_base_directory:
            os.makedirs(folder_path, exist_ok=True)
    
    return temp_folder_paths


def cleanup_temp_storage(temp_storage_paths: Dict[str, str]) -> None:
    """
    Clean up temporary storage directories and files.
    
    Args:
        temp_storage_paths: Dictionary of temporary folder paths to clean up
    """
    try:
        shutil.rmtree(temp_storage_paths['base'])
    except Exception as e:
        st.warning(f"Could not clean up temporary files: {str(e)}")


@dask.delayed
def process_chunk_delayed(chunk_dataframe: pd.DataFrame, chunk_identifier: str, app_id: int, 
                         game_themes_dict: Dict, embedder_dataset_name: str,
                         temp_storage_paths: Dict[str, str]) -> Dict[str, Any]:
    """
    Process a single chunk of review data using Dask delayed execution.
    
    This function runs on a Dask worker and performs:
    1. Gets the machine learning model from the published dataset
    2. Assigns topics to reviews using semantic similarity
    3. Separates positive and negative reviews
    4. Saves results to temporary files
    5. Returns processing statistics
    
    Args:
        chunk_dataframe: Pandas DataFrame containing the review data for this chunk
        chunk_identifier: Unique string ID for this chunk (for tracking)
        app_id: Steam app ID for the game these reviews belong to
        game_themes_dict: Dictionary mapping app IDs to their themes
        embedder_dataset_name: Name of the published sentence embedding model
        temp_storage_paths: Dictionary of temporary storage folder paths
        
    Returns:
        Dictionary containing processing statistics and metadata
    """
    from dask.distributed import get_worker
    
    # Get the sentence embedding model that was distributed to all workers
    current_worker = get_worker()
    sentence_embedder = current_worker.client.get_dataset(embedder_dataset_name)
    
    # Assign topics/themes to each review in this chunk using machine learning
    chunk_with_assigned_topics = assign_topic(chunk_dataframe, game_themes_dict, sentence_embedder)
    
    # Build one aggregated output row per topic for this chunk.
    aggregation_data_rows = []
    for topic_id, topic_specific_reviews in chunk_with_assigned_topics.groupby('topic_id', sort=False):
        total_review_count = int(len(topic_specific_reviews))
        positive_mask = topic_specific_reviews['voted_up']
        positive_reviews_list = topic_specific_reviews.loc[positive_mask, 'review'].tolist()
        negative_reviews_list = topic_specific_reviews.loc[~positive_mask, 'review'].tolist()
        positive_reviews_count = int(positive_mask.sum())  # voted_up is True/False

        aggregation_data_rows.append({
            'steam_appid': app_id,
            'topic_id': int(topic_id),
            'review_count': total_review_count,
            'likes_sum': positive_reviews_count,
            'dislikes_sum': total_review_count - positive_reviews_count,
            'positive_reviews': positive_reviews_list,
            'negative_reviews': negative_reviews_list
        })

    # Save one temp file per chunk to reduce filesystem fanout.
    if aggregation_data_rows:
        aggregation_filename = os.path.join(
            temp_storage_paths['aggregations'],
            f"chunk_{chunk_identifier}_app_{app_id}.parquet"
        )
        pd.DataFrame(aggregation_data_rows).to_parquet(aggregation_filename, compression='snappy')
    
    # Clean up memory before returning
    del chunk_with_assigned_topics
    gc.collect()
    
    # Return processing statistics for monitoring
    return {
        'chunk_id': chunk_identifier,
        'processed_rows': len(chunk_dataframe),
        'topics_found': len(aggregation_data_rows),
        'app_id': app_id
    }


def create_file_chunks(
    file_path: str,
    app_id: int,
    chunk_size: int
) -> List[Tuple[str, int, List[Tuple[int, int, int]]]]:
    """
    Create chunk boundaries for a file without loading all the data into memory.
    
    Large files need to be processed in smaller pieces to avoid memory issues.
    This function determines where to split the file into chunks.
    
    Args:
        file_path: Path to the parquet file
        app_id: Steam app ID to filter for
        chunk_size: Number of rows per chunk
        
    Returns:
        List of tuples: (file_path, app_id, row_group_slices)

    What Each Worker Will Do:

    Worker 1: "I'll process (row_group=0, rows 0-7499)"
    Worker 2: "I'll process (row_group=0, rows 7500-14999)"
    Worker 3: "I'll process (row_group=1, rows 0-7499)"
    """
    parquet_file = pq.ParquetFile(file_path)
    matching_row_groups: List[Tuple[int, int]] = []

    # Scan row groups once with only lightweight columns.
    for row_group_idx in range(parquet_file.num_row_groups):
        row_group_df = parquet_file.read_row_group(
            row_group_idx,
            columns=['steam_appid', 'review_language']
        ).to_pandas()
        matching_count = int(
            (
                (row_group_df['review_language'] == DEFAULT_LANGUAGE) &
                (row_group_df['steam_appid'] == app_id)
            ).sum()
        )
        if matching_count > 0:
            matching_row_groups.append((row_group_idx, matching_count))

    if not matching_row_groups:
        return []

    # Build chunks from filtered row slices so a single huge row-group can still parallelize.
    chunk_boundaries: List[Tuple[str, int, List[Tuple[int, int, int]]]] = []
    current_slices: List[Tuple[int, int, int]] = []
    current_rows = 0

    for row_group_idx, matching_count in matching_row_groups:
        row_group_offset = 0
        while row_group_offset < matching_count:
            remaining = chunk_size - current_rows
            take_rows = min(remaining, matching_count - row_group_offset)
            current_slices.append(
                (row_group_idx, row_group_offset, row_group_offset + take_rows)
            )
            current_rows += take_rows
            row_group_offset += take_rows

            if current_rows >= chunk_size:
                chunk_boundaries.append((file_path, app_id, current_slices))
                current_slices = []
                current_rows = 0

    if current_slices:
        chunk_boundaries.append((file_path, app_id, current_slices))

    return chunk_boundaries


def load_chunk_data(
    chunk_info: Tuple[str, int, List[Tuple[int, int, int]]]
) -> pd.DataFrame:
    """
    Load a specific chunk of data from a parquet file.
    
    This function loads only the parquet row groups assigned to the chunk,
    filters them for the correct language and game, and returns a clean DataFrame.
    
    Args:
        chunk_info: Tuple containing (file_path, app_id, row_group_slices)
        
    Returns:
        Pandas DataFrame containing the filtered chunk data
    """
    file_path, app_id, row_group_slices = chunk_info
    parquet_file = pq.ParquetFile(file_path)
    chunk_parts = []

    for row_group_idx, slice_start, slice_end in row_group_slices:
        row_group_df = parquet_file.read_row_group(
            row_group_idx,
            columns=PARQUET_COLUMNS
        ).to_pandas()
        filtered_row_group = row_group_df[
            (row_group_df['review_language'] == DEFAULT_LANGUAGE) &
            (row_group_df['steam_appid'] == app_id)
        ]
        if not filtered_row_group.empty:
            chunk_parts.append(filtered_row_group.iloc[slice_start:slice_end].copy())

    if not chunk_parts:
        return pd.DataFrame(columns=PARQUET_COLUMNS)

    return pd.concat(chunk_parts, ignore_index=True)


def aggregate_temp_results(temp_storage_paths: Dict[str, str], game_themes_dict: Dict,
                          game_name_mapping: Dict[int, str]) -> pd.DataFrame:
    """
    Combine all temporary results from worker processes into a final report.
    
    After all chunks are processed, we need to combine the results:
    1. Aggregate counting data (total reviews, likes, dislikes per theme)
    2. Collect all positive and negative reviews for each theme
    3. Create a final report with readable theme names and percentages
    
    Args:
        temp_storage_paths: Dictionary of temporary storage folder paths
        game_themes_dict: Dictionary mapping app IDs to their themes
        game_name_mapping: Dictionary mapping app IDs to readable game names
        
    Returns:
        Pandas DataFrame containing the final aggregated report
    """
    # Step 1: Read and combine all aggregation files
    aggregation_files = list(Path(temp_storage_paths['aggregations']).glob('*.parquet'))
    
    if not aggregation_files:
        return pd.DataFrame()
    
    # Combine all chunk aggregation data (counts + review lists)
    all_aggregation_data = []
    for aggregation_file in aggregation_files:
        all_aggregation_data.append(pd.read_parquet(aggregation_file))
    
    combined_aggregation_data = pd.concat(all_aggregation_data)

    def _normalize_reviews(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value if v is not None]
        if isinstance(value, tuple):
            return [str(v) for v in value if v is not None]
        if isinstance(value, np.ndarray):
            return [str(v) for v in value.tolist() if v is not None]
        if isinstance(value, pd.Series):
            return [str(v) for v in value.tolist() if v is not None]
        if hasattr(value, "as_py"):
            py_value = value.as_py()
            return _normalize_reviews(py_value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("[") and stripped.endswith("]"):
                import ast
                try:
                    parsed = ast.literal_eval(stripped)
                    return _normalize_reviews(parsed)
                except Exception:
                    return [stripped]
            return [stripped]
        if isinstance(value, Iterable):
            return [str(v) for v in value if v is not None]
        return [str(value)]

    def _merge_review_lists(series: pd.Series) -> List[str]:
        merged_reviews: List[str] = []
        for reviews in series:
            normalized_reviews = _normalize_reviews(reviews)
            if normalized_reviews:
                merged_reviews.extend(normalized_reviews)
        return merged_reviews

    # Group by app/topic and merge both metrics and review text.
    final_aggregation = combined_aggregation_data.groupby(['steam_appid', 'topic_id']).agg({
        'review_count': 'sum',
        'likes_sum': 'sum',
        'dislikes_sum': 'sum',
        'positive_reviews': _merge_review_lists,
        'negative_reviews': _merge_review_lists
    }).reset_index()

    # Step 2: Build the final report with readable information
    final_report_rows = []
    
    for _, aggregation_row in final_aggregation.iterrows():
        steam_app_id = int(aggregation_row['steam_appid'])
        topic_id = int(aggregation_row['topic_id'])
        
        # Get a readable theme name
        if steam_app_id in game_themes_dict:
            theme_names_list = list(game_themes_dict[steam_app_id].keys())
            readable_theme_name = theme_names_list[topic_id] if topic_id < len(theme_names_list) else f"Unknown Theme {topic_id}"
        else:
            readable_theme_name = f"Unknown Theme {topic_id}"
        
        # Calculate percentages
        total_reviews = aggregation_row['review_count']
        if total_reviews > 0:
            like_percentage = f"{(aggregation_row['likes_sum'] / total_reviews * 100):.1f}%"
            dislike_percentage = f"{(aggregation_row['dislikes_sum'] / total_reviews * 100):.1f}%"
        else:
            like_percentage = "0.0%"
            dislike_percentage = "0.0%"
        
        # Get the actual review text for this topic
        positive_reviews_for_topic = _normalize_reviews(aggregation_row.get('positive_reviews', []))
        negative_reviews_for_topic = _normalize_reviews(aggregation_row.get('negative_reviews', []))
        
        # Add this row to the final report
        final_report_rows.append({
            'steam_appid': steam_app_id,
            'Theme': readable_theme_name,
            '#Reviews': int(total_reviews),
            'Positive': int(aggregation_row['likes_sum']),
            'Negative': int(aggregation_row['dislikes_sum']),
            'LikeRatio': like_percentage,
            'DislikeRatio': dislike_percentage,
            'Positive_Reviews': positive_reviews_for_topic,
            'Negative_Reviews': negative_reviews_for_topic
        })
    
    return pd.DataFrame(final_report_rows)
