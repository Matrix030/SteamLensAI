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

import datetime
import gc
import os
import shutil
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dask
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
import torch
from dask import delayed
from dask.distributed import Client, LocalCluster, as_completed
from sentence_transformers import SentenceTransformer

# Import configuration and utility modules
from ..config.app_config import (
    DEFAULT_INTERIM_PATH,
    DEFAULT_LANGUAGE,
    PARQUET_COLUMNS,
    PROCESSING_CONFIG,
    SENTENCE_TRANSFORMER_MODEL,
)
from ..data.data_loader import extract_appid_and_name_from_parquet
from .topic_assignment import assign_topic

# Try to import monitoring utilities (optional dependency)
try:
    from ..utils.dask_monitor import (
        DaskMonitor,
        create_progress_tracker,
        display_performance_summary,
        update_progress_tracker,
    )

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
            "progress_bar": progress_bar,
            "status_text": status_text,
            "chunks_metric": None,
            "rows_metric": None,
            "time_metric": None,
            "eta_metric": None,
        }

    def update_progress_tracker(tracker, completed, total_chunks, total_rows, monitor):
        """Update basic progress when full monitoring isn't available"""
        progress = completed / total_chunks if total_chunks > 0 else 0
        tracker["progress_bar"].progress(progress)
        tracker["status_text"].write(
            f"Processing chunk {completed}/{total_chunks} ({total_rows:,} rows)"
        )

    def display_performance_summary(monitor, container):
        """Display basic completion message when full monitoring isn't available"""
        container.success("Processing complete!")


def _initialize_processing_layout() -> Dict[str, Any]:
    progress_column, cluster_info_column = st.columns([3, 1])

    with progress_column:
        progress_display_area = st.empty()
        main_status_area = st.empty()

    with cluster_info_column:
        cluster_dashboard_area = st.empty()
        system_metrics_area = st.empty()

    with main_status_area.container():
        status_display = st.empty()

    return {
        "progress_display_area": progress_display_area,
        "main_status_area": main_status_area,
        "cluster_dashboard_area": cluster_dashboard_area,
        "system_metrics_area": system_metrics_area,
        "status_display": status_display,
    }


def _load_theme_dictionary_for_processing(
    themes_file: str, status_display: Any
) -> Optional[Dict[Any, Any]]:
    try:
        from ..data.data_loader import load_theme_dictionary

        game_themes_dict = load_theme_dictionary(themes_file)
        if not game_themes_dict:
            return None
        status_display.write(f"✅ Loaded theme dictionary with {len(game_themes_dict)} games")
        return game_themes_dict
    except Exception as e:
        st.error(f"Error loading theme dictionary: {str(e)}")
        return None


def _setup_dask_infrastructure(
    processing_config: Dict[str, Any],
    system_metrics_area: Any,
    cluster_dashboard_area: Any,
    status_display: Any,
) -> Tuple[LocalCluster, Client, DaskMonitor, str]:
    with system_metrics_area.container():
        from ..config.app_config import display_system_config

        st.info(display_system_config())

    status_display.write(f"Starting Dask cluster with {processing_config['n_workers']} workers...")

    dask_cluster = LocalCluster(
        n_workers=processing_config["n_workers"],
        threads_per_worker=processing_config["threads_per_worker"],
        memory_limit=processing_config["memory_per_worker"],
        processes=True,
        dashboard_address=":8787",
    )
    dask_client = Client(dask_cluster)
    performance_monitor = DaskMonitor(dask_client)

    dashboard_web_link = dask_client.dashboard_link
    with cluster_dashboard_area.container():
        st.success("✅ Dask Cluster Ready")
        st.markdown(f"**[Open Dashboard]({dashboard_web_link})**")
        st.caption("Real-time monitoring")

    return dask_cluster, dask_client, performance_monitor, dashboard_web_link


def _publish_sentence_embedder(
    dask_client: Client, processing_config: Dict[str, Any], status_display: Any
) -> str:
    status_display.write(f"Initializing sentence embedder: {SENTENCE_TRANSFORMER_MODEL}")

    device_type = "cuda" if torch.cuda.is_available() and processing_config["use_gpu"] else "cpu"
    sentence_embedder = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, device=device_type)

    sentence_embedder.to("cpu")
    for model_parameter in sentence_embedder.parameters():
        if hasattr(model_parameter, "data"):
            model_parameter.data = model_parameter.data.cpu()

    embedder_dataset_name = f"embedder_{int(time.time())}"
    dask_client.publish_dataset(sentence_embedder, name=embedder_dataset_name)
    status_display.write(f"✅ Published embedder to {processing_config['n_workers']} workers")
    return embedder_dataset_name


def _collect_valid_uploaded_files(
    uploaded_files: List[Any],
    temporary_upload_directory: str,
    game_themes_dict: Dict[Any, Any],
    status_display: Any,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, Optional[int]]], Dict[int, str]]:
    valid_files_for_processing: List[Tuple[str, int]] = []
    skipped_files_with_reasons: List[Tuple[str, Optional[int]]] = []
    game_id_to_name_mapping: Dict[int, str] = {}

    for uploaded_file in uploaded_files:
        temp_file_path = os.path.join(temporary_upload_directory, uploaded_file.name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getbuffer())

        steam_app_id, game_name = extract_appid_and_name_from_parquet(temp_file_path)
        if steam_app_id and steam_app_id in game_themes_dict:
            valid_files_for_processing.append((temp_file_path, steam_app_id))
            if game_name:
                game_id_to_name_mapping[steam_app_id] = game_name
            status_display.write(
                f"✅ File '{uploaded_file.name}' has app ID {steam_app_id} - Processing"
            )
            continue

        skipped_files_with_reasons.append((uploaded_file.name, steam_app_id))
        status_display.write(f"⚠️ File '{uploaded_file.name}' has app ID {steam_app_id} - Skipping")

    return valid_files_for_processing, skipped_files_with_reasons, game_id_to_name_mapping


def _create_processing_chunks(
    valid_files_for_processing: List[Tuple[str, int]], chunk_size: int, status_display: Any
) -> List[Tuple[str, int, List[Tuple[int, int, int]], str]]:
    status_display.write("Creating processing chunks...")
    all_processing_chunks: List[Tuple[str, int, List[Tuple[int, int, int]], str]] = []
    chunk_id_counter = 0

    for file_path, app_id in valid_files_for_processing:
        file_chunks = create_file_chunks(file_path, app_id, chunk_size)
        for chunk_file_path, chunk_app_id, row_group_slices in file_chunks:
            all_processing_chunks.append(
                (chunk_file_path, chunk_app_id, row_group_slices, f"chunk_{chunk_id_counter}")
            )
            chunk_id_counter += 1

    status_display.write(f"Created {len(all_processing_chunks)} chunks for parallel processing")
    return all_processing_chunks


def _create_delayed_processing_tasks(
    all_processing_chunks: List[Tuple[str, int, List[Tuple[int, int, int]], str]],
    game_themes_dict: Dict[Any, Any],
    embedder_dataset_name: str,
    temp_storage_paths: Dict[str, str],
) -> Tuple[List[Any], Dict[str, float]]:
    delayed_processing_tasks: List[Any] = []
    task_submission_times: Dict[str, float] = {}

    for file_path, app_id, row_group_slices, chunk_id in all_processing_chunks:
        chunk_loading_task = delayed(load_chunk_data)((file_path, app_id, row_group_slices))
        chunk_processing_task = process_chunk_delayed(
            chunk_loading_task,
            chunk_id,
            app_id,
            game_themes_dict,
            embedder_dataset_name,
            temp_storage_paths,
        )
        delayed_processing_tasks.append(chunk_processing_task)
        task_submission_times[chunk_id] = time.time()

    return delayed_processing_tasks, task_submission_times


def _run_processing_tasks(
    dask_client: Client,
    delayed_processing_tasks: List[Any],
    all_processing_chunks: List[Tuple[str, int, List[Tuple[int, int, int]], str]],
    task_submission_times: Dict[str, float],
    progress_tracker: Dict[str, Any],
    total_chunks_to_process: int,
    performance_monitor: DaskMonitor,
    system_metrics_area: Any,
) -> Tuple[int, int]:
    computation_futures = dask_client.compute(delayed_processing_tasks)
    future_to_chunk_mapping = {
        computation_futures[i]: all_processing_chunks[i][3] for i in range(len(computation_futures))
    }

    completed_chunks_count = 0
    total_rows_processed_count = 0
    for completed_future in as_completed(computation_futures):
        chunk_id = future_to_chunk_mapping[completed_future]
        task_start_time = task_submission_times.get(chunk_id, time.time())
        chunk_result = completed_future.result()

        completed_chunks_count += 1
        total_rows_processed_count += chunk_result["processed_rows"]
        task_duration = time.time() - task_start_time
        performance_monitor.log_task_completion(
            chunk_id, task_duration, chunk_result["processed_rows"]
        )

        update_progress_tracker(
            progress_tracker,
            completed_chunks_count,
            total_chunks_to_process,
            total_rows_processed_count,
            performance_monitor,
        )

        if completed_chunks_count % 5 == 0:
            performance_monitor.display_cluster_metrics(system_metrics_area)
        if completed_chunks_count % 10 == 0:
            gc.collect()

    return completed_chunks_count, total_rows_processed_count


def _build_processing_result_payload(
    final_report: pd.DataFrame,
    valid_files: List[Tuple[str, int]],
    skipped_files: List[Tuple[str, Optional[int]]],
    processing_time: float,
    game_name_mapping: Dict[int, str],
    total_rows: int,
    dashboard_link: str,
    performance_monitor: DaskMonitor,
) -> Dict[str, Any]:
    return {
        "final_report": final_report,
        "valid_files": valid_files,
        "skipped_files": skipped_files,
        "processing_time": processing_time,
        "game_name_mapping": game_name_mapping,
        "total_rows": total_rows,
        "dashboard_link": dashboard_link,
        "performance_summary": performance_monitor.get_performance_summary(),
    }


def process_uploaded_files(
    uploaded_files: List[Any], themes_file: str = "game_themes.json"
) -> Optional[Dict[str, Any]]:
    """Process uploaded Steam review parquet files with distributed chunked execution."""
    phase_start_time = time.time()
    if not uploaded_files:
        st.warning("Please upload at least one Parquet file to begin processing.")
        return None

    layout = _initialize_processing_layout()
    progress_display_area = layout["progress_display_area"]
    main_status_area = layout["main_status_area"]
    cluster_dashboard_area = layout["cluster_dashboard_area"]
    system_metrics_area = layout["system_metrics_area"]
    status_display = layout["status_display"]

    game_themes_dict = _load_theme_dictionary_for_processing(themes_file, status_display)
    if not game_themes_dict:
        return None

    temp_storage_paths = create_temp_storage()
    status_display.write("✅ Created temporary storage for chunk processing")

    dask_cluster = None
    dask_client = None
    try:
        processing_config = PROCESSING_CONFIG
        dask_cluster, dask_client, performance_monitor, dashboard_web_link = (
            _setup_dask_infrastructure(
                processing_config, system_metrics_area, cluster_dashboard_area, status_display
            )
        )
        embedder_dataset_name = _publish_sentence_embedder(
            dask_client, processing_config, status_display
        )

        with tempfile.TemporaryDirectory() as temporary_upload_directory:
            valid_files, skipped_files, game_name_mapping = _collect_valid_uploaded_files(
                uploaded_files, temporary_upload_directory, game_themes_dict, status_display
            )
            if not valid_files:
                st.error("No valid files to process.")
                return None

            all_processing_chunks = _create_processing_chunks(
                valid_files, processing_config["chunk_size"], status_display
            )
            total_chunks_to_process = len(all_processing_chunks)
            progress_tracker = create_progress_tracker(
                total_chunks_to_process, progress_display_area
            )

            delayed_processing_tasks, task_submission_times = _create_delayed_processing_tasks(
                all_processing_chunks,
                game_themes_dict,
                embedder_dataset_name,
                temp_storage_paths,
            )
            progress_tracker["status_text"].write(
                f"Submitting {len(delayed_processing_tasks)} tasks to Dask cluster..."
            )

            completed_chunks_count, total_rows_processed_count = _run_processing_tasks(
                dask_client,
                delayed_processing_tasks,
                all_processing_chunks,
                task_submission_times,
                progress_tracker,
                total_chunks_to_process,
                performance_monitor,
                system_metrics_area,
            )

            st.session_state["total_rows"] = total_rows_processed_count
            update_progress_tracker(
                progress_tracker,
                completed_chunks_count,
                total_chunks_to_process,
                total_rows_processed_count,
                performance_monitor,
            )
            performance_monitor.display_cluster_metrics(system_metrics_area)

            progress_tracker["status_text"].write("Aggregating chunk results...")
            final_aggregated_report = aggregate_temp_results(
                temp_storage_paths, game_themes_dict, game_name_mapping
            )
            if final_aggregated_report.empty:
                st.error("No data after processing.")
                return None

            final_aggregated_report.to_csv(DEFAULT_INTERIM_PATH, index=False)
            progress_tracker["status_text"].write(
                f"✅ Saved sentiment report to {DEFAULT_INTERIM_PATH}"
            )
            display_performance_summary(performance_monitor, main_status_area)

            phase_elapsed_time = time.time() - phase_start_time
            formatted_elapsed_time = str(datetime.timedelta(seconds=int(phase_elapsed_time)))
            st.success(
                f"✅ **Data Processing Complete!**\n\n"
                f"- Total time: {formatted_elapsed_time}\n"
                f"- Processed: {total_rows_processed_count:,} reviews\n"
                f"- Parallel chunks: {total_chunks_to_process}\n"
                f"- Workers used: {processing_config['n_workers']}"
            )

            return _build_processing_result_payload(
                final_aggregated_report,
                valid_files,
                skipped_files,
                phase_elapsed_time,
                game_name_mapping,
                total_rows_processed_count,
                dashboard_web_link,
                performance_monitor,
            )

    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        import traceback

        st.error(f"Traceback: {traceback.format_exc()}")
        return None
    finally:
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
        "base": temporary_base_directory,
        "aggregations": os.path.join(temporary_base_directory, "aggregations"),
        "metadata": os.path.join(temporary_base_directory, "metadata"),
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
        shutil.rmtree(temp_storage_paths["base"])
    except Exception as e:
        st.warning(f"Could not clean up temporary files: {str(e)}")


@dask.delayed
def process_chunk_delayed(
    chunk_dataframe: pd.DataFrame,
    chunk_identifier: str,
    app_id: int,
    game_themes_dict: Dict,
    embedder_dataset_name: str,
    temp_storage_paths: Dict[str, str],
) -> Dict[str, Any]:
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
    for topic_id, topic_specific_reviews in chunk_with_assigned_topics.groupby(
        "topic_id", sort=False
    ):
        total_review_count = int(len(topic_specific_reviews))
        positive_mask = topic_specific_reviews["voted_up"]
        positive_reviews_list = topic_specific_reviews.loc[positive_mask, "review"].tolist()
        negative_reviews_list = topic_specific_reviews.loc[~positive_mask, "review"].tolist()
        positive_reviews_count = int(positive_mask.sum())  # voted_up is True/False

        aggregation_data_rows.append(
            {
                "steam_appid": app_id,
                "topic_id": int(topic_id),
                "review_count": total_review_count,
                "likes_sum": positive_reviews_count,
                "dislikes_sum": total_review_count - positive_reviews_count,
                "positive_reviews": positive_reviews_list,
                "negative_reviews": negative_reviews_list,
            }
        )

    # Save one temp file per chunk to reduce filesystem fanout.
    if aggregation_data_rows:
        aggregation_filename = os.path.join(
            temp_storage_paths["aggregations"], f"chunk_{chunk_identifier}_app_{app_id}.parquet"
        )
        pd.DataFrame(aggregation_data_rows).to_parquet(aggregation_filename, compression="snappy")

    # Clean up memory before returning
    del chunk_with_assigned_topics
    gc.collect()

    # Return processing statistics for monitoring
    return {
        "chunk_id": chunk_identifier,
        "processed_rows": len(chunk_dataframe),
        "topics_found": len(aggregation_data_rows),
        "app_id": app_id,
    }


def create_file_chunks(
    file_path: str, app_id: int, chunk_size: int
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
            row_group_idx, columns=["steam_appid", "review_language"]
        ).to_pandas()
        matching_count = int(
            (
                (row_group_df["review_language"] == DEFAULT_LANGUAGE)
                & (row_group_df["steam_appid"] == app_id)
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
            current_slices.append((row_group_idx, row_group_offset, row_group_offset + take_rows))
            current_rows += take_rows
            row_group_offset += take_rows

            if current_rows >= chunk_size:
                chunk_boundaries.append((file_path, app_id, current_slices))
                current_slices = []
                current_rows = 0

    if current_slices:
        chunk_boundaries.append((file_path, app_id, current_slices))

    return chunk_boundaries


def load_chunk_data(chunk_info: Tuple[str, int, List[Tuple[int, int, int]]]) -> pd.DataFrame:
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
            row_group_idx, columns=PARQUET_COLUMNS
        ).to_pandas()
        filtered_row_group = row_group_df[
            (row_group_df["review_language"] == DEFAULT_LANGUAGE)
            & (row_group_df["steam_appid"] == app_id)
        ]
        if not filtered_row_group.empty:
            chunk_parts.append(filtered_row_group.iloc[slice_start:slice_end].copy())

    if not chunk_parts:
        return pd.DataFrame(columns=PARQUET_COLUMNS)

    return pd.concat(chunk_parts, ignore_index=True)


def _read_aggregation_frames(temp_storage_paths: Dict[str, str]) -> List[pd.DataFrame]:
    aggregation_files = list(Path(temp_storage_paths["aggregations"]).glob("*.parquet"))
    return [pd.read_parquet(aggregation_file) for aggregation_file in aggregation_files]


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
        return _normalize_reviews(value.as_py())
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            import ast

            try:
                return _normalize_reviews(ast.literal_eval(stripped))
            except Exception:
                return [stripped]
        return [stripped]
    if isinstance(value, Iterable):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def _merge_review_lists(series: pd.Series) -> List[str]:
    merged_reviews: List[str] = []
    for reviews in series:
        merged_reviews.extend(_normalize_reviews(reviews))
    return merged_reviews


def _resolve_theme_name(steam_app_id: int, topic_id: int, game_themes_dict: Dict[Any, Any]) -> str:
    if steam_app_id not in game_themes_dict:
        return f"Unknown Theme {topic_id}"
    theme_names_list = list(game_themes_dict[steam_app_id].keys())
    if topic_id < len(theme_names_list):
        return theme_names_list[topic_id]
    return f"Unknown Theme {topic_id}"


def _format_percentage(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.0%"
    return f"{(numerator / denominator * 100):.1f}%"


def _build_final_report_row(
    aggregation_row: pd.Series, game_themes_dict: Dict[Any, Any]
) -> Dict[str, Any]:
    steam_app_id = int(aggregation_row["steam_appid"])
    topic_id = int(aggregation_row["topic_id"])
    total_reviews = int(aggregation_row["review_count"])
    positive_count = int(aggregation_row["likes_sum"])
    negative_count = int(aggregation_row["dislikes_sum"])

    return {
        "steam_appid": steam_app_id,
        "Theme": _resolve_theme_name(steam_app_id, topic_id, game_themes_dict),
        "#Reviews": total_reviews,
        "Positive": positive_count,
        "Negative": negative_count,
        "LikeRatio": _format_percentage(positive_count, total_reviews),
        "DislikeRatio": _format_percentage(negative_count, total_reviews),
        "Positive_Reviews": _normalize_reviews(aggregation_row.get("positive_reviews", [])),
        "Negative_Reviews": _normalize_reviews(aggregation_row.get("negative_reviews", [])),
    }


def aggregate_temp_results(
    temp_storage_paths: Dict[str, str], game_themes_dict: Dict, game_name_mapping: Dict[int, str]
) -> pd.DataFrame:
    """
    Combine all temporary results from worker processes into a final report.

    After all chunks are processed, we need to combine the results:
    1. Aggregate counting data (total reviews, likes, dislikes per theme)
    2. Collect all positive and negative reviews for each theme
    3. Create a final report with readable theme names and percentages
    """
    _ = game_name_mapping
    all_aggregation_data = _read_aggregation_frames(temp_storage_paths)
    if not all_aggregation_data:
        return pd.DataFrame()

    combined_aggregation_data = pd.concat(all_aggregation_data)
    final_aggregation = (
        combined_aggregation_data.groupby(["steam_appid", "topic_id"])
        .agg(
            {
                "review_count": "sum",
                "likes_sum": "sum",
                "dislikes_sum": "sum",
                "positive_reviews": _merge_review_lists,
                "negative_reviews": _merge_review_lists,
            }
        )
        .reset_index()
    )

    final_report_rows = [
        _build_final_report_row(aggregation_row, game_themes_dict)
        for _, aggregation_row in final_aggregation.iterrows()
    ]
    return pd.DataFrame(final_report_rows)
