#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import threading
import datetime
import streamlit as st
import pandas as pd
import dask
import math
from typing import Dict, Any, Optional, List, Tuple
from dask.distributed import Client, LocalCluster

from ..config.app_config import HARDWARE_CONFIG, DEFAULT_OUTPUT_PATH
from .summarization import prepare_partition, process_partition, update_main_progress


def _validate_report(final_report: pd.DataFrame) -> bool:
    """Validate that the report exists and is not empty."""
    if final_report is None or final_report.empty:
        st.error("No report to summarize.")
        return False
    return True


def _create_progress_indicators() -> Tuple[Any, Any, Any, Any, Any]:
    """Create and return progress indicator placeholders."""
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    dashboard_placeholder = st.empty()
    
    with progress_placeholder.container():
        status_text = st.empty()
        progress_bar = st.progress(0.0)
    
    return progress_placeholder, status_placeholder, dashboard_placeholder, status_text, progress_bar


def _check_gpu_availability(status_text: Any) -> Tuple[bool, int, str]:
    """Check GPU availability and return status information."""
    import torch
    gpu_available = torch.cuda.is_available()
    
    if gpu_available:
        device_count = torch.cuda.device_count()
        status_text.write(f"GPU detected: {device_count} device(s) available")
        memory_limit = HARDWARE_CONFIG['memory_per_worker']
    else:
        status_text.write("No GPU detected, using CPU mode")
        device_count = 0
        # Increase memory for CPU workers
        memory_limit = HARDWARE_CONFIG['memory_per_worker'] * 1.5
    
    return gpu_available, device_count, memory_limit


def _setup_dask_cluster(gpu_available: bool, memory_limit: str, status_text: Any, 
                       dashboard_placeholder: Any) -> Tuple[LocalCluster, Client]:
    """Set up Dask cluster with optimized settings."""
    # Allow more workers (up to 6) for GPU processing
    worker_count = max(HARDWARE_CONFIG['worker_count'],8 if gpu_available else 4)
    
    cluster = LocalCluster(
        n_workers=worker_count, 
        threads_per_worker=4,
        memory_limit=memory_limit
    )
    client = Client(cluster)
    status_text.write(f"Dask cluster initialized with {worker_count} workers")
    
    # Store client in session state for potential reset
    st.session_state.summarize_client = client
    
    # Display dashboard link for monitoring
    with dashboard_placeholder.container():
        st.success("✅ Dask Summarization Cluster Ready")
        st.markdown(f"**[Open Dask Dashboard]({client.dashboard_link})** (opens in new tab)")
        st.info("👁️ Monitor summarization tasks in real-time with this dashboard")
    
    # Store dashboard link in session state
    if 'summarize_dashboard_link' not in st.session_state:
        st.session_state.summarize_dashboard_link = client.dashboard_link
    
    return cluster, client


def _create_partitions(final_report: pd.DataFrame, n_workers: int, status_text: Any) -> List[dask.delayed]:
    """Create balanced partitions for distributed processing."""
    # ceiling‑divide so remainders aren’t dropped
    partition_size = math.ceil(len(final_report) / n_workers)

    partitions = []
    idx = 0
    part = 0
    while idx < len(final_report):
        end_idx = min(idx + partition_size, len(final_report))
        partitions.append(
            dask.delayed(prepare_partition)(idx, end_idx, final_report)
        )
        status_text.write(f"Prepared partition {part + 1} with {end_idx - idx} items")
        idx = end_idx
        part += 1

    return partitions



def _schedule_tasks(partitions: List[dask.delayed], hardware_config: Dict[str, Any]) -> List[dask.delayed]:
    """Schedule summarization tasks for each partition."""
    delayed_results = []
    
    for i, partition in enumerate(partitions):
        delayed_result = dask.delayed(process_partition)(partition, i, hardware_config)
        delayed_results.append(delayed_result)
    
    return delayed_results


def _compute_with_futures(client: Client, delayed_results: List[dask.delayed], 
                         progress_bar: Any, status_text: Any, 
                         final_report_length: int) -> Optional[List]:
    """Try to compute results using futures with progress monitoring."""
    try:
        # Submit tasks to cluster with async computation
        futures = client.compute(delayed_results)
        
        # Start progress monitor with minimal overhead
        stop_flag = [False]  # Use a list to make it mutable for the thread
        monitor_thread = threading.Thread(
            target=update_main_progress, 
            args=(futures, progress_bar, stop_flag, final_report_length)
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Wait for computation
        status_text.write("Computing sentiment-based summaries with optimal settings...")
        results = client.gather(futures)
        
        # Stop progress monitor
        stop_flag[0] = True
        if monitor_thread.is_alive():
            monitor_thread.join(timeout=3)
        
        return results
        
    except Exception as e:
        status_text.write(f"Error with futures: {e}")
        return None


def _compute_with_fallback(delayed_results: List[dask.delayed], 
                          status_text: Any) -> Optional[List]:
    """Fallback computation method without futures."""
    status_text.write("Falling back to direct computation...")
    try:
        results = dask.compute(*delayed_results)
        return results
    except Exception as e:
        status_text.write(f"Error in fallback computation: {e}")
        return None


def _process_results(results: List, status_text: Any) -> Optional[List[Tuple[int, str, str]]]:
    """Process and validate computation results."""
    if results is None:
        status_text.write("❌ Failed to generate summaries")
        return None
    
    # Process results efficiently
    all_results = []
    for worker_results in results:
        if worker_results:  # Check if we got valid results
            all_results.extend(worker_results)
    
    if not all_results:
        status_text.write("❌ No valid results generated")
        return None
    
    # Sort results
    all_results.sort(key=lambda x: x[0])
    
    return all_results


def _merge_summaries_with_report(final_report: pd.DataFrame, 
                                all_results: List[Tuple[int, str, str]]) -> pd.DataFrame:
    """Merge computed summaries with the original report."""
    # Extract positive and negative summaries
    indices = [result[0] for result in all_results]
    positive_summaries = [result[1] for result in all_results]
    negative_summaries = [result[2] for result in all_results]
    
    # Create a new DataFrame with the results
    result_df = pd.DataFrame({
        'index': indices,
        'Positive_Summary': positive_summaries,
        'Negative_Summary': negative_summaries
    }).set_index('index')
    
    # Merge with the original DataFrame
    final_report = final_report.join(result_df[['Positive_Summary', 'Negative_Summary']])
    
    # Fill any missing values (for rows that weren't processed)
    final_report['Positive_Summary'] = final_report['Positive_Summary'].fillna("Summary not available")
    final_report['Negative_Summary'] = final_report['Negative_Summary'].fillna("Summary not available")
    
    return final_report


def _save_results(final_report: pd.DataFrame, status_text: Any) -> bool:
    """Save the results to a CSV file."""
    try:
        final_report.to_csv(DEFAULT_OUTPUT_PATH)
        status_text.write(f"✅ Results saved to {DEFAULT_OUTPUT_PATH}")
        return True
    except Exception as e:
        status_text.write(f"⚠️ Failed to save results: {e}")
        return False


def _report_timing_metrics(start_time: float, phase_start_time: float, 
                          final_report_length: int, status_text: Any) -> None:
    """Report timing metrics for the summarization process."""
    elapsed_time = time.time() - start_time
    status_text.write(f"✅ **EXECUTION TIME:** {elapsed_time:.2f} seconds")
    status_text.write(f"✅ Average time per item: {elapsed_time/final_report_length:.4f} seconds")
    
    # Calculate elapsed time for this phase
    phase_elapsed_time = time.time() - phase_start_time
    formatted_time = str(datetime.timedelta(seconds=int(phase_elapsed_time)))
    status_text.write(f"✅ **TOTAL SUMMARIZATION TIME:** {formatted_time}")


def _cleanup_resources(client: Optional[Client], cluster: Optional[LocalCluster]) -> None:
    """Clean up Dask resources."""
    if client:
        try:
            client.close()
        except:
            pass
    if cluster:
        try:
            cluster.close()
        except:
            pass


def summarize_report(final_report: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Main function to summarize the sentiment report."""
    # Start phase timer
    phase_start_time = time.time()
    
    # Validate report
    if not _validate_report(final_report):
        return None
    
    # Create progress indicators
    progress_placeholder, status_placeholder, dashboard_placeholder, status_text, progress_bar = _create_progress_indicators()
    
    # Use hardware_config from config module
    hardware_config = HARDWARE_CONFIG
    
    status_text.write(f"Starting optimized Dask cluster for sentiment-based summarization")
    
    # Use try/finally to ensure proper cleanup of Dask resources
    cluster = None
    client = None
    
    try:
        # Check GPU availability
        gpu_available, device_count, memory_limit = _check_gpu_availability(status_text)
        
        # Setup Dask cluster
        cluster, client = _setup_dask_cluster(gpu_available, memory_limit, status_text, dashboard_placeholder)
        
        # Determine optimal partition sizes
        n_workers = len(cluster.workers)  # Get the actual number of workers
        partitions = _create_partitions(final_report, n_workers, status_text)
        
        # Schedule tasks
        status_text.write(f"Scheduling {len(partitions)} optimized partitions for sentiment analysis...")
        delayed_results = _schedule_tasks(partitions, hardware_config)
        
        # Start timing
        start_time = time.time()
        
        # Try to compute with futures first
        results = _compute_with_futures(client, delayed_results, progress_bar, 
                                       status_text, len(final_report))
        
        # If futures failed, try fallback method
        if results is None:
            results = _compute_with_fallback(delayed_results, status_text)
        
        # Update progress to completion
        progress_bar.progress(1.0)
        
        # Process results
        all_results = _process_results(results, status_text)
        if all_results is None:
            return None
        
        # Merge summaries with report
        final_report = _merge_summaries_with_report(final_report, all_results)
        
        # Report timing metrics
        _report_timing_metrics(start_time, phase_start_time, len(final_report), status_text)
        
        # Save results
        _save_results(final_report, status_text)
        
        return final_report
        
    except Exception as e:
        # Handle any unexpected errors
        status_text.write(f"❌ Error during summarization: {str(e)}")
        import traceback
        status_text.write(f"Error details: {traceback.format_exc()}")
        return None
        
    finally:
        # Clean up resources
        _cleanup_resources(client, cluster)