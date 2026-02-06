#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import streamlit as st
from typing import Dict, Any, Optional
from dask.distributed import Client
import pandas as pd

class DaskMonitor:
    """Monitor Dask cluster performance and display metrics"""
    
    def __init__(self, client: Client):
        self.client = client
        self.start_time = time.time()
        self.task_history = []
        
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status"""
        try:
            info = self.client.scheduler_info()
            
            # Count workers and threads
            n_workers = len(info.get('workers', {}))
            total_threads = sum(w.get('nthreads', 0) for w in info.get('workers', {}).values())
            total_memory = sum(w.get('memory_limit', 0) for w in info.get('workers', {}).values()) / (1024**3)
            
            # Get task status - safely handle missing keys
            processing = 0
            for worker in info.get('workers', {}).values():
                # Check multiple possible keys for active tasks
                if 'processing' in worker:
                    processing += len(worker['processing'])
                elif 'executing' in worker:
                    processing += len(worker['executing'])
                elif 'tasks' in worker:
                    processing += worker['tasks']
            
            return {
                'n_workers': n_workers,
                'total_threads': total_threads,
                'total_memory_gb': total_memory,
                'tasks_processing': processing,
                'dashboard_link': self.client.dashboard_link
            }
        except Exception as e:
            # Return safe defaults if there's an error
            return {
                'n_workers': 0,
                'total_threads': 0,
                'total_memory_gb': 0,
                'tasks_processing': 0,
                'dashboard_link': self.client.dashboard_link if hasattr(self.client, 'dashboard_link') else '#'
            }
    
    def get_worker_status(self) -> pd.DataFrame:
        """Get status of each worker"""
        try:
            info = self.client.scheduler_info()
            
            worker_data = []
            for worker_id, worker_info in info.get('workers', {}).items():
                # Safely extract worker information
                tasks = 0
                if 'processing' in worker_info:
                    tasks = len(worker_info['processing'])
                elif 'executing' in worker_info:
                    tasks = len(worker_info['executing'])
                elif 'tasks' in worker_info:
                    tasks = worker_info['tasks']
                
                worker_data.append({
                    'Worker': worker_id.split('/')[-1].split(':')[0],  # Short name
                    'Threads': worker_info.get('nthreads', 0),
                    'Memory (GB)': worker_info.get('memory_limit', 0) / (1024**3),
                    'Tasks': tasks,
                    'CPU %': round(worker_info.get('cpu', 0), 1),
                    'Memory %': round(worker_info.get('memory', 0) / worker_info.get('memory_limit', 1) * 100, 1)
                })
            
            return pd.DataFrame(worker_data)
        except Exception as e:
            # Return empty DataFrame if there's an error
            return pd.DataFrame(columns=['Worker', 'Threads', 'Memory (GB)', 'Tasks', 'CPU %', 'Memory %'])
    
    def display_cluster_metrics(self, container: st.delta_generator.DeltaGenerator):
        """Display cluster metrics in Streamlit"""
        status = self.get_cluster_status()
        
        with container:
            st.subheader("🖥️ Dask Cluster Status")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Workers", status['n_workers'])
            
            with col2:
                st.metric("Total Threads", status['total_threads'])
            
            with col3:
                st.metric("Total Memory", f"{status['total_memory_gb']:.1f} GB")
            
            with col4:
                st.metric("Active Tasks", status['tasks_processing'])
            
            # Worker details
            st.subheader("Worker Status")
            worker_df = self.get_worker_status()
            st.dataframe(worker_df, use_container_width=True)
            
            # Dashboard link
            st.info(f"📊 [Open Dask Dashboard]({status['dashboard_link']}) for detailed monitoring")
    
    def estimate_completion_time(self, completed_tasks: int, total_tasks: int) -> Optional[str]:
        """Estimate time to completion based on current progress"""
        if completed_tasks == 0:
            return None
        
        elapsed_time = time.time() - self.start_time
        rate = completed_tasks / elapsed_time  # tasks per second
        
        remaining_tasks = total_tasks - completed_tasks
        estimated_seconds = remaining_tasks / rate if rate > 0 else 0
        
        # Format time
        if estimated_seconds < 60:
            return f"{int(estimated_seconds)} seconds"
        elif estimated_seconds < 3600:
            return f"{int(estimated_seconds / 60)} minutes"
        else:
            hours = int(estimated_seconds / 3600)
            minutes = int((estimated_seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
    
    def log_task_completion(self, task_id: str, duration: float, rows_processed: int):
        """Log task completion for performance tracking"""
        self.task_history.append({
            'task_id': task_id,
            'duration': duration,
            'rows_processed': rows_processed,
            'timestamp': time.time()
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.task_history:
            return {}
        
        df = pd.DataFrame(self.task_history)
        
        total_duration = time.time() - self.start_time
        total_rows = df['rows_processed'].sum()
        avg_task_duration = df['duration'].mean()
        
        return {
            'total_duration_seconds': total_duration,
            'total_rows_processed': total_rows,
            'throughput_rows_per_second': total_rows / total_duration if total_duration > 0 else 0,
            'average_task_duration': avg_task_duration,
            'total_tasks_completed': len(self.task_history)
        }


def create_progress_tracker(total_chunks: int, status_placeholder: st.delta_generator.DeltaGenerator):
    """Create a progress tracking display"""
    
    progress_container = status_placeholder.container()
    
    with progress_container:
        st.subheader("📈 Processing Progress")
        
        # Progress bar
        progress_bar = st.progress(0.0)
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            chunks_metric = st.metric("Chunks Completed", "0", f"of {total_chunks}")
        
        with col2:
            rows_metric = st.metric("Rows Processed", "0", "")
        
        with col3:
            time_metric = st.metric("Elapsed Time", "0s", "")
        
        with col4:
            eta_metric = st.metric("Estimated Time", "Calculating...", "")
        
        # Status text
        status_text = st.empty()
    
    return {
        'progress_bar': progress_bar,
        'chunks_metric': chunks_metric,
        'rows_metric': rows_metric,
        'time_metric': time_metric,
        'eta_metric': eta_metric,
        'status_text': status_text
    }


def update_progress_tracker(tracker: Dict, completed: int, total_chunks: int, 
                          total_rows: int, monitor: DaskMonitor):
    """Update progress tracker display"""
    
    # Update progress bar
    progress = completed / total_chunks if total_chunks > 0 else 0
    tracker['progress_bar'].progress(progress)
    
    # Update metrics
    elapsed = time.time() - monitor.start_time
    
    # Format elapsed time
    if elapsed < 60:
        elapsed_str = f"{int(elapsed)}s"
    elif elapsed < 3600:
        elapsed_str = f"{int(elapsed / 60)}m {int(elapsed % 60)}s"
    else:
        hours = int(elapsed / 3600)
        minutes = int((elapsed % 3600) / 60)
        elapsed_str = f"{hours}h {minutes}m"
    
    # Update displays
    tracker['chunks_metric'].metric("Chunks Completed", f"{completed}", f"of {total_chunks}")
    tracker['rows_metric'].metric("Rows Processed", f"{total_rows:,}", "")
    tracker['time_metric'].metric("Elapsed Time", elapsed_str, "")
    
    # Estimate completion time
    eta = monitor.estimate_completion_time(completed, total_chunks)
    if eta:
        tracker['eta_metric'].metric("Estimated Time", eta, "remaining")
    else:
        tracker['eta_metric'].metric("Estimated Time", "Calculating...", "")
    
    # Update status text
    if completed < total_chunks:
        tracker['status_text'].write(
            f"⚡ Processing chunk {completed + 1} of {total_chunks}... "
            f"({progress * 100:.1f}% complete)"
        )
    else:
        tracker['status_text'].write("✅ All chunks processed successfully!")


def display_performance_summary(monitor: DaskMonitor, container: st.delta_generator.DeltaGenerator):
    """Display final performance summary"""
    summary = monitor.get_performance_summary()
    
    if not summary:
        return
    
    with container:
        st.subheader("🎯 Performance Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Time", 
                f"{summary['total_duration_seconds']:.1f} seconds"
            )
        
        with col2:
            st.metric(
                "Rows Processed", 
                f"{summary['total_rows_processed']:,}"
            )
        
        with col3:
            st.metric(
                "Throughput", 
                f"{summary['throughput_rows_per_second']:.0f} rows/sec"
            )
        
        # Additional metrics
        st.write(f"**Average task duration:** {summary['average_task_duration']:.2f} seconds")
        st.write(f"**Total tasks completed:** {summary['total_tasks_completed']}")