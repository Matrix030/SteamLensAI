#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import pandas as pd
from tqdm.auto import tqdm
from typing import List, Dict, Tuple, Any, Union, Optional
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from dask.distributed import Future

def prepare_partition(start_idx: int, end_idx: int, final_report: pd.DataFrame) -> pd.DataFrame:
    
    return final_report.iloc[start_idx:end_idx].copy()

def process_partition(partition_df: pd.DataFrame, worker_id: int, hardware_config: Dict[str, Any], 
                     model_dataset_name: Optional[str] = None, 
                     tokenizer_dataset_name: Optional[str] = None) -> List[Tuple[int, str, str]]:
    
    
    # Import needed packages
    from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
    import torch
    from dask.distributed import get_worker
    
    # Get summary length parameters from config (with fallback defaults)
    max_length = hardware_config.get('max_summary_length', 150)  # Increased default
    min_length = hardware_config.get('min_summary_length', 40)   # Increased default
    num_beams = hardware_config.get('num_beams', 4)              # Better quality
    
    print(f"Worker {worker_id} using summary lengths: min={min_length}, max={max_length}, beams={num_beams}")
    
    # Load model components with optimal settings
    print(f"Worker {worker_id} initializing with optimized settings")
    
    # [Previous model loading code remains the same...]
    if model_dataset_name is not None and tokenizer_dataset_name is not None:
        worker = get_worker()
        model = worker.client.get_dataset(model_dataset_name)
        tokenizer = worker.client.get_dataset(tokenizer_dataset_name)
        print(f"Worker {worker_id} using model and tokenizer from published datasets")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            model = model.to(device).half()
            print(f"Worker {worker_id} moved model to {device} with half precision")
        else:
            model = model.to(device)
            print(f"Worker {worker_id} moved model to {device}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(hardware_config['model_name'])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSeq2SeqLM.from_pretrained(
            hardware_config['model_name'],
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    
    # Create optimized pipeline
    summarizer = pipeline(
        task='summarization',
        model=model,
        tokenizer=tokenizer,
        framework='pt',
        model_kwargs={
            "use_cache": True,
            "return_dict_in_generate": True
        }
    )
    
    # Report GPU status if available
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"Worker {worker_id}: GPU Memory: {gpu_mem:.2f}GB allocated")
    
    # Updated batch processing function with configurable parameters
    def process_chunks_batched(chunks: List[str]) -> List[str]:
        """Process chunks in large batches for GPU with configurable summary length"""
        all_summaries = []
        
        for i in range(0, len(chunks), hardware_config['gpu_batch_size']):
            batch = chunks[i:i+hardware_config['gpu_batch_size']]
            batch_summaries = summarizer(
                batch,
                max_length=max_length,      # Use configurable max length
                min_length=min_length,      # Use configurable min length
                truncation=True,
                do_sample=False,
                num_beams=num_beams         # Use configurable beam search
            )
            all_summaries.extend([s["summary_text"] for s in batch_summaries])
            
            if i % (hardware_config['gpu_batch_size'] * 3) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                    
        return all_summaries
    
    # Updated hierarchical summary function with configurable length
    def hierarchical_summary(reviews: List[str]) -> str:
        """Create hierarchical summary with configurable length"""
        if not reviews or not isinstance(reviews, list) or len(reviews) == 0:
            return "No reviews available for summarization."
        
        # Fast path for small review sets
        if len(reviews) <= hardware_config['chunk_size']:
            doc = "\n\n".join(reviews)
            return summarizer(
                doc,
                max_length=max_length,      # Use configurable max length
                min_length=min_length,      # Use configurable min length
                truncation=True,
                do_sample=False,
                num_beams=num_beams         # Use configurable beam search
            )[0]['summary_text']
        
        # Process larger review sets with optimized chunking
        all_chunks = []
        for i in range(0, len(reviews), hardware_config['chunk_size']):
            batch = reviews[i:i+hardware_config['chunk_size']]
            text = "\n\n".join(batch)
            all_chunks.append(text)
        
        # Process chunks with optimized batching
        intermediate_summaries = process_chunks_batched(all_chunks)
        
        # Create final summary with longer length
        joined = " ".join(intermediate_summaries)
        return summarizer(
            joined,
            max_length=max_length,      # Use configurable max length
            min_length=min_length,      # Use configurable min length
            truncation=True,
            do_sample=False,
            num_beams=num_beams         # Use configurable beam search
        )[0]['summary_text']
    
    # [Rest of the function remains the same...]
    results = []
    
    with tqdm(total=len(partition_df), desc=f"Worker {worker_id}", position=worker_id) as pbar:
        for idx, row in partition_df.iterrows():
            positive_reviews = row['Positive_Reviews'] if isinstance(row['Positive_Reviews'], list) else []
            negative_reviews = row['Negative_Reviews'] if isinstance(row['Negative_Reviews'], list) else []
            
            positive_summary = hierarchical_summary(positive_reviews) if positive_reviews else "No positive reviews available."
            negative_summary = hierarchical_summary(negative_reviews) if negative_reviews else "No negative reviews available."
            
            results.append((idx, positive_summary, negative_summary))
            
            if len(results) % hardware_config['cleanup_frequency'] == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            pbar.update(1)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Worker {worker_id} completed successfully")
    return results

def update_main_progress(futures: List[Any], progress_bar: Any, stop_flag: List[bool], final_report_length: int) -> None:
    
    import time
    start_time = time.time()
    
    while not stop_flag[0]:
        # Count completed futures
        completed_count = sum(f.status == 'finished' for f in futures)
        completed_percentage = completed_count / len(futures) if len(futures) > 0 else 0
        
        # Update progress bar
        progress_bar.progress(completed_percentage)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Only check every 2 seconds to reduce overhead
        time.sleep(2) 