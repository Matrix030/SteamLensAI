#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def get_theme_embeddings(app_ids: List[int], game_themes: Dict[int, Dict[str, List[str]]], 
                         embedder: SentenceTransformer) -> Dict[int, np.ndarray]:
    
    # Move the model to the appropriate device
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder.to(device)
        
    embeddings = {}
    for appid in app_ids:
        if appid not in embeddings and appid in game_themes:
            emb_list = []
            for theme, seeds in game_themes[appid].items():
                seed_emb = embedder.encode(seeds, convert_to_numpy=True)
                emb_list.append(seed_emb.mean(axis=0))
            if emb_list:  # Ensure there's at least one embedding
                embeddings[appid] = np.vstack(emb_list)
    return embeddings

def assign_topic(df_partition: pd.DataFrame, game_themes: Dict[int, Dict[str, List[str]]], 
                 embedder: SentenceTransformer) -> pd.DataFrame:
    
    # If no rows, return as-is
    if df_partition.empty:
        df_partition['topic_id'] = []
        return df_partition
    
    # Get unique app IDs in this partition
    app_ids = df_partition['steam_appid'].unique().tolist()
    app_ids = [int(appid) for appid in app_ids]
    
    # Get embeddings only for app IDs in this partition
    local_theme_embeddings = get_theme_embeddings(app_ids, game_themes, embedder)
    
    # Move the model to the appropriate device
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder.to(device)
    
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