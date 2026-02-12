#!/usr/bin/env python3


from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Per-worker caches. Dask workers run in separate processes, so this is safe and local.
_THEME_EMBEDDING_CACHE: Dict[Tuple[int, int], np.ndarray] = {}
_EMBEDDER_DEVICE_CACHE: Dict[int, str] = {}


def _ensure_embedder_device(embedder: SentenceTransformer) -> None:
    import torch

    embedder_key = id(embedder)
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    if _EMBEDDER_DEVICE_CACHE.get(embedder_key) != target_device:
        embedder.to(target_device)
        _EMBEDDER_DEVICE_CACHE[embedder_key] = target_device


def _build_theme_embedding(
    appid: int, game_themes: Dict[int, Dict[str, List[str]]], embedder: SentenceTransformer
) -> np.ndarray:
    theme_seed_lists = list(game_themes.get(appid, {}).values())
    if not theme_seed_lists:
        return np.empty((0, 0), dtype=np.float32)

    all_seeds: List[str] = []
    lengths: List[int] = []
    for seeds in theme_seed_lists:
        normalized_seeds = [str(seed) for seed in seeds if seed]
        all_seeds.extend(normalized_seeds)
        lengths.append(len(normalized_seeds))

    if not all_seeds:
        return np.empty((0, 0), dtype=np.float32)

    seed_embeddings = embedder.encode(
        all_seeds, convert_to_numpy=True, batch_size=64, show_progress_bar=False
    )

    theme_embeddings: List[np.ndarray] = []
    start_idx = 0
    for length in lengths:
        if length <= 0:
            theme_embeddings.append(np.zeros(seed_embeddings.shape[1], dtype=np.float32))
            continue
        end_idx = start_idx + length
        theme_embeddings.append(seed_embeddings[start_idx:end_idx].mean(axis=0))
        start_idx = end_idx

    return np.vstack(theme_embeddings) if theme_embeddings else np.empty((0, 0), dtype=np.float32)


def get_theme_embeddings(
    app_ids: List[int], game_themes: Dict[int, Dict[str, List[str]]], embedder: SentenceTransformer
) -> Dict[int, np.ndarray]:
    _ensure_embedder_device(embedder)

    embeddings: Dict[int, np.ndarray] = {}
    embedder_key = id(embedder)

    for appid in app_ids:
        if appid not in game_themes:
            continue
        cache_key = (embedder_key, appid)
        cached_embedding = _THEME_EMBEDDING_CACHE.get(cache_key)
        if cached_embedding is None:
            cached_embedding = _build_theme_embedding(appid, game_themes, embedder)
            _THEME_EMBEDDING_CACHE[cache_key] = cached_embedding
        if cached_embedding.size:
            embeddings[appid] = cached_embedding

    return embeddings


def assign_topic(
    df_partition: pd.DataFrame,
    game_themes: Dict[int, Dict[str, List[str]]],
    embedder: SentenceTransformer,
) -> pd.DataFrame:

    # If no rows, return as-is
    if df_partition.empty:
        df_partition["topic_id"] = pd.Series(dtype=np.int32)
        return df_partition

    # Get unique app IDs in this partition
    app_ids = df_partition["steam_appid"].astype(int).unique().tolist()

    # Get embeddings only for app IDs in this partition
    local_theme_embeddings = get_theme_embeddings(app_ids, game_themes, embedder)

    _ensure_embedder_device(embedder)

    reviews = df_partition["review"].fillna("").astype(str).tolist()
    # Compute embeddings in one go with batching
    review_embeds = embedder.encode(
        reviews, convert_to_numpy=True, batch_size=64, show_progress_bar=False
    )

    # Assign topics in app-id batches to avoid per-row cosine calls.
    appid_array = df_partition["steam_appid"].astype(int).to_numpy()
    topic_ids = np.zeros(len(df_partition), dtype=np.int32)
    for appid in np.unique(appid_array):
        idx = np.where(appid_array == appid)[0]
        theme_embs = local_theme_embeddings.get(int(appid))
        if theme_embs is None or theme_embs.size == 0:
            continue
        sims = cosine_similarity(review_embeds[idx], theme_embs)
        topic_ids[idx] = sims.argmax(axis=1).astype(np.int32)

    df_partition["topic_id"] = topic_ids.tolist()
    return df_partition
