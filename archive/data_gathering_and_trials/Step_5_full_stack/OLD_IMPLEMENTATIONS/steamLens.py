#!/usr/bin/env python3
# Steam Theme Analysis
# This script analyzes Steam game reviews by clustering them into predefined themes
import os
import json
import random
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask import delayed, compute
from dask.distributed import Client

# dask-ml modules
from dask_ml.feature_extraction.text import HashingVectorizer
from dask_ml.wrappers import Incremental

# scikit-learn
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer



# Reproducibility
random.seed(42)
np.random.seed(42)

# Global constants
N_FEATURES = 2**18    # ~260K dims
BATCH_SIZE = 5000     # partial_fit chunk size
SAMPLE_PER_C = 5000   # TF-IDF docs per cluster

# Game themes dictionary
with open('game_themes.json', 'r') as file:
    GAME_THEMES = json.load(file)

# Convert string keys to integers if needed (JSON only supports string keys)
GAME_THEMES = {int(k): v for k, v in GAME_THEMES.items()}

n_themes = len(GAME_THEMES)


# A single, stateless vectoriser we'll reuse in each task
global_vec = HashingVectorizer(
    n_features=N_FEATURES,
    alternate_sign=False,
    stop_words='english',
    dtype=np.float32
)

@delayed
def analyse_one_game(appid, themes):
    """
    Analyzes reviews for a single game, clustering them into predefined themes.
    
    Args:
        appid (int): Steam application ID
        themes (dict): Dictionary of themes and related keywords
        
    Returns:
        pandas.DataFrame or None: Analysis report for the game, or None if data not available
    """
    

    # 1) Find your parquet file
    CANDIDATE = [
        'parquet_output_indie',
        'parquet_output_theme_combo'
    ]
    path = next((os.path.join(d, f"{appid}.parquet") for d in CANDIDATE 
                if os.path.exists(os.path.join(d, f"{appid}.parquet"))), None)
    if path is None:
        print(f"⚠️ No file for {appid}")
        return None

    # 2) Load & filter (pandas is fine for <30 MB per game)
    df = pd.read_parquet(path, columns=['review', 'votes_up', 'voted_up', 'review_language'])
    df = df[df.review_language=='english'].dropna(subset=['review'])
    df['review'] = df.review.astype(str)
    n = len(df)
    if n == 0:
        print(f"⚠️ No English reviews for {appid}")
        return None

    # Instantiate the vectorizer inside the function
    vec = HashingVectorizer(
        n_features=2**18,
        alternate_sign=False,
        stop_words='english',
        dtype=np.float32
    )

    # 3) Seed centroids
    pseudo = [" ".join(ws) for ws in themes.values()]
    pseudo_sparse = vec.transform(pseudo)                # SciPy CSR
    init_centroids = np.vstack([r.toarray().ravel() for r in pseudo_sparse])

    # 4) Out-of-core clustering
    mbkm = MiniBatchKMeans(
        n_clusters=len(themes),
        init=init_centroids,
        n_init=1,
        random_state=42,
        batch_size=5000
    )
    km = Incremental(mbkm, predict_meta=np.zeros(1, dtype=int))

    labels = np.empty(n, dtype=int)
    for i in range(0, n, 5000):
        block = df['review'].iloc[i:i+5000]
        Xb = vec.transform(block)
        km.partial_fit(Xb)
        labels[i:i+5000] = km.predict(Xb)

    df['topic_id'] = labels

    # 5) Collect all reviews per theme
    reviews_per_theme = {
        tid: df.loc[df.topic_id==tid, 'review'].tolist()
        for tid in range(len(themes))
    }

    # 6) Build your report
    counts = df.groupby('topic_id').review.count()
    likes = df[df.voted_up].groupby('topic_id').review.count()

    report = pd.DataFrame({
        'steam_appid': appid,
        'Theme': list(themes.keys()),
        '#Reviews': counts.values,
        'LikeRatio': (likes/counts*100).round(1).astype(str) + '%',
        'Reviews': [reviews_per_theme[tid] for tid in range(len(themes))]
    })
    return report

def main():
    """Main function to run the analysis process"""
    # Launch a local Dask client
    client = Client()
    print(client)
    
    # Dispatch all games in parallel on threads
    tasks = [analyse_one_game(appid, themes) for appid, themes in GAME_THEMES.items()]
    
    # Use the threaded scheduler so everything runs in-process (no Dask worker death)
    reports = compute(*tasks, scheduler='threads')
    
    # Filter out any None (missing/empty) and concat
    reports = [r for r in reports if r is not None]
    final_report = pd.concat(reports, ignore_index=True)
    
    # Save results to CSV
    final_report.to_csv('steam_theme_reports.csv', index=False)
    print("Analysis complete. Results saved to 'steam_theme_reports.csv'")

    
    # Clean up
    client.close()

if __name__ == "__main__":
    main()