#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os
import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
from typing import Tuple, Optional, List, Dict, Any

from ..config.app_config import POTENTIAL_NAME_FIELDS, PARQUET_COLUMNS, BLOCKSIZE_THRESHOLDS, BLOCKSIZES, DEFAULT_LANGUAGE

from typing import Optional, Tuple
import os
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

def extract_appid_and_name_from_parquet(
    file_path: str
) -> Tuple[Optional[int], Optional[str]]:
    # Define which columns we want
    try:
        pf = pq.ParquetFile(file_path)
        available = set(pf.schema.names)
        cols = ['steam_appid'] + [f for f in POTENTIAL_NAME_FIELDS if f in available]
        
        # Read the first row group (small and fast) into a DataFrame
        df = pf.read_row_group(0, columns=cols).to_pandas()
    except Exception as e:
        st.error(f"Error reading {os.path.basename(file_path)}: {e}")
        return None, None

    # Ensure we have at least one valid app ID
    df = df.dropna(subset=['steam_appid'])
    if df.empty:
        return None, None

    # Most common app_id
    try:
        app_id = int(df['steam_appid'].mode()[0])
    except Exception:
        return None, None

    # Look for the first non‑empty name field
    for name_col in POTENTIAL_NAME_FIELDS:
        if name_col in df.columns:
            vals = df[name_col].dropna()
            if not vals.empty:
                return app_id, vals.mode()[0]

    return app_id, None


def extract_appid_from_parquet(file_path: str) -> Optional[int]:
    
    app_id, _ = extract_appid_and_name_from_parquet(file_path)
    return app_id

def determine_blocksize(file_size_gb: float) -> str:
    
    if file_size_gb > BLOCKSIZE_THRESHOLDS['large']:
        return BLOCKSIZES['large']
    elif file_size_gb > BLOCKSIZE_THRESHOLDS['medium']:
        return BLOCKSIZES['medium']
    else:
        return BLOCKSIZES['small']

def load_theme_dictionary(themes_file: str) -> Dict[int, Dict[str, List[str]]]:
    
    import json
    
    try:
        with open(themes_file, 'r') as f:
            raw = json.load(f)
        return {int(appid): themes for appid, themes in raw.items()}
    except FileNotFoundError:
        st.error(f"Theme file '{themes_file}' not found. Please upload it first.")
        return {}
    except Exception as e:
        st.error(f"Error loading theme dictionary: {str(e)}")
        return {} 