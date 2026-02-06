import os
import psutil
import torch
from pathlib import Path

# Get the project root directory (where streamlit is run from)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Create directories if they don't exist
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_CSV_DIR = ARTIFACTS_DIR / "outputs"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"

os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Streamlit Page Configuration
STREAMLIT_PAGE_CONFIG = {
    "page_title": "steamLensAI",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# System detection
SYSTEM_MEMORY_GB = psutil.virtual_memory().total / (1024**3)
CPU_COUNT = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)
GPU_AVAILABLE = torch.cuda.is_available()
GPU_MEMORY_GB = 0

if GPU_AVAILABLE:
    try:
        # Get GPU memory in GB
        GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except:
        GPU_MEMORY_GB = 0

# Processing configuration optimized for 32GB RAM, 8 CPU, 16GB VRAM
if SYSTEM_MEMORY_GB >= 32:
    # Configuration for high-end systems (like yours)
    PROCESSING_CONFIG = {
        'n_workers': 6,  # Leave 2 cores for system/main process
        'threads_per_worker': 2,  # 2 threads per worker
        'memory_per_worker': '4GB',  # 6 workers * 4GB = 24GB (leaving 8GB for system)
        'chunk_size': 5000,  # Smaller chunks for better parallelization
        'use_gpu': GPU_AVAILABLE and GPU_MEMORY_GB >= 8,  # Use GPU if available with enough memory
        'gpu_batch_size': 512 if GPU_MEMORY_GB >= 16 else 256,  # Larger batches for 16GB VRAM
    }
elif SYSTEM_MEMORY_GB >= 16:
    # Configuration for mid-range systems
    PROCESSING_CONFIG = {
        'n_workers': 8,
        'threads_per_worker': 2,
        'memory_per_worker': '3.5GB',  # 4 workers * 3GB = 12GB
        'chunk_size': 7500,
        'use_gpu': GPU_AVAILABLE and GPU_MEMORY_GB >= 4,
        'gpu_batch_size': 256 if GPU_MEMORY_GB >= 8 else 128,
    }
else:
    # Configuration for low-memory systems
    PROCESSING_CONFIG = {
        'n_workers': 2,
        'threads_per_worker': 2,
        'memory_per_worker': '3GB',  # 2 workers * 3GB = 6GB
        'chunk_size': 10000,
        'use_gpu': False,  # Disable GPU for low-memory systems
        'gpu_batch_size': 64,
    }

# Default hardware configuration for summarization (optimized for your system)
if SYSTEM_MEMORY_GB >= 30 and GPU_MEMORY_GB >= 15.5:
    # Optimized for your 32GB RAM + 16GB VRAM system
    HARDWARE_CONFIG = {
        'worker_count': 8,  # More workers for 32GB system
        'memory_per_worker': '4GB',  # Higher memory per worker
        'gpu_batch_size': 512,  # Larger batch for 16GB VRAM
        'model_name': 'sshleifer/distilbart-cnn-12-6',
        'chunk_size': 800,
        'checkpoint_frequency': 15,
        'cleanup_frequency': 3,
        'max_summary_length': 300,
        'min_summary_length': 80,
        'num_beams': 6,
    }
elif SYSTEM_MEMORY_GB >= 16:
    # Configuration for mid-range systems
    HARDWARE_CONFIG = {
        'worker_count': 8,
        'memory_per_worker': '4GB',
        'gpu_batch_size': 256,
        'model_name': 'sshleifer/distilbart-cnn-12-6',
        'chunk_size': 600,
        'checkpoint_frequency': 20,
        'cleanup_frequency': 5,
        'max_summary_length': 200,
        'min_summary_length': 60,
        'num_beams': 4,
    }
else:
    # Configuration for low-memory systems
    HARDWARE_CONFIG = {
        'worker_count': 2,
        'memory_per_worker': '2GB',
        'gpu_batch_size': 96,
        'model_name': 'sshleifer/distilbart-cnn-12-6',
        'chunk_size': 400,
        'checkpoint_frequency': 25,
        'cleanup_frequency': 10,
        'max_summary_length': 150,
        'min_summary_length': 40,
        'num_beams': 2,
    }

# Default file and path settings
DEFAULT_THEME_FILE = str(Path(__file__).parent / "game_themes.json")
DEFAULT_OUTPUT_PATH = str(OUTPUT_CSV_DIR / "sentiment_summaries.csv")
DEFAULT_INTERIM_PATH = str(OUTPUT_CSV_DIR / "sentiment_report.csv")

# Model settings
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'

# Default language for filtering reviews
DEFAULT_LANGUAGE = 'english'

# Default batch sizes for different numbers of app IDs
APP_ID_BATCH_SIZES = {
    'very_large': 3,    # > 1000 app IDs
    'large': 5,         # > 500 app IDs
    'medium': 10,       # > 100 app IDs
    'small': 100        # <= 100 app IDs
}

# File size thresholds for blocksize determination
BLOCKSIZE_THRESHOLDS = {
    'large': 1.0,       # > 1GB
    'medium': 0.1,      # > 100MB
    'small': 0.0        # <= 100MB
}

# Blocksizes for different file sizes
BLOCKSIZES = {
    'large': '16MB',
    'medium': '32MB',
    'small': '256MB'
}

# Fields that might contain game name in Parquet files
POTENTIAL_NAME_FIELDS = [
    'name', 
    'game_name', 
    'title', 
    'short_description', 
    'about_the_game'
]

# Columns to read from Parquet files
PARQUET_COLUMNS = [
    'steam_appid', 
    'review', 
    'review_language', 
    'voted_up'
]

# Memory management settings
MEMORY_WARNING_THRESHOLD = 0.8  # Warn when memory usage exceeds 80%
MEMORY_CRITICAL_THRESHOLD = 0.9  # Critical when memory usage exceeds 90%

# Temporary file settings
TEMP_FILE_COMPRESSION = 'snappy'  # Fast compression for temp files
CLEANUP_TEMP_FILES = True  # Automatically clean up temp files after processing

# Display system configuration on startup
def display_system_config():
    """Display detected system configuration"""
    config_text = f"""
    🖥️ **System Configuration Detected:**
    - RAM: {SYSTEM_MEMORY_GB:.1f} GB
    - CPU Cores: {CPU_COUNT}
    - GPU: {'Available' if GPU_AVAILABLE else 'Not Available'}
    - GPU Memory: {GPU_MEMORY_GB:.1f} GB
    
    ⚙️ **Processing Configuration:**
    - Workers: {PROCESSING_CONFIG['n_workers']}
    - Memory per Worker: {PROCESSING_CONFIG['memory_per_worker']}
    - Chunk Size: {PROCESSING_CONFIG['chunk_size']:,} reviews
    - GPU Processing: {'Enabled' if PROCESSING_CONFIG['use_gpu'] else 'Disabled'}
    """
    return config_text
