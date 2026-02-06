# SteamLensAI Architecture

## Overview

SteamLensAI is a Streamlit-based application for analyzing and summarizing Steam game reviews using distributed computing and NLP techniques.

## Directory Structure

```
src/steamlensai/
├── app.py                  # Main Streamlit entry point
├── core/                   # Core processing logic
│   ├── process_files.py    # File processing with Dask
│   ├── topic_assignment.py # Semantic topic assignment
│   ├── summarization.py    # Review summarization
│   └── summarize_processor.py # Summarization orchestration
├── data/                   # Data handling
│   └── data_loader.py      # Parquet file loading utilities
├── ui/                     # Streamlit UI components
│   ├── sidebar.py         # Sidebar configuration
│   ├── upload_tab.py      # File upload interface
│   ├── summarize_tab.py   # Summarization interface
│   └── results_tab.py     # Results display
├── config/                # Configuration
│   ├── app_config.py      # Application configuration
│   └── game_themes.json   # Game theme definitions
└── utils/                 # Utility functions
    ├── dask_monitor.py    # Dask cluster monitoring
    └── system_utils.py    # System resource utilities
```

## Key Components

### Core Processing Pipeline
1. **File Processing** (`core/process_files.py`)
   - Reads parquet files using Dask
   - Validates app IDs against theme dictionary
   - Performs topic assignment to reviews

2. **Topic Assignment** (`core/topic_assignment.py`)
   - Uses Sentence Transformers for semantic embeddings
   - Calculates cosine similarity with game themes
   - Assigns reviews to most relevant themes

3. **Summarization** (`core/summarization.py`)
   - Generates concise summaries for positive/negative reviews
   - Uses DistilBART for abstractive summarization
   - Handles GPU acceleration when available

4. **Summarization Orchestration** (`core/summarize_processor.py`)
   - Manages the summarization workflow
   - Handles Dask cluster coordination
   - Manages hardware resources

### User Interface
- **Streamlit-based** for easy interaction
- **Three-tab interface**: Upload & Process, Summarize, Results
- **Real-time progress monitoring** with Dask dashboard integration
- **System configuration detection** for automatic optimization

### Configuration
- **Hardware-aware** resource allocation
- **Dynamic worker scaling** based on available resources
- **Adaptive batch sizing** for GPU processing
- **Configurable** processing parameters

## Data Flow

```
Parquet Files
    ↓
[File Processing]
    ├─→ Extract app ID & game name
    ├─→ Validate against themes
    ├─→ Filter English reviews
    └─→ Assign to themes
    ↓
[Topic Assignment]
    ├─→ Generate embeddings
    ├─→ Calculate similarity
    └─→ Assign themes
    ↓
[Summarization]
    ├─→ Separate positive/negative
    ├─→ Batch processing
    └─→ Generate summaries
    ↓
CSV Output
```

## Technology Stack

- **Streamlit**: Web UI framework
- **Dask**: Distributed computing
- **Pandas**: Data manipulation
- **PyArrow**: Parquet file handling
- **Sentence Transformers**: Text embeddings
- **Transformers**: NLP models
- **PyTorch**: Deep learning backend
- **scikit-learn**: ML utilities

## Performance Optimization

### System Detection
- RAM, CPU cores, GPU availability
- GPU memory allocation
- Adaptive configuration based on hardware

### Processing Optimization
- Configurable worker count
- Memory-efficient chunk processing
- GPU batch size optimization
- Parallel processing across workers

### Memory Management
- Automatic GPU memory cleanup
- Streaming parquet file reading
- Temporary file compression
- Configurable cleanup frequency

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run src/steamlensai/app.py
```

## Configuration

Edit `src/steamlensai/config/app_config.py` to:
- Adjust processing parameters
- Change model selections
- Modify batch sizes
- Update theme file paths
