# Setup Guide for SteamLensAI

## Prerequisites

- Python 3.8+
- pip or uv (recommended)
- CUDA toolkit (optional, for GPU acceleration)

## Installation

### Option 1: Using pip

```bash
# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Option 2: Using uv (recommended)

```bash
# Create virtual environment and install
uv sync

# Or install dependencies only
uv pip install -r requirements.txt
```

## Running the Application

### Development Mode

```bash
# Navigate to the project directory
cd /path/to/SteamLensAI/steamLensAI

# Run the Streamlit app
streamlit run src/steamlensai/app.py
```

The application will be available at `http://localhost:8501`

### Production Mode

For production deployment, consider:
- Using a Streamlit Cloud account
- Docker containerization
- Gunicorn with a Streamlit-compatible server

## Configuration

### System Configuration
The application automatically detects your system resources:
- RAM (total memory)
- CPU cores
- GPU availability and memory

Edit `src/steamlensai/config/app_config.py` to manually override:

```python
# Processing configuration
PROCESSING_CONFIG = {
    'n_workers': 6,           # Number of Dask workers
    'threads_per_worker': 2,  # Threads per worker
    'memory_per_worker': '4GB',  # Memory allocation
    'chunk_size': 5000,       # Processing chunk size
    'use_gpu': True,          # Enable GPU
    'gpu_batch_size': 512,    # GPU batch size
}

# Summarization configuration
HARDWARE_CONFIG = {
    'worker_count': 8,
    'memory_per_worker': '4GB',
    'gpu_batch_size': 512,
    'model_name': 'sshleifer/distilbart-cnn-12-6',
    'chunk_size': 800,
    'max_summary_length': 300,
    'min_summary_length': 80,
    'num_beams': 6,
}
```

### Theme File
The application requires a `game_themes.json` file:

```json
{
  "123456": {
    "action": ["combat", "fast-paced"],
    "story": ["narrative", "plot"],
    "graphics": ["visuals", "engine"]
  }
}
```

Upload this file via the Streamlit UI sidebar.

## Input Data Format

### Parquet Files
Required columns:
- `steam_appid`: Game application ID
- `review`: Review text
- `review_language`: Language code
- `voted_up`: Boolean sentiment indicator

## Output Files

Generated files are saved to `artifacts/outputs/`:
- `sentiment_report.csv`: Processed reviews by theme
- `sentiment_summaries.csv`: Generated summaries

## Troubleshooting

### GPU Issues
If GPU processing fails:
1. Check CUDA toolkit installation: `nvidia-smi`
2. Verify PyTorch GPU support: `python -c "import torch; print(torch.cuda.is_available())"`
3. Disable GPU in config: Set `use_gpu: False`

### Memory Issues
If you experience out-of-memory errors:
1. Reduce `n_workers` in PROCESSING_CONFIG
2. Reduce `memory_per_worker`
3. Reduce `chunk_size` for file processing
4. Reduce `gpu_batch_size`

### Dask Dashboard
Access the Dask dashboard during processing:
- URL is printed in console
- Shows worker status and task graphs
- Useful for performance monitoring

## Development Setup

### Code Quality Tools
Install development dependencies:
```bash
pip install -e ".[dev]"
```

Run code quality checks:
```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

## Docker Setup (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/steamlensai/app.py"]
```

Build and run:
```bash
docker build -t steamlensai .
docker run -p 8501:8501 steamlensai
```
