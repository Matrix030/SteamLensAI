"""Pytest configuration and shared fixtures."""

import pytest
import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample DataFrame for testing."""
    import pandas as pd
    return pd.DataFrame({
        'steam_appid': [1, 2, 3],
        'review': ['Great game!', 'Not bad', 'Terrible'],
        'voted_up': [True, True, False]
    })


@pytest.fixture
def temp_config(tmp_path):
    """Fixture providing temporary configuration."""
    config = {
        'test_dir': str(tmp_path),
        'output_dir': str(tmp_path / 'output'),
    }
    return config
