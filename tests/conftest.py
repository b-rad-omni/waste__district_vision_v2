"""Test configuration and fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'model': {
            'name': 'yolov8n.pt',
            'pretrained': True
        },
        'training': {
            'epochs': 1,
            'batch_size': 1,
            'learning_rate': 0.01
        }
    }
