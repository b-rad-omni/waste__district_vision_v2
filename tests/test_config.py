"""Tests for configuration utilities."""

import pytest
import yaml
from src.utils.config import Config


def test_config_load(temp_dir, sample_config):
    """Test configuration loading."""
    config_file = temp_dir / "test_config.yaml"
    
    with open(config_file, 'w') as f:
        yaml.dump(sample_config, f)
    
    config = Config(str(config_file))
    
    assert config.get('model.name') == 'yolov8n.pt'
    assert config.get('training.epochs') == 1


def test_config_get_default():
    """Test getting configuration with default value."""
    config = Config()
    
    assert config.get('nonexistent.key', 'default') == 'default'


def test_config_update():
    """Test configuration update."""
    config = Config()
    config.update({'new_key': 'new_value'})
    
    assert config.get('new_key') == 'new_value'
