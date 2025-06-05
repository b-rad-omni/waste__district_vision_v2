import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ModularConfigManager:
    """Manages component-specific configuration loading and merging"""
    
    def __init__(self, project_root: Path = None):
        """Initialize config manager with project root"""
        if project_root is None:
            # Auto-detect project root (assumes this file is in src/utils/)
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.configs_dir = self.project_root / "configs"
    
    def load_component_config(self, component_name: str, custom_config_path: str = None) -> Dict[str, Any]:
        """
        Load configuration for a specific component with proper precedence:
        1. default_{component}.yaml (base)
        2. local_{component}.yaml (if exists) 
        3. custom config file (if specified)
        4. Environment variables (highest priority)
        """
        config = {}
        component_dir = self.configs_dir / component_name
        
        # 1. Load default component config (must exist)
        default_config_path = component_dir / f"default_{component_name}.yaml"
        if not default_config_path.exists():
            raise FileNotFoundError(f"Default config not found: {default_config_path}")
        
        config = self._load_yaml_file(default_config_path)
        print(f"✅ Loaded default {component_name} config from: {default_config_path}")
        
        # 2. Load local component config (optional)
        local_config_path = component_dir / f"local_{component_name}.yaml"
        if local_config_path.exists():
            local_config = self._load_yaml_file(local_config_path)
            config = self._deep_merge(config, local_config)
            print(f"✅ Loaded local {component_name} config from: {local_config_path}")
        else:
            print(f"ℹ️  No local {component_name} config found (optional): {local_config_path}")
        
        # 3. Load custom config (optional)
        if custom_config_path:
            custom_path = Path(custom_config_path)
            if not custom_path.is_absolute():
                custom_path = self.project_root / custom_path
            
            if custom_path.exists():
                custom_config = self._load_yaml_file(custom_path)
                config = self._deep_merge(config, custom_config)
                print(f"✅ Loaded custom {component_name} config from: {custom_path}")
            else:
                raise FileNotFoundError(f"Custom config not found: {custom_path}")
        
        # 4. Apply environment variable overrides
        config = self._apply_env_overrides(config, component_name)
        
        # 5. Resolve paths to absolute paths
        config = self._resolve_paths(config)
        
        return config
    
    def load_shared_config(self) -> Dict[str, Any]:
        """Load shared configuration that applies to all components"""
        shared_dir = self.configs_dir / "shared"
        shared_config_path = shared_dir / "default_shared.yaml"
        
        if shared_config_path.exists():
            config = self._load_yaml_file(shared_config_path)
            print(f"✅ Loaded shared config from: {shared_config_path}")
            
            # Check for local shared overrides
            local_shared_path = shared_dir / "local_shared.yaml"
            if local_shared_path.exists():
                local_config = self._load_yaml_file(local_shared_path)
                config = self._deep_merge(config, local_config)
                print(f"✅ Loaded local shared config from: {local_shared_path}")
            
            return self._resolve_paths(config)
        else:
            print(f"ℹ️  No shared config found (optional): {shared_config_path}")
            return {}
    
    def load_full_config(self, component_name: str, custom_config_path: str = None) -> Dict[str, Any]:
        """Load both shared and component-specific configuration"""
        # Start with shared config
        config = self.load_shared_config()
        
        # Merge with component-specific config
        component_config = self.load_component_config(component_name, custom_config_path)
        config = self._deep_merge(config, component_config)
        
        return config
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML file safely"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {file_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading {file_path}: {e}")
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config: Dict, component_name: str) -> Dict:
        """Apply environment variable overrides for specific component"""
        # Component-specific environment mappings
        env_mappings = {
            'data_collection': {
                'CAMERA_INDEX': ['camera', 'index'],
                'HEADLESS': ['system', 'headless'],
                'PRIMARY_DIR': ['storage', 'primary_dir'],
                'FALLBACK_DIR': ['storage', 'fallback_dir'],
                'BURST_DURATION': ['collection', 'burst_duration'],
                'MIN_FRAMES_PER_HOUR': ['collection', 'min_frames_per_hour']
            },
            'inference': {
                'MODEL_PATH': ['model', 'path'],
                'CONFIDENCE_THRESHOLD': ['model', 'confidence_threshold'],
                'BATCH_SIZE': ['inference', 'batch_size']
            },
            'training': {
                'EPOCHS': ['training', 'epochs'],
                'LEARNING_RATE': ['training', 'learning_rate'],
                'BATCH_SIZE': ['training', 'batch_size']
            }
        }
        
        if component_name in env_mappings:
            for env_var, config_path in env_mappings[component_name].items():
                env_value = os.getenv(env_var)
                if env_value is not None:
                    # Navigate to nested config location
                    current = config
                    for key in config_path[:-1]:
                        if key not in current:
                            current[key] = {}
                        current = current[key]
                    
                    # Convert common types
                    if env_var in ['CAMERA_INDEX', 'BATCH_SIZE', 'EPOCHS', 'MIN_FRAMES_PER_HOUR']:
                        env_value = int(env_value)
                    elif env_var in ['HEADLESS']:
                        env_value = env_value.lower() in ('1', 'true', 'yes')
                    elif env_var in ['BURST_DURATION', 'CONFIDENCE_THRESHOLD', 'LEARNING_RATE']:
                        env_value = float(env_value)
                    
                    current[config_path[-1]] = env_value
                    print(f"🔧 Environment override: {env_var} = {env_value}")
        
        return config
    
    def _resolve_paths(self, config: Dict) -> Dict:
        """Convert relative paths to absolute paths"""
        # Common path keys that need resolution
        path_keys = [
            ['storage', 'primary_dir'],
            ['storage', 'fallback_dir'], 
            ['model', 'path'],
            ['data', 'train_dir'],
            ['data', 'val_dir'],
            ['logging', 'log_dir']
        ]
        
        for path_components in path_keys:
            current = config
            # Navigate to the path location
            for key in path_components[:-1]:
                if key in current and isinstance(current[key], dict):
                    current = current[key]
                else:
                    break
            else:
                # If we successfully navigated, resolve the path
                final_key = path_components[-1]
                if final_key in current and current[final_key]:
                    path = Path(current[final_key])
                    if not path.is_absolute():
                        # Relative to project root
                        path = self.project_root / path
                    current[final_key] = str(path.resolve())
        
        return config


# Convenience functions for easy usage
def load_data_collection_config(custom_config_path: str = None) -> Dict[str, Any]:
    """Load data collection configuration"""
    manager = ModularConfigManager()
    return manager.load_full_config('data_collection', custom_config_path)

def load_inference_config(custom_config_path: str = None) -> Dict[str, Any]:
    """Load inference configuration"""
    manager = ModularConfigManager()
    return manager.load_full_config('inference', custom_config_path)

def load_training_config(custom_config_path: str = None) -> Dict[str, Any]:
    """Load training configuration"""
    manager = ModularConfigManager()
    return manager.load_full_config('training', custom_config_path)