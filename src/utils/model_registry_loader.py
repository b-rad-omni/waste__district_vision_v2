"""
Model Registry - Easy model loading and management
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

class ModelRegistry:
    def __init__(self, registry_path: str = "models/model_registry.json"):
        self.registry_path = registry_path
        self.models = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load the model registry from JSON file."""
        try:
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model registry not found at {self.registry_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in model registry: {self.registry_path}")
    
    def get_model_path(self, model_name: str) -> str:
        """Get the full path to a model by name or alias."""
        model_info = self.get_model_info(model_name)
        model_path = os.path.join("models", model_info["path"])
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        return model_path
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get full information about a model by name or alias."""
        # Check if it's a direct name match
        for model in self.models["models"]:
            if model["name"] == model_name:
                return model
        
        # Check if it's an alias
        for model in self.models["models"]:
            if model_name in model.get("alias", []):
                return model
        
        # If no match found
        available = self.list_models()
        raise ValueError(f"Model '{model_name}' not found. Available: {available}")
    
    def list_models(self) -> List[str]:
        """List all available model names and aliases."""
        models = []
        for model in self.models["models"]:
            models.append(model["name"])
            models.extend(model.get("alias", []))
        return sorted(set(models))
    
    def get_default_model(self) -> str:
        """Get the default model name."""
        return self.models.get("default_model", "latest")
    
    def print_model_info(self, model_name: str = None):
        """Print detailed information about a model or all models."""
        if model_name:
            model = self.get_model_info(model_name)
            print(f"\n📊 Model: {model['name']}")
            print(f"   Description: {model['description']}")
            print(f"   Path: {model['path']}")
            print(f"   Status: {model['status']}")
            print(f"   Metrics: mAP={model['metrics']['mAP']}, Precision={model['metrics']['precision']}")
            print(f"   Trained: {model['training']['trained_on']} ({model['training']['epochs']} epochs)")
        else:
            print("\n📋 Available Models:")
            for model in self.models["models"]:
                status_emoji = "🟢" if model["status"] == "production" else "🟡" if model["status"] == "experimental" else "🔵"
                print(f"   {status_emoji} {model['name']}: {model['description']}")
                if model.get("alias"):
                    print(f"      Aliases: {', '.join(model['alias'])}")
    
    def add_model(self, model_info: Dict):
        """Add a new model to the registry."""
        self.models["models"].append(model_info)
        self._save_registry()
    
    def _save_registry(self):
        """Save the registry back to JSON file."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.models, f, indent=2)

# Convenience function
def load_model_by_name(model_name: str = None):
    """Load a YOLO model by registry name. If None, loads default."""
    from ultralytics import YOLO
    
    registry = ModelRegistry()
    
    if model_name is None:
        model_name = registry.get_default_model()
    
    model_path = registry.get_model_path(model_name)
    print(f"Loading model: {model_name} from {model_path}")
    
    return YOLO(model_path)