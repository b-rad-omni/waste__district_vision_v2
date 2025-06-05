"""
Model Registry Management Tool
"""
import argparse
from utils.model_registry_loader import ModelRegistry

def main():
    parser = argparse.ArgumentParser(description="Manage model registry")
    parser.add_argument('--list', action='store_true', help='List all models')
    parser.add_argument('--info', type=str, help='Show info for specific model')
    parser.add_argument('--path', type=str, help='Get path for specific model')
    
    args = parser.parse_args()
    
    registry = ModelRegistry()
    
    if args.list:
        registry.print_model_info()
    elif args.info:
        registry.print_model_info(args.info)
    elif args.path:
        print(registry.get_model_path(args.path))
    else:
        print("Use --list, --info <model>, or --path <model>")

if __name__ == "__main__":
    main()