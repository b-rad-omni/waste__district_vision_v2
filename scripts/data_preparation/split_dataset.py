#!/usr/bin/env python3
"""
Dataset Splitting CLI Tool

Command-line interface for performing stratified dataset splits while maintaining
class distribution across train/validation/test splits for YOLO datasets.

Examples:
    # Basic 70/20/10 split
    python scripts/data_preparation/split_dataset.py --source ./all_data --output ./dataset_splits
    
    # Custom ratios without test set
    python scripts/data_preparation/split_dataset.py --source ./all_data --output ./splits --ratios 0.8 0.2
    
    # Custom ratios with test set
    python scripts/data_preparation/split_dataset.py --source ./all_data --output ./splits --ratios 0.7 0.2 0.1
    
    # With custom class names and random seed
    python scripts/data_preparation/split_dataset.py --source ./all_data --output ./splits --class-map classes.json --seed 123
"""

import argparse
import json
import sys
from pathlib import Path

# Import from our modular source
try:
    from src.data.dataset_splitter import StratifiedDatasetSplitter
except ImportError:
    print("Error: Could not import StratifiedDatasetSplitter. Make sure you've installed the package:")
    print("  pip install -e .")
    sys.exit(1)


def load_class_names(class_map_file: str) -> dict:
    """Load class names from JSON file."""
    try:
        with open(class_map_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Class map file not found: {class_map_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in class map file: {class_map_file}")
        sys.exit(1)


def validate_ratios(ratios: list) -> tuple:
    """Validate and return train/val/test ratios."""
    if len(ratios) == 2:
        train_ratio, val_ratio = ratios
        test_ratio = 0.0
    elif len(ratios) == 3:
        train_ratio, val_ratio, test_ratio = ratios
    else:
        print("Error: Ratios must have 2 values (train, val) or 3 values (train, val, test)")
        sys.exit(1)
    
    if abs(sum(ratios) - 1.0) > 1e-6:
        print(f"Error: Ratios must sum to 1.0, got {sum(ratios)}")
        sys.exit(1)
    
    if any(r < 0 for r in ratios):
        print("Error: All ratios must be positive")
        sys.exit(1)
    
    return train_ratio, val_ratio, test_ratio


def main():
    parser = argparse.ArgumentParser(
        description="Perform stratified dataset splitting for YOLO datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic 70/20/10 split
  python scripts/data_preparation/split_dataset.py --source ./all_data --output ./splits
  
  # 80/20 split (no test set)
  python scripts/data_preparation/split_dataset.py --source ./all_data --output ./splits --ratios 0.8 0.2
  
  # Custom ratios with test set
  python scripts/data_preparation/split_dataset.py --source ./all_data --output ./splits --ratios 0.6 0.3 0.1
  
  # With custom class names and reproducible seed
  python scripts/data_preparation/split_dataset.py --source ./all_data --output ./splits \\
    --class-map classes.json --seed 42 --min-samples 5
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source directory containing .txt label files and corresponding image files'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output base directory (will create train/val/test subdirectories)'
    )
    
    # Split configuration
    parser.add_argument(
        '--ratios',
        type=float,
        nargs='+',
        default=[0.7, 0.2, 0.1],
        help='Train/validation/test ratios (2 or 3 values, must sum to 1.0)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible splits (default: 42)'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=3,
        help='Minimum samples per class required for stratification (default: 3)'
    )
    
    # Configuration options
    parser.add_argument(
        '--class-map',
        type=str,
        help='JSON file mapping class IDs to names (e.g., {"0": "tote", "1": "box"})'
    )
    
    # Output options
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually copying files'
    )
    
    args = parser.parse_args()
    
    # Validate source directory
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source directory does not exist: {args.source}")
        sys.exit(1)
    
    # Check for label files
    txt_files = list(source_path.glob("*.txt"))
    if not txt_files:
        print(f"Error: No .txt label files found in source directory: {args.source}")
        sys.exit(1)
    
    if not args.quiet:
        print(f"Found {len(txt_files)} label files in source directory")
    
    # Validate ratios
    train_ratio, val_ratio, test_ratio = validate_ratios(args.ratios)
    
    # Load class names if provided
    class_names = None
    if args.class_map:
        class_names = load_class_names(args.class_map)
        if not args.quiet:
            print(f"Loaded custom class mapping with {len(class_names)} classes")
    
    # Initialize splitter
    splitter = StratifiedDatasetSplitter(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        min_samples_per_class=args.min_samples,
        random_seed=args.seed
    )
    
    if class_names:
        splitter.set_class_names(class_names)
    
    if not args.quiet:
        print(f"Split ratios: {train_ratio:.1%} train, {val_ratio:.1%} val, {test_ratio:.1%} test")
        print(f"Minimum samples per class: {args.min_samples}")
        print(f"Random seed: {args.seed}")
    
    try:
        if args.dry_run:
            if not args.quiet:
                print("\n🔍 DRY RUN - No files will be copied")
            
            # Collect class samples to show what would happen
            class_samples = splitter.collect_class_samples(args.source)
            train_files, val_files, test_files, split_report = splitter.perform_stratified_split(class_samples)
            
            print(f"\nWould create:")
            print(f"  Train: {len(train_files)} files")
            print(f"  Val: {len(val_files)} files")
            print(f"  Test: {len(test_files)} files")
            
            print(f"\nClass distribution preview:")
            for item in split_report:
                if item['status'] == 'split':
                    print(f"  {item['class_name']}: {item['total']} → {item['train']}|{item['val']}|{item['test']}")
                else:
                    print(f"  {item['class_name']}: {item['total']} samples (insufficient)")
        
        else:
            # Perform actual split
            results = splitter.split_dataset(
                source_dir=args.source,
                output_base=args.output,
                verbose=not args.quiet
            )
            
            if not args.quiet:
                print(f"\n✅ Split complete! Files saved to:")
                for split, path in results['output_dirs'].items():
                    print(f"  {split}: {path}")
    
    except Exception as e:
        print(f"Error during dataset splitting: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()