#!/usr/bin/env python3
"""
Label Analysis CLI Tool

Command-line interface for analyzing YOLO label distributions and dataset balance.
Provides class stratification analysis across training/validation/test splits.

Examples:
    # Basic analysis of train/val splits
    python scripts/data_preparation/analyze_labels.py --train-dir datasets/train --val-dir datasets/val
    
    # Include test split and save to CSV
    python scripts/data_preparation/analyze_labels.py --train-dir datasets/train --val-dir datasets/val --test-dir datasets/test --output analysis.csv
    
    # Use custom class names from JSON file
    python scripts/data_preparation/analyze_labels.py --train-dir datasets/train --val-dir datasets/val --class-map custom_classes.json
    
    # Quick analysis with default paths
    python scripts/data_preparation/analyze_labels.py --dataset-root datasets/
"""

import argparse
import json
import sys
from pathlib import Path

# Import from our modular source
try:
    from src.data.label_analyzer import LabelAnalyzer
except ImportError:
    print("Error: Could not import LabelAnalyzer. Make sure you've installed the package:")
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


def validate_directory(path: str, name: str) -> str:
    """Validate that directory exists and contains .txt files."""
    if not Path(path).exists():
        print(f"Error: {name} directory does not exist: {path}")
        sys.exit(1)
    
    txt_files = list(Path(path).glob("*.txt"))
    if not txt_files:
        print(f"Warning: No .txt label files found in {name} directory: {path}")
    
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze YOLO label distributions across dataset splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic train/val analysis
  python scripts/data_preparation/analyze_labels.py --train-dir ./train --val-dir ./val
  
  # Full analysis with test set and CSV output
  python scripts/data_preparation/analyze_labels.py \\
    --train-dir ./train --val-dir ./val --test-dir ./test \\
    --output class_distribution.csv
  
  # Quick analysis using dataset root (looks for train/val/test subdirs)
  python scripts/data_preparation/analyze_labels.py --dataset-root ./my_dataset
        """
    )
    
    # Dataset directories
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--dataset-root',
        type=str,
        help='Root directory containing train/val/test subdirectories'
    )
    
    dir_group = parser.add_argument_group('individual directories')
    dir_group.add_argument(
        '--train-dir',
        type=str,
        help='Directory containing training label files (.txt)'
    )
    dir_group.add_argument(
        '--val-dir',
        type=str,
        help='Directory containing validation label files (.txt)'
    )
    dir_group.add_argument(
        '--test-dir',
        type=str,
        help='Directory containing test label files (.txt) [optional]'
    )
    
    # Configuration options
    parser.add_argument(
        '--class-map',
        type=str,
        help='JSON file mapping class IDs to names (e.g., {"0": "tote", "1": "box"})'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Save analysis to CSV file'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only output CSV, suppress console output'
    )
    
    args = parser.parse_args()
    
    # Handle dataset root vs individual directories
    if args.dataset_root:
        root = Path(args.dataset_root)
        if not root.exists():
            print(f"Error: Dataset root directory does not exist: {args.dataset_root}")
            sys.exit(1)
        
        train_dir = str(root / "train")
        val_dir = str(root / "val") 
        test_dir = str(root / "test") if (root / "test").exists() else None
    else:
        if not args.train_dir or not args.val_dir:
            print("Error: When not using --dataset-root, both --train-dir and --val-dir are required")
            sys.exit(1)
        
        train_dir = args.train_dir
        val_dir = args.val_dir
        test_dir = args.test_dir
    
    # Validate directories
    train_dir = validate_directory(train_dir, "Training")
    val_dir = validate_directory(val_dir, "Validation")
    if test_dir:
        test_dir = validate_directory(test_dir, "Test")
    
    # Load class names if provided
    class_names = None
    if args.class_map:
        class_names = load_class_names(args.class_map)
    
    # Initialize analyzer
    analyzer = LabelAnalyzer(class_names=class_names)
    
    try:
        # Perform analysis
        if not args.quiet:
            analyzer.print_analysis(train_dir, val_dir, test_dir)
        
        # Save to CSV if requested
        if args.output:
            analyzer.save_analysis(train_dir, val_dir, args.output, test_dir)
            if args.quiet:
                print(f"Analysis saved to {args.output}")
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()