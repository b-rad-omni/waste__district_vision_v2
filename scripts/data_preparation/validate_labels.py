#!/usr/bin/env python3
"""
Label Validation CLI Tool

Command-line interface for finding, validating, and managing YOLO label files.
Provides functionality for searching specific classes, finding problematic labels, 
and performing comprehensive dataset validation.

Examples:
    # Comprehensive dataset validation
    python scripts/data_preparation/validate_labels.py --directory ./dataset/train --validate
    
    # Find all files containing a specific class
    python scripts/data_preparation/validate_labels.py --directory ./dataset --find-class 5
    
    # Find empty label files
    python scripts/data_preparation/validate_labels.py --directory ./dataset --find-class blank
    
    # Find weak confidence predictions
    python scripts/data_preparation/validate_labels.py --directory ./predictions --find-weak 0.3
    
    # Find orphaned files
    python scripts/data_preparation/validate_labels.py --directory ./dataset --find-orphans
"""

import argparse
import json
import sys
from pathlib import Path

# Import from our modular source
try:
    from src.data.label_validator import LabelValidator
except ImportError:
    print("Error: Could not import LabelValidator. Make sure you've installed the package:")
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


def print_class_search_results(matches, target_class, show_details=False):
    """Print results from class search."""
    if not matches:
        print(f"No files found containing class '{target_class}'")
        return
    
    print(f"\n🔍 Found {len(matches)} files containing class '{target_class}':")
    print("-" * 60)
    
    for label_file, matching_lines in matches:
        print(f"📄 {label_file.name}")
        if show_details:
            for line in matching_lines[:3]:  # Show first 3 matches
                print(f"   {line}")
            if len(matching_lines) > 3:
                print(f"   ... and {len(matching_lines) - 3} more")
        print()


def print_problem_results(problems, show_details=False):
    """Print results from problem detection."""
    total_problems = sum(len(problem_list) for problem_list in problems.values())
    
    if total_problems == 0:
        print("✅ No problems found!")
        return
    
    print(f"\n🚨 Found {total_problems} problems:")
    print("-" * 60)
    
    for problem_type, problem_list in problems.items():
        if not problem_list:
            continue
        
        print(f"\n{problem_type.replace('_', ' ').title()}: {len(problem_list)}")
        
        if show_details:
            for file_path, description in problem_list[:5]:  # Show first 5
                print(f"  📄 {file_path.name}: {description}")
            if len(problem_list) > 5:
                print(f"  ... and {len(problem_list) - 5} more")


def print_orphan_results(orphaned_labels, orphaned_images, show_details=False):
    """Print results from orphan detection."""
    if not orphaned_labels and not orphaned_images:
        print("✅ No orphaned files found!")
        return
    
    if orphaned_labels:
        print(f"\n🏷️  Orphaned Labels ({len(orphaned_labels)}):")
        if show_details:
            for label_file in orphaned_labels[:10]:
                print(f"  📄 {label_file.name}")
            if len(orphaned_labels) > 10:
                print(f"  ... and {len(orphaned_labels) - 10} more")
    
    if orphaned_images:
        print(f"\n🖼️  Orphaned Images ({len(orphaned_images)}):")
        if show_details:
            for image_file in orphaned_images[:10]:
                print(f"  📄 {image_file.name}")
            if len(orphaned_images) > 10:
                print(f"  ... and {len(orphaned_images) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Validate and analyze YOLO label files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full dataset validation
  python scripts/data_preparation/validate_labels.py --directory ./train --validate
  
  # Find all bucket samples (class 5)
  python scripts/data_preparation/validate_labels.py --directory ./dataset --find-class 5 --details
  
  # Find empty label files
  python scripts/data_preparation/validate_labels.py --directory ./dataset --find-class blank
  
  # Find weak predictions below 30% confidence
  python scripts/data_preparation/validate_labels.py --directory ./predictions --find-weak 0.3
  
  # Check for orphaned files only
  python scripts/data_preparation/validate_labels.py --directory ./dataset --find-orphans --details
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--directory',
        type=str,
        required=True,
        help='Directory containing label files to validate'
    )
    
    # Operation modes (mutually exclusive)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument(
        '--validate',
        action='store_true',
        help='Perform comprehensive dataset validation'
    )
    operation.add_argument(
        '--find-class',
        type=str,
        help='Find files containing specific class ID (or "blank"/"empty" for empty files)'
    )
    operation.add_argument(
        '--find-weak',
        type=float,
        help='Find files with predictions below confidence threshold'
    )
    operation.add_argument(
        '--find-orphans',
        action='store_true',
        help='Find orphaned label/image files'
    )
    operation.add_argument(
        '--statistics',
        action='store_true',
        help='Show dataset statistics only'
    )
    
    # Configuration options
    parser.add_argument(
        '--class-map',
        type=str,
        help='JSON file mapping class IDs to names'
    )
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['jpg', 'jpeg', 'png', 'bmp'],
        help='Supported image file extensions (default: jpg jpeg png bmp)'
    )
    
    # Output options
    parser.add_argument(
        '--details',
        action='store_true',
        help='Show detailed information in results'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate directory
    directory_path = Path(args.directory)
    if not directory_path.exists():
        print(f"Error: Directory does not exist: {args.directory}")
        sys.exit(1)
    
    # Check for label files
    txt_files = list(directory_path.glob("*.txt"))
    if not txt_files and not args.quiet:
        print(f"Warning: No .txt files found in directory: {args.directory}")
    
    # Load class names if provided
    class_names = None
    if args.class_map:
        class_names = load_class_names(args.class_map)
        if not args.quiet:
            print(f"Loaded custom class mapping with {len(class_names)} classes")
    
    # Initialize validator
    validator = LabelValidator(supported_extensions=args.extensions)
    if class_names:
        validator.set_class_names(class_names)
    
    try:
        results = {}
        
        if args.validate:
            # Comprehensive validation
            if not args.quiet:
                print(f"🔍 Validating dataset in: {args.directory}")
            
            results = validator.validate_dataset(args.directory, verbose=not args.quiet)
        
        elif args.find_class:
            # Find specific class
            target_class = args.find_class.lower()
            if not args.quiet:
                print(f"🔍 Searching for class '{target_class}' in: {args.directory}")
            
            matches = validator.find_files_with_class(args.directory, target_class)
            print_class_search_results(matches, target_class, args.details)
            results = {'matches': [(str(f), lines) for f, lines in matches]}
        
        elif args.find_weak is not None:
            # Find weak confidence predictions
            if not args.quiet:
                print(f"🔍 Finding weak predictions (< {args.find_weak}) in: {args.directory}")
            
            problems = validator.find_problematic_labels(
                args.directory, 
                confidence_threshold=args.find_weak,
                check_format=False,
                check_bounds=False
            )
            
            weak_problems = problems['weak_confidence']
            if weak_problems:
                print(f"\n🚨 Found {len(weak_problems)} weak predictions:")
                if args.details:
                    for file_path, description in weak_problems:
                        print(f"  📄 {file_path.name}: {description}")
            else:
                print("✅ No weak predictions found!")
            
            results = {'weak_predictions': [(str(f), desc) for f, desc in weak_problems]}
        
        elif args.find_orphans:
            # Find orphaned files
            if not args.quiet:
                print(f"🔍 Finding orphaned files in: {args.directory}")
            
            orphaned_labels, orphaned_images = validator.find_orphaned_files(args.directory)
            print_orphan_results(orphaned_labels, orphaned_images, args.details)
            
            results = {
                'orphaned_labels': [str(f) for f in orphaned_labels],
                'orphaned_images': [str(f) for f in orphaned_images]
            }
        
        elif args.statistics:
            # Show statistics only
            if not args.quiet:
                print(f"📊 Analyzing dataset statistics: {args.directory}")
            
            stats = validator.get_dataset_statistics(args.directory)
            
            print("\n📊 Dataset Statistics:")
            print("-" * 50)
            print(f"Label Files: {stats['total_label_files']}")
            print(f"Image Files: {stats['total_image_files']}")
            print(f"Empty Labels: {stats['empty_label_files']}")
            print(f"Total Annotations: {stats['total_annotations']}")
            print(f"Avg Annotations/File: {stats['avg_annotations_per_file']:.1f}")
            
            if stats['class_distribution']:
                print(f"\nClass Distribution:")
                for class_id, count in sorted(stats['class_distribution'].items()):
                    class_name = validator.class_names.get(class_id, class_id)
                    print(f"  {class_name} ({class_id}): {count}")
            
            results = {'statistics': stats}
        
        # Save results to JSON if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            if not args.quiet:
                print(f"\n💾 Results saved to: {args.output}")
    
    except Exception as e:
        print(f"Error during validation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()