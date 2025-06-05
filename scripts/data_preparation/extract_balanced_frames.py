"""
CLI interface for balanced frame extraction

Simplified CLI script that imports from the src package.
"""
import argparse
import sys
from pathlib import Path

# Add src to path for importing
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from data.frame_extractor import FrameExtractor


def main():
    """Main function to run the frame extraction tool"""
    parser = argparse.ArgumentParser(
        description="Extract balanced frames from YOLO prediction results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic balanced extraction
  python scripts/extraction/extract_balanced_frames.py \
      --predictions-dir "predictions/labels" \
      --source-images "raw_frames" \
      --output-dir "selected_frames" \
      --max-per-class 50

  # Priority-based extraction focusing on underrepresented classes
  python scripts/extraction/extract_balanced_frames.py \
      --predictions-dir "predictions/labels" \
      --source-images "raw_frames" \
      --output-dir "selected_frames" \
      --max-per-class 100 \
      --strategy priority

  # Random sampling
  python scripts/extraction/extract_balanced_frames.py \
      --predictions-dir "predictions/labels" \
      --source-images "raw_frames" \
      --output-dir "selected_frames" \
      --max-per-class 25 \
      --strategy random
        """
    )
    
    # Required arguments
    parser.add_argument('--predictions-dir', required=True,
                       help='Directory containing YOLO .txt prediction files')
    parser.add_argument('--source-images', required=True,
                       help='Directory containing source images (.jpg, .png, etc.)')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for selected frames')
    
    # Optional arguments
    parser.add_argument('--max-per-class', type=int, default=50,
                       help='Maximum number of frames to extract per class (default: 50)')
    parser.add_argument('--strategy', choices=['balanced', 'priority', 'random'], 
                       default='balanced',
                       help='Frame selection strategy (default: balanced)')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze and show statistics, do not extract frames')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be extracted without actually copying files')
    
    args = parser.parse_args()
    
    try:
        # Initialize extractor
        extractor = FrameExtractor(
            predictions_dir=args.predictions_dir,
            source_images_dir=args.source_images,
            output_dir=args.output_dir
        )
        
        # Analyze predictions
        class_to_images = extractor.analyze_predictions()
        
        if not class_to_images:
            print("❌ No classes found in prediction files")
            return
        
        # Create class mapping
        class_mapping = extractor.create_class_mapping()
        
        # Print analysis summary
        extractor.print_analysis_summary(class_to_images, class_mapping)
        
        if args.analyze_only:
            print("\n✅ Analysis complete (analyze-only mode)")
            return
        
        # Select frames
        selected_frames = extractor.select_balanced_frames(
            class_to_images, 
            args.max_per_class, 
            args.strategy
        )
        
        if args.dry_run:
            print(f"\n🔍 DRY RUN - Would extract:")
            total_would_extract = sum(len(frames) for frames in selected_frames.values())
            for class_id, frames in selected_frames.items():
                class_name = class_mapping.get(class_id, f"Class_{class_id}")
                print(f"  {class_name}: {len(frames)} frames")
            print(f"\nTotal: {total_would_extract} frames")
            return
        
        # Extract frames
        total_requested, total_copied = extractor.extract_frames(selected_frames, class_mapping)
        
        # Print final summary
        print(f"\n✅ Frame extraction complete!")
        print(f"📊 Summary:")
        print(f"   Total requested: {total_requested}")
        print(f"   Total copied: {total_copied}")
        print(f"   Output directory: {args.output_dir}")
        
        if total_copied < total_requested:
            print(f"⚠️  Note: {total_requested - total_copied} frames could not be copied (missing source files)")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()