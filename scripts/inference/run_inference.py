"""
CLI interface for model inference analysis
"""
import argparse
import sys
import datetime
from pathlib import Path

from models.inference_analyzer import InferenceAnalyzer





def main():
    parser = argparse.ArgumentParser(
        description="Standalone model inference tool - works with any image directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference on any directory
  python run_inference.py /path/to/test_images --model latest
  
  # Test different models
  python run_inference.py ./test_batch_1 --model stable --conf 0.3
  python run_inference.py ~/Downloads/validation_set --model fast --max 10
  
  # Save results
  python run_inference.py /data/test_images --model latest --save
  python run_inference.py ./test_batch --model stable --save --output results_v1
  
  # Batch processing (no display)
  python run_inference.py /large/dataset --model latest --no-display --save
  
  # Compare models quickly
  python run_inference.py ./test_images --compare --models latest stable fast
  
  # Analyze directory contents
  python run_inference.py ./unknown_directory --analyze
  
  # Save only images with buckets, including .txt labels
    python scripts/inference/run_inference.py \
    --data_dir datasets/raw_frames/2025-05-16 \
    --model buckets \
    --conf 0.05 \
    --save-detection-class bucket \
    --save-txt-labels \
    --max 10

    # This will create:
    # results/predictions/model_buckets_2025-05-30_14-23/
    # ├── annotated_image1.jpg
    # ├── annotated_image2.jpg
    # └── labels/
    #     ├── image1.txt
    #     └── image2.txt
            """
    )
    
    # Required/optional inputs
    parser.add_argument('--data_dir', default='datasets/test_sets/batch1_Totes_large_foreignobj_bags/',
                        help='Directory containing images to process')

    # Model selection
    parser.add_argument('--model', '-m', default='latest',
                        help='Model name or alias from registry (default: latest)')
    parser.add_argument('--list-models', action='store_true',
                        help='List available models and exit')

    # Processing options
    parser.add_argument('--max', type=int,
                        help='Maximum number of images to process')
    parser.add_argument('--conf', type=float, default=0.20,
                        help='Detection confidence threshold (default: 0.20)')
    parser.add_argument('--save-detection-class', nargs='+', default=None,
                    help='Only save images containing these detection classes (e.g. --save-detection-class bucket person)')
    parser.add_argument('--no-txt-labels', action='store_false', dest='save_txt_labels',
                    help='turn off saving YOLO format .txt label files to labels/ subfolder in output directory')
    parser.add_argument('--copy-originals', action='store_true',
                    help='Copy original images (without annotations) to images/ subfolder for CVAT review')

    # Output control
    parser.add_argument('--no-save', action='store_false', dest='save',
                        help='Do not save annotated results to disk (default: saves results)')
    parser.add_argument('--output', '-o', default=None,
                        help='Custom output directory (default: results/predictions/{model}_{timestamp})')
    parser.add_argument('--no-display', action='store_false',
                        help='Do not show display window (batch mode)')

    # Analysis
    parser.add_argument('--analyze', action='store_true',
                        help='Just analyze directory contents and exit')
    parser.add_argument('--compare', action='store_true',
                        help='Compare multiple models on same images')
    parser.add_argument('--models', nargs='+',
                        help='Models to use for comparison (default: latest stable fast)')

    args = parser.parse_args()

    # ✅ Dynamic output path if none provided
    if args.output is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        safe_model_name = Path(args.model).stem
        args.output = Path("results/predictions") / f"model_{safe_model_name}_{timestamp}"
    else:
        args.output = Path(args.output)

    # ✅ Only make the directory if we're saving
    if args.save:
        args.output = args.output.resolve()
        args.output.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize analyzer
        analyzer = InferenceAnalyzer()
        
        # Handle list models
        if args.list_models:
            analyzer.model_registry.print_model_info()
            return
        
        # Handle analyze
        if args.analyze:
            analyzer.analyze_directory(args.data_dir)
            return
        
        # Handle compare
        if args.compare:
            analyzer.compare_models(
                data_dir=args.data_dir,
                models=args.models,
                max_images=args.max or 5,
                confidence=args.conf
            )
            return
        
        # Run standard inference
        analyzer.run_inference(
            data_dir=args.data_dir,
            model_name=args.model,
            max_images=args.max,
            confidence=args.conf,
            save_results=args.save,
            show_display=not args.no_display,
            output_dir=args.output,
            save_detection_class=args.save_detection_class,  
            save_txt_labels=args.save_txt_labels,
            copy_originals=args.copy_originals
        )
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()