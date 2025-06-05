#!/usr/bin/env python3
"""Export trained YOLOv8 model to different formats."""

import argparse
import logging
from pathlib import Path
from ultralytics import YOLO


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def export_model(model_path: str, formats: list, output_dir: str = None):
    """Export model to specified formats.
    
    Args:
        model_path: Path to the .pt model file
        formats: List of export formats
        output_dir: Output directory for exported models
    """
    logger = logging.getLogger(__name__)
    
    # Load model
    model = YOLO(model_path)
    
    # Set output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(model_path).parent
    
    logger.info(f"Exporting model from: {model_path}")
    logger.info(f"Output directory: {output_path}")
    
    # Export to each format
    for format_name in formats:
        try:
            logger.info(f"Exporting to {format_name}...")
            exported_path = model.export(format=format_name, project=str(output_path))
            logger.info(f"✅ Exported to: {exported_path}")
        except Exception as e:
            logger.error(f"❌ Failed to export to {format_name}: {e}")


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(description='Export YOLOv8 model')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the .pt model file'
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        default=['onnx'],
        choices=['onnx', 'torchscript', 'tensorflow', 'tflite', 'coreml', 'openvino'],
        help='Export formats'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for exported models'
    )
    
    args = parser.parse_args()
    
    setup_logging()
    export_model(args.model, args.formats, args.output_dir)


if __name__ == "__main__":
    main()
