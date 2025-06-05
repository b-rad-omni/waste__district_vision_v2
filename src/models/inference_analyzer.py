"""
Reusable inference analysis functionality
"""
import os
import cv2
import glob
import shutil
from pathlib import Path
from datetime import datetime
from utils.model_registry_loader import ModelRegistry

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

class InferenceAnalyzer:  
    """Reusable inference tool - works with any image directory"""
  
    def __init__(self):
        """Initialize with just the model registry"""
        # Only load model registry - no YAML config needed
        self.model_registry = ModelRegistry()
        print("📋 Model registry loaded")
    
    def get_images_from_directory(self, directory):
        """Get list of image files from any directory"""
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Look for common image formats
        image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        images = []
        
        directory = os.path.abspath(directory)
        
        for pattern in image_patterns:
            images.extend(glob.glob(os.path.join(directory, pattern)))
            # Also check subdirectories
            images.extend(glob.glob(os.path.join(directory, '**', pattern), recursive=True))
        
        return sorted(images)
    
    def analyze_directory(self, directory):
        """Analyze what's in a directory"""
        images = self.get_images_from_directory(directory)
        
        print(f"\n📁 Directory Analysis: {directory}")
        print(f"   Total images found: {len(images)}")
        
        if images:
            # Show date range if possible
            try:
                first_mod = os.path.getmtime(images[0])
                last_mod = os.path.getmtime(images[-1])
                first_date = datetime.fromtimestamp(first_mod).strftime("%Y-%m-%d %H:%M")
                last_date = datetime.fromtimestamp(last_mod).strftime("%Y-%m-%d %H:%M")
                print(f"   Date range: {first_date} → {last_date}")
            except:
                pass
            
            # Show file size info
            try:
                total_size = sum(os.path.getsize(img) for img in images)
                avg_size = total_size / len(images)
                print(f"   Total size: {total_size/(1024*1024):.1f} MB")
                print(f"   Average size: {avg_size/(1024*1024):.2f} MB per image")
            except:
                pass
                
            # Show first few files
            print(f"   Sample files:")
            for img in images[:3]:
                filename = os.path.basename(img)
                print(f"      {filename}")
            if len(images) > 3:
                print(f"      ... and {len(images)-3} more")
        
        return images
    
    def should_save_image(self, results, save_detection_class):
        """
        Check if image should be saved based on detected classes
        
        Args:
            results: YOLO results object
            save_detection_class: List of detection class names to save, or None to save all
        
        Returns:
            bool: True if image should be saved
        """
        if save_detection_class is None:
            return True  # Save all images if no filter specified
        
        if results.boxes is None or len(results.boxes) == 0:
            return False  # No detections, don't save
        
        # Get class names from the model
        class_names = results.names  # Dictionary mapping class_id -> class_name
        
        # Get detected class IDs
        detected_class_ids = results.boxes.cls.cpu().numpy() if results.boxes.cls is not None else []
        
        # Debug: Print what we found
        detected_labels = []
        for class_id in detected_class_ids:
            class_name = class_names.get(int(class_id), "unknown")
            detected_labels.append(class_name)
        
        print(f"   Detected classes: {detected_labels}")
        print(f"   Looking for: {save_detection_class}")
        
        # Convert detected class IDs to names and check against save_detection_class
        for class_id in detected_class_ids:
            class_name = class_names.get(int(class_id), "unknown")
            if class_name in save_detection_class:
                print(f"   ✅ Match found: '{class_name}' in save_detection_class")
                return True
        
        print(f"   ❌ No matching detection classes found")
        return False
    
    def save_yolo_labels(self, results, filename, labels_dir, image_shape):
        """
        Save detection results in YOLO format (.txt file)
        
        Args:
            results: YOLO results object
            filename: Original image filename
            labels_dir: Directory to save label files
            image_shape: (height, width, channels) of original image
        """
        if results.boxes is None or len(results.boxes) == 0:
            # Create empty label file for images with no detections
            label_filename = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_filename)
            with open(label_path, 'w') as f:
                pass  # Empty file
            return
        
        # Get image dimensions for normalization
        img_height, img_width = image_shape[:2]
        
        # Prepare label data
        label_lines = []
        
        # Get detection data
        boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        classes = results.boxes.cls.cpu().numpy()  # class indices
        confidences = results.boxes.conf.cpu().numpy()  # confidence scores
        
        for box, cls, conf in zip(boxes, classes, confidences):
            # Convert to YOLO format (class x_center y_center width height)
            x1, y1, x2, y2 = box
            
            # Calculate center and dimensions (normalized)
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            # Format: class_id x_center y_center width height
            label_line = f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            label_lines.append(label_line)
        
        # Save to file
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_filename)
        
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))
    
    def run_inference(self, data_dir, model_name="latest", max_images=None, 
                     confidence=0.30, save_results=True, show_display=False,
                     output_dir=None, save_detection_class=None, save_txt_labels=False,
                     copy_originals=False):
        """
        Run inference on images in any directory
        
        Args:
            data_dir: Directory containing images (required)
            model_name: Model name or alias from registry
            max_images: Limit number of images to process
            confidence: Detection confidence threshold
            save_results: Save annotated images
            show_display: Show results in window
            output_dir: Custom output directory for saved results
            save_detection_class: List of detection classes to save, or None to save all
            save_txt_labels: Save YOLO format .txt label files
            copy_originals: Copy original images without annotations
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics package required for inference")
        
        # Get model from registry
        print(f"\n🔍 Loading model: {model_name}")
        try:
            model_info = self.model_registry.get_model_info(model_name)
            model_path = self.model_registry.get_model_path(model_name)
            print(f"📊 Model: {model_info['description']}")
            print(f"📈 mAP: {model_info['metrics']['mAP']}, Precision: {model_info['metrics']['precision']}")
            print(f"🏷️  Status: {model_info['status']}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return
        
        # Load YOLO model
        try:
            model = YOLO(model_path)
            print(f"✅ Model loaded from: {model_path}")
        except Exception as e:
            print(f"❌ Error initializing YOLO model: {e}")
            return
        
        # Get images to process
        try:
            images = self.get_images_from_directory(data_dir)
            if not images:
                print(f"❌ No images found in: {data_dir}")
                return
            
            if max_images:
                images = images[:max_images]
                print(f"📸 Processing first {len(images)} images (limited by --max)")
            else:
                print(f"📸 Processing all {len(images)} images")
                
        except Exception as e:
            print(f"❌ Error finding images: {e}")
            return
        
        # Create output directory if saving results
        if save_results:
            if output_dir:
                results_dir = output_dir
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dir_name = os.path.basename(os.path.abspath(data_dir))
                results_dir = f"inference_results_{dir_name}_{model_name}_{timestamp}"
            
            os.makedirs(results_dir, exist_ok=True)
            print(f"💾 Saving results to: {results_dir}")
            
            # Create labels subdirectory if saving txt labels
            if save_txt_labels:
                labels_dir = os.path.join(results_dir, "labels")
                os.makedirs(labels_dir, exist_ok=True)
                print(f"📝 Saving .txt labels to: {labels_dir}")
            
            # Create images subdirectory if copying originals
            if copy_originals:
                images_dir = os.path.join(results_dir, "images")
                os.makedirs(images_dir, exist_ok=True)
                print(f"📷 Copying original images to: {images_dir}")
            
            if save_detection_class:
                print(f"🏷️  Only saving images containing: {', '.join(save_detection_class)}")
        
        # Fast batch processing (like the CLI)
        print(f"\n🚀 Starting batch inference with confidence threshold: {confidence}")
        
        if show_display or not save_results:
            # If you need display or custom processing, use the old slow method
            total_detections = 0
            processed_count = 0
            saved_count = 0
            
            if show_display:
                print("Press 'q' to quit, 'n' for next image, 's' to save current image")
            
            for i, image_path in enumerate(images):
                try:
                    should_save = False
                    detections = 0
                    filename = os.path.basename(image_path)
                    
                    # Load image
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"⚠️  Skipping {os.path.basename(image_path)} - could not load")
                        continue
                    
                    # Run inference
                    results = model(image, conf=confidence, verbose=False)
                    
                    # Count detections
                    detections = len(results[0].boxes) if results[0].boxes is not None else 0
                    total_detections += detections
                    processed_count += 1
                    
                    print(f"[{i+1}/{len(images)}] {filename}: {detections} detections")
                    
                    # Annotate image
                    annotated_image = results[0].plot()
                    
                    # Add info overlay
                    info_text = f"Model: {model_name} | Detections: {detections} | Conf: {confidence}"
                    cv2.putText(annotated_image, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_image, filename, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Check if we should save this image
                    should_save = self.should_save_image(results[0], save_detection_class)
                    
                    if save_results and should_save:
                        # Save annotated image
                        output_path = os.path.join(results_dir, f"annotated_{filename}")
                        cv2.imwrite(output_path, annotated_image)
                        
                        # Copy original image if requested
                        if copy_originals:
                            original_path = os.path.join(images_dir, filename)
                            shutil.copy2(image_path, original_path)
                        
                        # Save .txt labels if requested
                        if save_txt_labels and results[0].boxes is not None:
                            self.save_yolo_labels(results[0], filename, labels_dir, image.shape)
                        
                        saved_count += 1
                    
                    # Display if requested
                    if show_display:
                        # Resize for display if image is too large
                        h, w = annotated_image.shape[:2]
                        if w > 1200 or h > 800:
                            scale = min(1200/w, 800/h)
                            new_w, new_h = int(w*scale), int(h*scale)
                            annotated_image = cv2.resize(annotated_image, (new_w, new_h))
                        
                        cv2.imshow(f"Inference Results - {model_name}", annotated_image)
                        
                        # Wait for key press
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord('q'):
                            print("Stopping inference...")
                            break
                        elif key == ord('s') and not save_results:
                            # Save single image
                            save_path = f"detection_{filename}"
                            cv2.imwrite(save_path, annotated_image)
                            print(f"💾 Saved: {save_path}")
                    
                except Exception as e:
                    print(f"❌ Error processing {image_path}: {e}")
                    continue
        
        else:
            # Fast batch processing with label filtering
            print("🚀 Using fast batch processing...")
            
            total_detections = 0
            processed_count = 0
            saved_count = 0
            
            for i, image_path in enumerate(images):
                try:
                    # Initialize variables at the start
                    should_save = False
                    detections = 0
                    filename = os.path.basename(image_path)
                    
                    # Load and process image
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"[{i+1}/{len(images)}] {filename}: ❌ Could not load image")
                        continue
                    
                    # Run inference
                    results = model(image, conf=confidence, verbose=False)
                    
                    # Count detections
                    detections = len(results[0].boxes) if results[0].boxes is not None else 0
                    total_detections += detections
                    processed_count += 1
                    
                    print(f"[{i+1}/{len(images)}] {filename}: {detections} detections", end="")
                    
                    # Debug: Show what we're checking
                    print(f"\n   DEBUG: save_results={save_results}, save_detection_class={save_detection_class}")
                    
                    # Check if we should save this image
                    should_save = self.should_save_image(results[0], save_detection_class)
                    print(f"   DEBUG: should_save_image returned: {should_save}")
                    
                    if save_results and should_save:
                        # Save annotated image
                        annotated_image = results[0].plot()
                        annotated_path = os.path.join(results_dir, f"annotated_{filename}")
                        cv2.imwrite(annotated_path, annotated_image)
                        
                        # Copy original image if requested
                        if copy_originals:
                            original_path = os.path.join(images_dir, filename)
                            shutil.copy2(image_path, original_path)
                        
                        # Save .txt labels if requested
                        if save_txt_labels and results[0].boxes is not None:
                            self.save_yolo_labels(results[0], filename, labels_dir, image.shape)
                        
                        saved_count += 1
                        print("   FINAL: SAVED")
                    else:
                        # Don't save - either save_results=False or no matching detection classes
                        if save_detection_class is not None and not should_save:
                            print("   FINAL: skipped (no matching detection classes)")
                        else:
                            print("   FINAL: not saved")
                
                except Exception as e:
                    print(f"❌ Error processing {image_path}: {e}")
                    continue
            
            print(f"✅ Batch processing complete!")
            if save_results:
                print(f"📁 Saved {saved_count}/{processed_count} images to: {results_dir}")
                if save_txt_labels:
                    print(f"📝 Saved corresponding .txt labels to: {labels_dir}")
        
        # Cleanup
        if show_display:
            cv2.destroyAllWindows()
        
        # Print summary
        print(f"\n📊 Inference Summary:")
        print(f"   Model: {model_name} ({model_info['description']})")
        print(f"   Data directory: {data_dir}")
        print(f"   Images processed: {processed_count}")
        print(f"   Total detections: {total_detections}")
        print(f"   Average detections per image: {total_detections/max(processed_count,1):.1f}")
        if save_results:
            if save_detection_class:
                print(f"   Images saved (with detection classes {save_detection_class}): {saved_count}")
            else:
                print(f"   Images saved: {saved_count}")
            print(f"   Results saved to: {results_dir}")
    
    def compare_models(self, data_dir, models=None, max_images=5, confidence=0.5):
        """
        Quick comparison of multiple models on the same images
        
        Args:
            data_dir: Directory containing test images
            models: List of model names to compare (default: ["latest", "stable", "fast"])
            max_images: Number of images to test
            confidence: Detection confidence threshold
        """
        if models is None:
            models = ["latest", "stable", "fast"]
        
        print(f"\n🔄 Model Comparison on {max_images} images")
        print(f"📁 Data directory: {data_dir}")
        print(f"🎯 Confidence threshold: {confidence}")
        
        # Get test images
        images = self.get_images_from_directory(data_dir)[:max_images]
        if not images:
            print(f"❌ No images found in: {data_dir}")
            return
        
        results = {}
        
        for model_name in models:
            try:
                print(f"\n🔍 Testing model: {model_name}")
                model_info = self.model_registry.get_model_info(model_name)
                model_path = self.model_registry.get_model_path(model_name)
                model = YOLO(model_path)
                
                total_detections = 0
                for image_path in images:
                    image = cv2.imread(image_path)
                    if image is not None:
                        inference_results = model(image, conf=confidence, verbose=False)
                        detections = len(inference_results[0].boxes) if inference_results[0].boxes is not None else 0
                        total_detections += detections
                
                avg_detections = total_detections / len(images)
                results[model_name] = {
                    'total_detections': total_detections,
                    'avg_detections': avg_detections,
                    'mAP': model_info['metrics']['mAP'],
                    'precision': model_info['metrics']['precision']
                }
                
                print(f"   ✅ {total_detections} total detections, {avg_detections:.1f} avg per image")
                
            except Exception as e:
                print(f"   ❌ Error with model {model_name}: {e}")
                results[model_name] = None
        
        # Print comparison summary
        print(f"\n📊 Comparison Summary:")
        print(f"{'Model':<12} {'Avg Det':<8} {'Total':<8} {'mAP':<6} {'Precision':<10}")
        print("-" * 50)
        
        for model_name, result in results.items():
            if result:
                print(f"{model_name:<12} {result['avg_detections']:<8.1f} {result['total_detections']:<8} "
                      f"{result['mAP']:<6.2f} {result['precision']:<10.2f}")
            else:
                print(f"{model_name:<12} {'ERROR':<8} {'ERROR':<8} {'ERROR':<6} {'ERROR':<10}")