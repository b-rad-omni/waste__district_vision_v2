from ultralytics import YOLO
import multiprocessing

def main():
    # Initialize model
    
    model = " path/to/your/yolov8n.pt"  # Update with your model path
    model = YOLO(model)  # Load the YOLO model
    
    # Train with 1080p resolution
    model.train(
        
    data='path/to/your/dataset.yaml',  # Update with your dataset YAML file
    
        epochs=60,               # DEFAULT: 100 | Total training epochs
                                # RECOMMENDATION: Start with 50, use early stopping
        
        imgsz=(1920, 1080),              # DEFAULT: 640 | Input image size for training
                                # Your choice: 1920 matches your security cam resolution
        
        batch=2,                # DEFAULT: 16 | Batch size (adjust based on GPU memory)
                                # Higher = faster training but needs more VRAM
                                # Lower = slower but uses less memory
        
        # ==================== LEARNING RATE & OPTIMIZATION ==================
        lr0=0.01,                # DEFAULT: 0.01 | Initial learning rate
                                # Controls how big steps the model takes when learning
                                # Higher = learns faster but might overshoot optimal weights
                                # Lower = learns slower but more stable
        
        lrf=0.01,                # DEFAULT: 0.01 | Final learning rate (lr0 * lrf)
                                # Learning rate at end of training (after decay)
                                # Smaller = fine-tunes more carefully at the end
        
        momentum=0.937,          # DEFAULT: 0.937 | SGD momentum factor
                                # Helps optimization not get stuck in local minima
                                # Higher = smoother optimization path
        
        weight_decay=0.0005,     # DEFAULT: 0.0005 | L2 regularization penalty
                                # Prevents weights from getting too large (reduces overfitting)
                                # Higher = more regularization, simpler model
                                # RECOMMENDATION: Keep default, proven to work well
        
        warmup_epochs=3,         # DEFAULT: 3 | Number of warmup epochs
                                # Gradually increases learning rate from 0 to lr0
                                # Helps training stability at the beginning
        
        warmup_momentum=0.8,     # DEFAULT: 0.8 | Momentum during warmup phase
        warmup_bias_lr=0.1,      # DEFAULT: 0.1 | Bias learning rate during warmup
        
        # ==================== DATA AUGMENTATION - GEOMETRIC ==================
        degrees=10.0,            # DEFAULT: 0.0 | Random rotation range (-degrees to +degrees)
                                # RECOMMENDATION: 10.0 for security cameras
                                # Helps model handle slightly tilted camera angles
                                # Too high can create unrealistic orientations
        
        translate=0.2,           # DEFAULT: 0.1 | Random translation (fraction of image size)
                                # Shifts objects around in the frame
                                # Helps model detect objects anywhere in the image
                                # RECOMMENDATION: Keep default 0.1
        
        scale=0.8,               # DEFAULT: 0.5 | Random scale range (1±scale)
                                # Makes objects appear larger/smaller
                                # Critical for multi-camera with different distances
                                # 0.5 means objects can be 50% to 150% of original size
        
        shear=0.05,               # DEFAULT: 0.0 | Shear transformation
                                # RECOMMENDATION: 0.0 for security cameras
                                # Shear creates perspective distortion that might not help
        
        perspective=0.001,         # DEFAULT: 0.0 | Random perspective transformation
                                # RECOMMENDATION: 0.0 unless cameras have perspective issues
        
        flipud=0.05,              # DEFAULT: 0.0 | Probability of vertical flip
                                # RECOMMENDATION: 0.0 for security cameras
                                # Vertical flips don't make sense for overhead/wall-mounted cameras
        
        fliplr=0.5,              # DEFAULT: 0.5 | Probability of horizontal flip
                                # RECOMMENDATION: Keep 0.5
                                # Objects can appear from left or right side equally
        
        # ==================== DATA AUGMENTATION - COLOR/LIGHTING =============
        hsv_h=0.05,             # DEFAULT: 0.015 | Hue augmentation range
                                # Changes color tint slightly
                                # RECOMMENDATION: Keep default - good for lighting variations
        
        hsv_s=0.4,               # DEFAULT: 0.7 | Saturation augmentation range
                                # Changes color intensity/vividness
                                # RECOMMENDATION: Keep default - handles different camera settings
        
        hsv_v=0.6,               # DEFAULT: 0.4 | Value (brightness) augmentation range
                                # CRITICAL for security cameras with day/night variations
                                # RECOMMENDATION: Keep default or slightly higher (0.5)
        
        # ==================== ADVANCED AUGMENTATION TECHNIQUES ===============
        mosaic=1.0,              # DEFAULT: 1.0 | Probability of mosaic augmentation
                                # Combines 4 different images into one training sample
                                # RECOMMENDATION: 0.8 for your multi-camera setup
                                # Your images already have 4 camera views, so full mosaic
                                # might be too complex. Reduce to 0.8 or 0.5
        
        mixup=0.2,               # DEFAULT: 0.0 | Probability of mixup augmentation
                                # Blends two images with transparency
                                # RECOMMENDATION: Keep 0.0 for object detection
                                # Creates unrealistic transparent effects
        
        copy_paste=0.2,          # DEFAULT: 0.1 | Probability of copy-paste augmentation
                                # Copies objects from one image and pastes into another
                                # RECOMMENDATION: 0.3 for multi-camera learning
                                # Helps model learn same objects across different camera contexts
        
        erasing=0.4,             # DEFAULT: 0.4 | Random erasing probability
                                # Randomly erases rectangular patches
                                # Helps model not rely on specific image regions
                                # RECOMMENDATION: Keep default
        
        crop_fraction=1.0,       # DEFAULT: 1.0 | Fraction of image to crop for training
                                # 1.0 means use full image, lower values crop randomly
                                # RECOMMENDATION: Keep 1.0 for security cameras
        
        # ==================== TRAINING BEHAVIOR & SAFETY NETS ===============
        patience=25,            # DEFAULT: 100 | Early stopping patience (epochs)
                                # RECOMMENDATION: 5-10 for your overfitting issues
                                # Stops training if validation doesn't improve
                                # Lower = stops sooner, prevents overfitting
        
        save=True,               # DEFAULT: True | Save train checkpoints
        save_period=8,          # DEFAULT: -1 | Save checkpoint every x epochs (-1 = only last)
                                # RECOMMENDATION: 5 | Save every 5 epochs to pick best one
        
        val=True,                # DEFAULT: True | Validate/test during training
        plots=True,              # DEFAULT: True | Save training plots
        
        # ==================== OPTIMIZATION ALGORITHM ======================
        optimizer='auto',        # DEFAULT: 'auto' | Optimizer choice
                                # Options: 'SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp'
                                # RECOMMENDATION: 'AdamW' for complex multi-camera scenarios
                                # AdamW often works better than SGD for challenging datasets
        
        close_mosaic=10,         # DEFAULT: 10 | Disable mosaic in final X epochs
                                # Allows model to fine-tune on normal images at the end
                                # RECOMMENDATION: Keep default 10
        
        # ==================== LOSS FUNCTION WEIGHTS =======================
        box=7.5,                 # DEFAULT: 7.5 | Box loss weight
                                # How much to penalize incorrect bounding box predictions
                                # RECOMMENDATION: Keep default unless specific issues
        
        cls=0.5,                 # DEFAULT: 0.5 | Classification loss weight  
                                # How much to penalize incorrect class predictions
                                # RECOMMENDATION: Keep default
        
        dfl=1.5,                 # DEFAULT: 1.5 | Distribution focal loss weight
                                # Advanced loss component for better localization
                                # RECOMMENDATION: Keep default
        
        # ==================== ADVANCED SETTINGS ===========================
        rect=False,              # DEFAULT: False | Rectangular training
                                # True = preserves aspect ratios, False = square images
                                # RECOMMENDATION: False for maximum augmentation variety
        
        cos_lr=True,            # DEFAULT: False | Cosine learning rate scheduler
                                # Alternative to linear LR decay
                                # RECOMMENDATION: Try True if training is unstable
        
        overlap_mask=True,       # DEFAULT: True | Mask overlapping segments
        mask_ratio=2,            # DEFAULT: 4 | Mask downsample ratio
        dropout=0.1,             # DEFAULT: 0.0 | Dropout probability
                                # RECOMMENDATION: 0.1-0.2 if overfitting persists
                                
        
         name="yolov8_Waste_District_Vision_v2_1st",  # Name for this training run
        
        # ==================== FINAL RECOMMENDATIONS FOR YOUR CASE ==========
        # Based on your multi-camera security setup and overfitting issues:
        
        # MODIFIED RECOMMENDATIONS:
        # epochs=40,             # Shorter than default, rely on early stopping
        # patience=5,            # Much lower for early stopping
        # mosaic=0.8,            # Reduced due to existing 4-camera complexity  
        # copy_paste=0.3,        # Higher to help cross-camera learning
        # hsv_v=0.5,             # Slightly higher for day/night variations
        # optimizer='AdamW',     # Often better for complex scenarios
        # save_period=5,         # Save more frequently to pick best epoch
    )

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

# ==================== USAGE NOTES ==========================================
"""
OVERFITTING PREVENTION STRATEGY:
1. Start with these defaults
2. Monitor training vs validation loss closely  
3. If overfitting occurs:
   - Reduce mosaic to 0.5
   - Increase weight_decay to 0.001
   - Reduce lr0 to 0.005
   - Set patience to 5
   - Add dropout=0.1

MULTI-CAMERA OPTIMIZATION:
1. Increase copy_paste to 0.3 (cross-camera learning)
2. Increase hsv_v to 0.5 (lighting variations)
3. Keep degrees at 10 (slight camera angle differences)
4. Consider optimizer='AdamW' (better for complex data)

PERFORMANCE TUNING:
- If training is too slow: reduce imgsz to 1280 or 960
- If GPU memory issues: reduce batch size
- If validation mAP plateaus: try cos_lr=True
- If still overfitting: add dropout=0.1-0.2
"""