import cv2
import time
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO
from interference.bucket_infraction_detector import BucketInfractionDetector

def run_detection_with_infractions():
    # Your existing paths
    #model_path = 'c:/Users/Brad/Documents/GitHub/waste__district_vision_v2/models/trained/yolov8_autopipe_v5-23_merged_small_optimizedv2/weights/best.pt'
    model_path = 'c:/Users/Brad/Documents/GitHub/waste__district_vision_v2/models/trained/yolov8_Waste_District_Vision_v2_1st/weights/best.pt'
    #source_path = 'c:/Users/Brad/Documents/GitHub/waste__district_vision_v2/datasets/test_sets/timeline_1.mp4'
    source_path = 'c:/Users/Brad/Documents/GitHub/waste__district_vision_v2/datasets/test_sets/Demo.mkv'
    tracker_config = 'c:/Users/Brad/Documents/GitHub/waste__district_vision_v2/configs/tracking/buck_byte30.yaml'
    
    # Load model
    model = YOLO(model_path)
    
    # Initialize infraction detector
    bucket_detector = BucketInfractionDetector()
    
    # Configure tracking
    model.track(
        source=source_path,
        conf=0.38,
        iou=0.10,
        tracker=tracker_config,
        save=True,
        stream=True,  # Enable streaming for frame-by-frame processing
        verbose=False
    )
    
    # Process each frame
    for frame_idx, result in enumerate(model.track(
        source=source_path,
        conf=0.38,
        iou=0.20,
        tracker=tracker_config,
        stream=True
    )):
        
        # Extract tracking data
        if result.boxes is not None and result.boxes.id is not None:
            # Convert YOLO results to detection objects
            detections = []
            
            for i in range(len(result.boxes)):
                # Get detection data
                track_id = int(result.boxes.id[i])
                bbox = result.boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                confidence = float(result.boxes.conf[i])
                class_id = int(result.boxes.cls[i])
                class_name = model.names[class_id]
                
                # Create detection object
                detection = {
                    'track_id': track_id,
                    'bbox': bbox,
                    'confidence': confidence,
                    'class_name': class_name
                }
                detections.append(detection)
            
            # Check for infractions
            current_time = time.time()
            infractions = bucket_detector.process_detections(detections, current_time)
            
            # Handle any infractions
            for infraction in infractions:
                print(f"🚨 INFRACTION DETECTED!")
                print(f"   Track ID: {infraction.track_id}")
                print(f"   Object: {infraction.object_class}")
                print(f"   Zone: {infraction.zone}")
                print(f"   Entry Count: {infraction.entry_count}")
                print(f"   Time: {infraction.timestamp}")
                
                # Save infraction clip (optional)
                save_infraction_clip(result, infraction, frame_idx)
        
        # Optional: Display frame with zones
        display_frame_with_zones(result, bucket_detector)

def save_infraction_clip(result, infraction, frame_idx):
    """Save a clip around the infraction"""
    clip_filename = f"infraction_{infraction.track_id}_{int(time.time())}.mp4"
    clip_path = f"clips/{clip_filename}"  # Save to clips folder
    
    # Create clips directory if it doesn't exist
    os.makedirs("clips", exist_ok=True)
    
    # Save current frame as image (simple approach)
    cv2.imwrite(clip_path.replace('.mp4', '.jpg'), result.orig_img)
    print(f"   💾 Saved image: {clip_path.replace('.mp4', '.jpg')}")

def display_frame_with_zones(result, bucket_detector):
    """Display frame with detection zones overlaid"""
    frame = result.orig_img
    
    # Calculate scale factor FIRST
    height, width = frame.shape[:2]
    scale_factor = 0.5  # Or whatever you're using
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Draw zones on ORIGINAL size frame, then resize
    for zone_name, zone_coords in bucket_detector.BIN_ZONES.items():
        pts = np.array(zone_coords, np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        cv2.putText(frame, zone_name, tuple(zone_coords[0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # NOW resize the frame with zones already drawn
    resized_frame = cv2.resize(frame, (new_width, new_height))
    
    cv2.imshow('Detection with Zones', resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True

if __name__ == "__main__":
    run_detection_with_infractions()