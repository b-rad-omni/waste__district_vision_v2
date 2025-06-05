import os
import cv2
import numpy as np
import time
import threading
import queue
import schedule
import shutil
import json
import sys
from datetime import datetime
from collections import deque
from pathlib import Path


from data.collectors.motion_detector import MotionDetector
from data.collectors.frame_analyzer import FrameAnalyzer
from data.collectors.camera_manager import CameraManager
from utils.storage_manager import StorageManager
from utils.config_manager import load_data_collection_config


class DataCollectionSystem:
    """
    Professional data collection system for YOLO training data.
    
    This class encapsulates all the functionality from your original procedural script
    into a clean, maintainable, object-oriented design with configuration management.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the data collection system with modular configuration.
        
        This method replaces all the global variable declarations from your original script
        and sets up the system using the new configuration management system.
        
        Args:
            config_path: Optional path to custom configuration file
        """
        print("🔧 Loading data collection configuration...")
        
        # Load configuration using the new modular config manager
        self.config = load_data_collection_config(config_path)
        
        # Print configuration summary for debugging
        print(f"📷 Camera index: {self.config['camera']['index']}")
        print(f"💾 Primary storage: {self.config['storage']['primary_dir']}")
        print(f"🔄 Fallback storage: {self.config['storage']['fallback_dir']}")
        
        # === CONVERTED GLOBAL VARIABLES ===
        # These were global variables in your original script, now they're instance variables
        
        # Core system state
        self.capture_enabled = False  # Will be controlled by schedule
        self.headless = self.config['system']['headless']
        
        # Storage settings (from config)
        self.primary_dir = self.config['storage']['primary_dir']
        self.fallback_dir = self.config['storage']['fallback_dir']
        self.metadata_dir = os.path.join(self.primary_dir, "metadata")
        self.min_gb_free = 5  # Could be moved to config later
        
        # Create directories
        os.makedirs(self.primary_dir, exist_ok=True)
        os.makedirs(self.fallback_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Motion detection parameters (from config)
        self.persist_frames = self.config['motion_detection']['persist_frames']
        self.min_blob_area = self.config['motion_detection']['min_blob_area']
        self.scale_factor = self.config['motion_detection']['scale_factor']
        self.small_kernel = (3, 3)  # Could be moved to config
        
        # Burst capture settings (from config)
        self.burst_duration = self.config['collection']['burst_duration']
        self.burst_fps = self.config['collection']['burst_fps']
        self.cooldown_sec = self.config['collection']['cooldown_sec']
        
        # Diversity settings (from config)
        self.frame_history_size = self.config['frame_analysis']['frame_history_size']
        self.min_difference_threshold = self.config['frame_analysis']['min_difference_threshold']
        self.current_hour_frame_count = 0
        self.last_hour_captured = -1
        self.min_frames_per_hour = self.config['collection']['min_frames_per_hour']
        self.max_frames_per_hour = self.config['collection']['max_frames_per_hour']
        
        # ROI settings (hardcoded for now, could be moved to config)
        self.rois = [
            (0.5, 0.50, 1.00, 1.0),  # Fine-tuned upper right
            (0.5, 0.50, 1.00, 1.0),  # Fine-tuned lower right
        ]
        self.roi_weights = [0.4, 0.3, 0.2, 0.1]
        self.use_roi_diversity = self.config['frame_analysis']['use_roi_diversity']
        
        # Feature toggles (could be moved to config)
        self.enable_people_detection = True
        self.enable_metadata_output = True
        self.enable_hourly_tuning = True
        
        # Thread-safe queue for disk writing
        self.cmd_queue = queue.Queue(maxsize=30)
        
        # Tracking recent frames for diversity check
        self.last_frames = deque(maxlen=self.frame_history_size)
        
        # Session statistics
        self.total_frames_captured = 0
        self.frames_skipped_similarity = 0
        
        # Motion tracking variables
        self.motion_counter = 0
        self.last_trigger_time = 0
        
        # Camera and detection objects (will be initialized in start_collection)
        self.cap = None
        self.bg_subtractor = None
        self.person_cascade = None
        self.writer_thread = None
        self.scheduler_thread = None
        
        print("✅ DataCollectionSystem initialized successfully!")
    
    # === CONVERTED UTILITY FUNCTIONS ===
    # These were global functions in your original script, now they're instance methods
    
    def get_free_space_gb(self, path):
        """Check available disk space in gigabytes"""
        total, used, free = shutil.disk_usage(path)
        return free // (2**30)  # Convert bytes to GB
    
    def get_output_dir(self):
        """Determine which directory to use based on available space"""
        return self.primary_dir if self.get_free_space_gb(self.primary_dir) > self.min_gb_free else self.fallback_dir
    
    def safe_queue_put(self, cmd, timeout=1):
        """Add item to queue with timeout to prevent blocking"""
        try:
            self.cmd_queue.put(cmd, block=True, timeout=timeout)
            return True
        except queue.Full:
            print(f"[{datetime.now()}] WARNING: Queue full (size={self.cmd_queue.qsize()}). Dropping frame.")
            return False
    
    # === CONVERTED FRAME ANALYSIS FUNCTIONS ===
    
    def extract_roi(self, frame, roi_coords, frame_width, frame_height):
        """Extract a region of interest from a frame"""
        x1, y1, x2, y2 = roi_coords
        
        # Convert relative coordinates to absolute
        x1_abs = max(0, int(x1 * frame_width))
        y1_abs = max(0, int(y1 * frame_height))
        x2_abs = min(frame_width, int(x2 * frame_width))
        y2_abs = min(frame_height, int(y2 * frame_height))
        
        # Ensure the ROI is valid (non-zero area)
        if x1_abs >= x2_abs or y1_abs >= y2_abs:
            return frame[0:1, 0:1] if frame.size > 0 else np.zeros((1, 1, 3), dtype=np.uint8)
        
        return frame[y1_abs:y2_abs, x1_abs:x2_abs]
    
    def calculate_frame_difference(self, frame1, frame2):
        """Calculate histogram difference between two frames"""
        # Convert to grayscale for histogram comparison
        if len(frame1.shape) == 3:
            frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        else:
            frame1_gray = frame1
            
        if len(frame2.shape) == 3:
            frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            frame2_gray = frame2
        
        # Compare histograms
        hist1 = cv2.calcHist([frame1_gray], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([frame2_gray], [0], None, [64], [0, 256])
        
        # HISTCMP_CORREL returns 1.0 for identical histograms
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Convert to difference score (0-1)
        return 1.0 - similarity
    
    def roi_based_diversity_check(self, new_frame, past_frames, min_difference, frame_width, frame_height):
        """Check frame diversity using multiple regions of interest (ROIs)"""
        if len(past_frames) == 0:
            return True  # Always keep first frame
        
        # For each past frame, check ROI differences
        for past_frame in past_frames:
            # Calculate weighted average difference across all ROIs
            total_weighted_diff = 0
            
            for i, (roi_coords, weight) in enumerate(zip(self.rois, self.roi_weights)):
                # Extract ROIs from both frames
                new_roi = self.extract_roi(new_frame, roi_coords, frame_width, frame_height)
                past_roi = self.extract_roi(past_frame, roi_coords, frame_width, frame_height)
                
                # Skip tiny or empty ROIs
                if new_roi.size == 0 or past_roi.size == 0:
                    continue
                    
                # Calculate difference for this ROI
                roi_diff = self.calculate_frame_difference(new_roi, past_roi)
                
                # Add weighted difference
                total_weighted_diff += roi_diff * weight
            
            # If this frame is too similar to a past frame, reject it
            if total_weighted_diff < min_difference:
                return False
        
        # Different enough from all past frames
        return True
    
    def collect_diverse_training_data(self, frame, last_frames_list, min_difference=0.15, frame_width=640, frame_height=480):
        """Sophisticated frame diversity check focused on training data quality"""
        if len(last_frames_list) == 0:
            return True  # Always keep first frame
        
        # Use ROI-based analysis if enabled
        if self.use_roi_diversity:
            return self.roi_based_diversity_check(
                frame, last_frames_list, min_difference, frame_width, frame_height
            )
        
        # Fall back to whole-frame analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check against last N frames using histogram comparison
        for past_frame in last_frames_list:
            past_gray = cv2.cvtColor(past_frame, cv2.COLOR_BGR2GRAY)
            
            # Compare histograms for more semantic difference detection
            hist1 = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist2 = cv2.calcHist([past_gray], [0], None, [64], [0, 256])
            
            # HISTCMP_CORREL returns 1.0 for identical histograms
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            if similarity > (1 - min_difference):
                # Too similar to a recent frame
                return False
        
        # Different enough from all recent frames
        return True
    
    def contains_people(self, frame, cascade):
        """Check if frame contains people"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        people = cascade.detectMultiScale(gray, 1.1, 3, minSize=(60, 120))
        return len(people) > 0
    
    def draw_rois(self, frame, rois, frame_width, frame_height):
        """Draw ROIs on frame for visualization and debugging"""
        vis_frame = frame.copy()
        
        for i, roi in enumerate(rois):
            x1, y1, x2, y2 = roi
            
            # Convert relative coordinates to absolute
            x1_abs = int(x1 * frame_width)
            y1_abs = int(y1 * frame_height)
            x2_abs = int(x2 * frame_width)
            y2_abs = int(y2 * frame_height)
            
            # Draw ROI rectangle
            color = (0, 255, 0) if i == 0 else (0, 0, 255)
            cv2.rectangle(vis_frame, (x1_abs, y1_abs), (x2_abs, y2_abs), color, 2)
            
            # Add ROI number
            cv2.putText(vis_frame, f"ROI {i+1}", (x1_abs + 5, y1_abs + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return vis_frame
    
    # === CONVERTED SCHEDULING FUNCTIONS ===
    
    def enable_capture(self):
        """Enable frame capture based on schedule"""
        self.capture_enabled = True
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ▶ Capture ENABLED")
    
    def disable_capture(self):
        """Disable frame capture based on schedule"""
        self.capture_enabled = False
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ■ Capture DISABLED")
    
    def print_statistics(self):
        """Print capture statistics periodically"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Stats: {self.total_frames_captured} frames captured, "
              f"{self.frames_skipped_similarity} skipped due to similarity")
    
    def setup_schedule(self):
        """Set up the capture schedule"""
        # Clear any existing schedule
        schedule.clear()
        
        # Morning
        schedule.every().day.at("06:00").do(self.enable_capture)
        schedule.every().day.at("09:00").do(self.disable_capture)
        # Lunch time
        schedule.every().day.at("11:30").do(self.enable_capture)
        schedule.every().day.at("13:30").do(self.disable_capture)
        # Afternoon
        schedule.every().day.at("13:30").do(self.enable_capture)
        schedule.every().day.at("17:00").do(self.disable_capture)
        # Evening
        schedule.every().day.at("18:00").do(self.enable_capture)
        schedule.every().day.at("20:00").do(self.disable_capture)
        # Night
        schedule.every().day.at("20:00").do(self.enable_capture)
        schedule.every().day.at("23:59").do(self.disable_capture)
        # Statistics reporting
        schedule.every().hour.do(self.print_statistics)
    
    def run_scheduler(self):
        """Thread function to run the scheduler"""
        while True:
            schedule.run_pending()
            time.sleep(30)  # Check schedule every 30 seconds
    
    # === DISK WRITER THREAD ===
    
    def writer_worker(self):
        """Background thread that handles writing frames to disk"""
        while True:
            try:
                data = self.cmd_queue.get()
                if isinstance(data, tuple) and len(data) == 2:  # Standard frame
                    fname, frame = data
                    cv2.imwrite(fname, frame)
                elif isinstance(data, tuple) and len(data) == 3:  # Frame with metadata
                    fname, frame, metadata = data
                    cv2.imwrite(fname, frame)
                    # Save metadata with same base filename but different extension
                    if self.enable_metadata_output:
                        metadata_file = os.path.join(
                            self.metadata_dir, 
                            os.path.basename(fname).replace('.jpg', '.json')
                        )
                        with open(metadata_file, 'w') as f:
                            json.dump(metadata, f)
                self.cmd_queue.task_done()
            except Exception as e:
                print(f"[{datetime.now()}] ERROR in writer thread: {e}")
    
    # === MAIN EXECUTION METHOD ===
    # This is where your original main capture logic goes
    
    def start_collection(self):
        """
        Start the data collection process.
        
        This method contains all the logic from your original script's main execution section.
        It's the equivalent of everything after "=== MAIN CAPTURE LOGIC ===" in your original.
        """
        print("📸 Enhanced training data collection starting...")
        print(f"📁 Saving to: {self.primary_dir} (fallback: {self.fallback_dir})")
        
        # Set up and start the scheduler
        self.setup_schedule()
        self.scheduler_thread = threading.Thread(target=self.run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        # Start the writer thread
        self.writer_thread = threading.Thread(target=self.writer_worker, daemon=True)
        self.writer_thread.start()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.config['camera']['index'])
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {self.config['camera']['index']}")
        
        # Get camera properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📹 Camera resolution: {width}x{height}")
        
        # Initialize background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=False
        )
        
        # Initialize person detector
        self.person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        
        try:
            # === MAIN CAPTURE LOOP ===
            # This is the main while loop from your original script
            while True:
                # Get current frame
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️ Frame grab failed.")
                    time.sleep(0.5)
                    continue
                
                # Skip processing if capture is disabled by schedule
                if not self.capture_enabled:
                    time.sleep(1)
                    continue
                
                # Current timestamp for logging and naming
                now = time.time()
                current_datetime = datetime.now()
                
                # Check if we entered a new hour for hour-based diversity
                current_hour = current_datetime.hour
                if self.enable_hourly_tuning and current_hour != self.last_hour_captured:
                    self.last_hour_captured = current_hour
                    self.current_hour_frame_count = 0
                    print(f"[{current_datetime.strftime('%H:%M:%S')}] ⏱️ New hour: Adjusting capture parameters")
                    
                    # Be less strict about frame differences at the start of a new hour
                    if self.current_hour_frame_count < self.min_frames_per_hour * 0.25:
                        dynamic_min_difference = self.min_difference_threshold * 0.7
                    else:
                        dynamic_min_difference = self.min_difference_threshold
                
                # Motion detection processing
                small = cv2.resize(frame, (int(width * self.scale_factor), int(height * self.scale_factor)))
                
                # Apply background subtraction
                mask = self.bg_subtractor.apply(small, learningRate=0.001)
                
                # Clean up the mask with morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.small_kernel)
                clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # Find contours in the mask
                contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Calculate the largest motion area
                max_area = max((cv2.contourArea(c) for c in contours), default=0)
                small_area = (width * self.scale_factor) * (height * self.scale_factor)
                
                # Check if the motion is significant
                has_significant_motion = max_area > (self.min_blob_area * small_area)
                
                # Check for people in the frame
                has_people = self.contains_people(small, self.person_cascade) if self.enable_people_detection else False
                
                # Increment motion counter based on detections
                if has_significant_motion:
                    self.motion_counter += 1
                    if has_people:
                        self.motion_counter += 2
                else:
                    self.motion_counter = max(0, self.motion_counter - 1)
                
                # If we've detected enough consecutive motion frames and we're not in cooldown
                if self.motion_counter >= self.persist_frames and (now - self.last_trigger_time) > self.cooldown_sec:
                    ts = current_datetime.strftime("%Y%m%d_%H%M%S")
                    print(f"[{ts}] Motion detected → Capturing burst for {self.burst_duration} seconds")
                    
                    # Calculate frame interval based on target FPS
                    frame_interval = 1.0 / self.burst_fps
                    burst_end = now + self.burst_duration
                    
                    # Dynamic parameters based on hour-based collection stats
                    if self.current_hour_frame_count < self.min_frames_per_hour:
                        actual_min_difference = self.min_difference_threshold * 0.7
                        actual_burst_fps = self.burst_fps * 1.5
                    elif self.current_hour_frame_count > self.max_frames_per_hour:
                        actual_min_difference = self.min_difference_threshold * 1.3
                        actual_burst_fps = self.burst_fps * 0.7
                    else:
                        actual_min_difference = self.min_difference_threshold
                        actual_burst_fps = self.burst_fps
                    
                    frame_interval = 1.0 / actual_burst_fps
                    
                    # Burst capture loop
                    burst_frames_captured = 0
                    while time.time() < burst_end:
                        ret, frame = self.cap.read()
                        if not ret:
                            break
                        
                        # Check if this frame is sufficiently different from recent ones
                        if not self.collect_diverse_training_data(
                            frame, list(self.last_frames), 
                            min_difference=actual_min_difference,
                            frame_width=width,
                            frame_height=height
                        ):
                            self.frames_skipped_similarity += 1
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🟡 Frame skipped due to similarity")
                            time.sleep(frame_interval * 0.5)
                            continue
                        
                        # Frame is different enough, proceed with saving
                        save_dir = self.get_output_dir()
                        frame_ts = int((time.time() - now) * 1000)
                        frame_time_str = current_datetime.strftime("%Y-%m-%dT%H-%M-%S")
                        people_tag = "Y" if has_people else "N"
                        motion_pct = 100.0 * (max_area / small_area)
                        fname = os.path.join(
                            save_dir,
                            f"{frame_time_str}_ms{frame_ts:04d}_p{people_tag}_m{motion_pct:.2f}.jpg"
                        )
                        
                        # Create metadata for this frame
                        metadata = {
                            'timestamp': current_datetime.isoformat(),
                            'hour': current_datetime.hour,
                            'dayofweek': current_datetime.weekday(),
                            'motion_size': float(max_area / small_area),
                            'has_people': has_people,
                            'frame_number': self.total_frames_captured
                        }
                        
                        # Queue the frame and metadata for saving
                        self.safe_queue_put((fname, frame, metadata))
                        
                        # Update tracking variables
                        self.last_frames.append(frame.copy())
                        burst_frames_captured += 1
                        self.total_frames_captured += 1
                        self.current_hour_frame_count += 1
                        
                        time.sleep(frame_interval)
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Burst complete, captured {burst_frames_captured} frames")
                    
                    # Reset motion counter and update last trigger time
                    self.last_trigger_time = time.time()
                    self.motion_counter = 0
                
                # Display live feed if not in headless mode
                if not self.headless:
                    status_frame = frame.copy()
                    
                    # Draw ROIs for debugging if enabled
                    if self.use_roi_diversity:
                        status_frame = self.draw_rois(status_frame, self.rois, width, height)
                    
                    cv2.putText(status_frame, f"Motion: {self.motion_counter}/{self.persist_frames}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(status_frame, f"Hour frames: {self.current_hour_frame_count}/{self.max_frames_per_hour}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(status_frame, f"ROI Mode: {'ON' if self.use_roi_diversity else 'OFF'}", 
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow("Live Feed", status_frame)
                    if 'clean' in locals():
                        cv2.imshow("Motion Mask", clean)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        except KeyboardInterrupt:
            print("🛑 Interrupted by user.")
        finally:
            self.stop_collection()
    
    def stop_collection(self):
        """Clean up resources and stop data collection"""
        print("🛑 Stopping data collection...")
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"📊 Final statistics: {self.total_frames_captured} frames captured, "
              f"{self.frames_skipped_similarity} skipped due to similarity")
    
    def print_config_summary(self):
        """Print a detailed summary of the current configuration"""
        print("\n" + "="*60)
        print("DATA COLLECTION CONFIGURATION SUMMARY")
        print("="*60)
        
        # Camera configuration
        print(f"📷 Camera Index: {self.config['camera']['index']}")
        print(f"🖥️  Headless Mode: {self.headless}")
        
        # Storage configuration
        print(f"💾 Primary Directory: {self.primary_dir}")
        print(f"🔄 Fallback Directory: {self.fallback_dir}")
        print(f"📋 Metadata Directory: {self.metadata_dir}")
        
        # Collection behavior configuration
        print(f"⏱️  Burst Duration: {self.burst_duration}s")
        print(f"📸 Burst FPS: {self.burst_fps}")
        print(f"⏸️  Cooldown Period: {self.cooldown_sec}s")
        print(f"📊 Frames per Hour: {self.min_frames_per_hour}-{self.max_frames_per_hour}")
        
        # Motion detection configuration
        print(f"🎯 Min Blob Area: {self.min_blob_area}")
        print(f"📏 Scale Factor: {self.scale_factor}")
        print(f"🔄 Persist Frames: {self.persist_frames}")
        
        # Frame analysis configuration
        print(f"📚 Frame History Size: {self.frame_history_size}")
        print(f"🔍 Min Difference Threshold: {self.min_difference_threshold}")
        print(f"🎯 Use ROI Diversity: {self.use_roi_diversity}")
        
        # Feature toggles
        print(f"👥 People Detection: {self.enable_people_detection}")
        print(f"📄 Metadata Output: {self.enable_metadata_output}")
        print(f"⏰ Hourly Tuning: {self.enable_hourly_tuning}")
        
        print("="*60 + "\n")


def main():
    """
    Main entry point with enhanced configuration support.
    
    This function replaces the global execution logic from your original script
    and provides a clean command-line interface for the data collection system.
    """
    import argparse
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Waste District Vision Data Collection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration Examples:
  python main_data_collection_v2.py                    # Use default + local config
  python main_data_collection_v2.py --show-config     # Show configuration and exit
  python main_data_collection_v2.py --config custom.yaml  # Use custom config file
  
Environment Variables:
  CAMERA_INDEX=1 python main_data_collection_v2.py    # Override camera index
  HEADLESS=1 python main_data_collection_v2.py        # Run in headless mode
        """
    )
    
    parser.add_argument('--config', '-c', 
                       help='Path to custom configuration file (optional)')
    parser.add_argument('--show-config', action='store_true',
                       help='Display configuration summary and exit')
    
    args = parser.parse_args()
    
    try:
        # Initialize the data collection system with the specified configuration
        print("🚀 Initializing Waste District Vision Data Collection System...")
        system = DataCollectionSystem(config_path=args.config)
        
        # If user just wants to see the configuration, show it and exit
        if args.show_config:
            system.print_config_summary()
            print("Configuration displayed. Exiting without starting data collection.")
            return
        
        # Show configuration summary before starting
        system.print_config_summary()
        
        # Start the data collection process
        print("🎬 Starting data collection process...")
        print("Press Ctrl+C to stop data collection gracefully.")
        system.start_collection()
        
    except KeyboardInterrupt:
        print("\n⏹️  Received interrupt signal. Stopping data collection...")
        if 'system' in locals():
            system.stop_collection()
        print("✅ Data collection stopped successfully.")
    except FileNotFoundError as e:
        print(f"❌ Configuration Error: {e}")
        print("💡 Make sure your configuration files exist in the configs/ directory.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()