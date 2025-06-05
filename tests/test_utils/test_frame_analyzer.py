#!/usr/bin/env python3
from src.data.collectors.frame_analyzer import FrameAnalyzer
import cv2
import numpy as np

def test_frame_analyzer():
    print("🔍 Testing frame analyzer...")
    
    # Create analyzer
    analyzer = FrameAnalyzer(
        frame_history_size=3,
        min_difference_threshold=0.1,
        use_roi_diversity=True
    )
    
    # Test with dummy frames
    print("Creating test frames...")
    
    # Create similar frames (should be rejected)
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2[100:200, 100:200] = 50  # Slight difference
    
    # Create very different frame (should be accepted)
    frame3 = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Test diversity checking
    result1 = analyzer.is_frame_diverse(frame1)
    result2 = analyzer.is_frame_diverse(frame2)  # Should be rejected (too similar)
    result3 = analyzer.is_frame_diverse(frame3)  # Should be accepted (very different)
    
    print(f"Frame 1 (first): {result1} ✅")
    print(f"Frame 2 (similar): {result2} ❌ (expected)")
    print(f"Frame 3 (different): {result3} ✅")
    
    # Test ROI drawing
    vis_frame = analyzer.draw_rois(frame1)
    print(f"ROI visualization created: {vis_frame.shape}")
    
    # Get statistics
    stats = analyzer.get_stats()
    print(f"Stats: {stats}")
    
    print("✅ Frame analyzer test complete!")

if __name__ == "__main__":
    test_frame_analyzer()