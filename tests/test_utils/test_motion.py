from src.data.collectors.motion_detector import MotionDetector
import cv2

detector = MotionDetector()
cap = cv2.VideoCapture(0)  # Use your camera index

print("Testing motion detector... Press 'q' to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    result = detector.detect_motion(frame)
    print(f"Motion: {result['motion_counter']}/{detector.persist_frames}, "
          f"Percentage: {result['motion_percentage']:.1f}%")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
print("✅ Motion detector test complete!")