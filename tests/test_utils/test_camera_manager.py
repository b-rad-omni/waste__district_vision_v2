from src.data.collectors.camera_manager import CameraManager
import cv2

def test_camera_manager():
    print("📹 Testing camera manager...")
    
    try:
        # Test camera initialization
        camera = CameraManager(camera_index=2)  # Use your camera index
        camera.initialize_camera()
        
        print(f"✅ Camera initialized successfully")
        
        # Test camera properties
        props = camera.get_camera_properties()
        print(f"📊 Camera properties: {props}")
        
        # Test frame reading
        success, frame = camera.read_frame()
        if success:
            print(f"✅ Frame read successful: {frame.shape}")
        else:
            print("❌ Frame read failed")
        
        # Test a few more frames
        for i in range(3):
            success, frame = camera.read_frame()
            if success:
                print(f"Frame {i+1}: {frame.shape}")
            else:
                print(f"❌ Frame {i+1} failed")
                break
        
        # Clean up
        camera.release()
        print("✅ Camera manager test complete!")
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")

def test_context_manager():
    print("🔄 Testing context manager...")
    
    try:
        # Test using context manager (with statement)
        with CameraManager(camera_index=2) as camera:
            success, frame = camera.read_frame()
            if success:
                print(f"✅ Context manager frame: {frame.shape}")
            else:
                print("❌ Context manager frame failed")
        
        print("✅ Context manager test complete!")
        
    except Exception as e:
        print(f"❌ Context manager test failed: {e}")

if __name__ == "__main__":
    test_camera_manager()
    print()
    test_context_manager()