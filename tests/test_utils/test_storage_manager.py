#!/usr/bin/env python3
from src.utils.storage_manager import StorageManager
import tempfile
import os
from datetime import datetime

def test_storage_manager():
    print("💾 Testing storage manager...")
    
    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        primary_dir = os.path.join(temp_dir, "primary")
        fallback_dir = os.path.join(temp_dir, "fallback")
        
        # Create storage manager
        storage = StorageManager(
            primary_dir=primary_dir,
            fallback_dir=fallback_dir,
            min_gb_free=0.001  # Very small for testing
        )
        
        print("✅ Storage manager created")
        
        # Test directory creation
        assert os.path.exists(primary_dir), "Primary directory not created"
        assert os.path.exists(fallback_dir), "Fallback directory not created"
        print("✅ Directories created successfully")
        
        # Test free space checking
        primary_free = storage.get_free_space_gb(primary_dir)
        print(f"📊 Primary free space: {primary_free}GB")
        
        # Test filename generation
        filename = storage.generate_filename(
            ms=1234,
            people=True,
            motion_pct=45.67
        )
        print(f"📝 Generated filename: {filename}")
        
        # Test full path generation
        full_path = storage.get_full_path(filename)
        print(f"📁 Full path: {full_path}")
        
        # Test metadata saving
        test_metadata = {
            'timestamp': datetime.now().isoformat(),
            'motion_size': 0.05,
            'has_people': True,
            'frame_number': 42
        }
        
        storage.save_metadata(filename, test_metadata)
        print("✅ Metadata saved")
        
        # Test storage stats
        stats = storage.get_storage_stats()
        print(f"📊 Storage stats: {stats}")
        
        print("✅ Storage manager test complete!")

if __name__ == "__main__":
    test_storage_manager()