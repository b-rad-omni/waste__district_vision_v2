"""
Storage Management for Data Collection
"""
import os
import shutil
import json
from datetime import datetime
from pathlib import Path

class StorageManager:
    def __init__(self, primary_dir, fallback_dir, metadata_dir=None, min_gb_free=5):
        """
        Initialize storage manager
        
        Args:
            primary_dir: Primary storage location
            fallback_dir: Backup storage if primary is full
            metadata_dir: Directory for metadata files
            min_gb_free: Minimum GB to keep free on primary storage
        """
        self.primary_dir = Path(primary_dir)
        self.fallback_dir = Path(fallback_dir)
        self.min_gb_free = min_gb_free
        
        # Create directories
        self.primary_dir.mkdir(parents=True, exist_ok=True)
        self.fallback_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata directory
        if metadata_dir:
            self.metadata_dir = Path(metadata_dir)
        else:
            self.metadata_dir = self.primary_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
    def get_free_space_gb(self, path):
        """Check available disk space in gigabytes"""
        total, used, free = shutil.disk_usage(path)
        return free // (2**30)  # Convert bytes to GB
    
    def get_output_dir(self):
        """Determine which directory to use based on available space"""
        primary_free = self.get_free_space_gb(self.primary_dir)
        
        if primary_free > self.min_gb_free:
            return self.primary_dir
        else:
            print(f"⚠️ Primary storage low ({primary_free}GB), using fallback")
            return self.fallback_dir
    
    def generate_filename(self, timestamp=None, **kwargs):
        """
        Generate filename with timestamp and metadata
        
        Args:
            timestamp: datetime object (defaults to now)
            **kwargs: Additional metadata for filename
            
        Returns:
            str: Generated filename
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Base timestamp
        time_str = timestamp.strftime("%Y-%m-%dT%H-%M-%S")
        
        # Add optional metadata to filename
        parts = [time_str]
        
        if 'ms' in kwargs:
            parts.append(f"ms{kwargs['ms']:04d}")
        if 'people' in kwargs:
            parts.append(f"p{'Y' if kwargs['people'] else 'N'}")
        if 'motion_pct' in kwargs:
            parts.append(f"m{kwargs['motion_pct']:.2f}")
        
        filename = "_".join(parts) + ".jpg"
        return filename
    
    def get_full_path(self, filename):
        """Get full path for saving file"""
        output_dir = self.get_output_dir()
        return output_dir / filename
    
    def save_metadata(self, filename, metadata):
        """
        Save metadata for a frame
        
        Args:
            filename: Image filename (will change extension to .json)
            metadata: Dictionary of metadata to save
        """
        # Create metadata filename
        base_name = Path(filename).stem
        metadata_file = self.metadata_dir / f"{base_name}.json"
        
        # Add storage info to metadata
        metadata_with_storage = {
            **metadata,
            'storage_info': {
                'primary_dir': str(self.primary_dir),
                'fallback_dir': str(self.fallback_dir),
                'saved_to': str(self.get_output_dir()),
                'primary_free_gb': self.get_free_space_gb(self.primary_dir),
                'fallback_free_gb': self.get_free_space_gb(self.fallback_dir)
            }
        }
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata_with_storage, f, indent=2)
    
    def get_storage_stats(self):
        """Get current storage statistics"""
        return {
            'primary_dir': str(self.primary_dir),
            'fallback_dir': str(self.fallback_dir),
            'primary_free_gb': self.get_free_space_gb(self.primary_dir),
            'fallback_free_gb': self.get_free_space_gb(self.fallback_dir),
            'using_primary': self.get_free_space_gb(self.primary_dir) > self.min_gb_free,
            'min_gb_threshold': self.min_gb_free
        }