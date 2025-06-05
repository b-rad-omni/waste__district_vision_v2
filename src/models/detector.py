"""YOLOv8 detection model wrapper."""

from typing import List, Optional, Tuple
import torch
from ultralytics import YOLO
import cv2
import numpy as np


class YOLOv8Detector:
    """YOLOv8 object detection wrapper."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize the detector.
        
        Args:
            model_path: Path to the trained model
            device: Device to run inference on
        """
        self.model = YOLO(model_path)
        self.device = device
        
    def predict(
        self, 
        image: np.ndarray,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4
    ) -> List[dict]:
        """Run inference on an image.
        
        Args:
            image: Input image as numpy array
            conf_threshold: Confidence threshold
            nms_threshold: NMS threshold
            
        Returns:
            List of detection dictionaries
        """
        results = self.model(
            image,
            conf=conf_threshold,
            iou=nms_threshold,
            device=self.device
        )
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    detection = {
                        'bbox': box.xyxy[0].cpu().numpy().tolist(),
                        'confidence': float(box.conf[0]),
                        'class_id': int(box.cls[0]),
                        'class_name': self.model.names[int(box.cls[0])]
                    }
                    detections.append(detection)
        
        return detections
