import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class InfractionEvent:
    """Data class for infraction events"""
    track_id: int
    zone: str
    object_class: str
    entry_count: int
    timestamp: datetime
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    severity: str  # 'warning', 'critical'
    clip_start_time: Optional[datetime] = None


class BucketInfractionDetector:
    """Specialized detector for bucket/large object infractions around bin doors"""
    
    def __init__(self, debug=False):
        # Target classes for infraction detection
        self.TARGET_CLASSES = ['bucket', 'large foreign object']
        self.debug = debug
        
        # Bin door zones (coordinates for location in bottom-right camera)
        self.BIN_ZONES = {
            'bin_door_1': [(1110, 838), (1182, 797), (1142, 891), (1211, 852)],
            'bin_door_2': [(1215, 786), (1283, 747), (1323, 791), (1250, 831)],
            
            #MKV
            'bin_door_5': [(1300,952), (1417, 878), (1496, 955),(1383, 1042)],
            'bin_door_6': [(1465, 843), (1557, 785), (1621, 848), (1546, 919)],
            'bin_door_7': [(1589, 759), (1647, 723), (1702, 775), (1655, 822)],
            'bin_door_8': [(1671, 706), (1707, 678), (1754, 724), (1724, 757)],
            
            #JPEG
            #'bin_door_5': [(1250,939), (1367, 869), (1442, 949),(1324, 1028)],
            #'bin_door_6': [(1416, 839), (1508, 789), (1577, 850), (1486, 917)],
            #'bin_door_7': [(1547, 762), (1613, 725), (1667, 779), (1612, 820)],
            #'bin_door_8': [(1639, 709), (1683, 682), (1734, 726), (1697, 762)],
        } 
        
        # Infraction rules
        self.RULES = {
            'max_entries_threshold': 2,        # Trigger after 2 entries
            'time_window_minutes': 2,         # Within 3 minutes
            'entry_cooldown_seconds': 5,      # Minimum time between counting entries
            'min_confidence': 0.38,             # Minimum detection confidence
            'alert_cooldown_minutes': 3       # Prevent spam alerts
        }
        
        # Tracking data - improved structure
        self.zone_entries = {}          # {zone_name: [entry_events]}
        self.track_positions = {}       # {track_id: [(bbox, timestamp)]}
        self.last_zone_exit = {}        # {track_id: {zone: timestamp}}
        self.active_infractions = {}    # {track_id: InfractionEvent}
        self.alert_cooldowns = {}       # {zone: last_alert_time}
        self.track_zone_history = {}    # {track_id: {zone: last_entry_time}}
        
    def process_detections(self, detections, frame_timestamp):
        """
        Process YOLO/ByteTrack detections for bucket infractions
        
        Args:
            detections: List of detection objects with track_id, class_name, bbox, confidence
            frame_timestamp: Current frame timestamp
        """
        if self.debug:
            print(f"🚨 BUCKET DEBUG: Processing {len(detections)} detections at {datetime.fromtimestamp(frame_timestamp)}")
        
        infractions = []
        current_time = datetime.fromtimestamp(frame_timestamp)
        
        target_detections = 0
        
        # Process each detection
        for detection in detections:
            if self.debug:
                print(f"  🔍 Detection: {detection.get('class_name', 'NO_CLASS')} - Keys: {list(detection.keys())}")
            
            # Validate detection structure
            if not self._validate_detection(detection):
                continue
            
            # Filter for target classes only
            if detection['class_name'].lower() not in [cls.lower() for cls in self.TARGET_CLASSES]:
                if self.debug:
                    print(f"    ❌ Not target class: {detection['class_name']}")
                continue
                
            if self.debug:
                print(f"    ✅ Target class found: {detection['class_name']}")
            target_detections += 1
                
            # Check confidence threshold (fixed logic)
            confidence = detection.get('confidence', 1.0)  # Default to 1.0 if no confidence
            if confidence < self.RULES['min_confidence']:
                if self.debug:
                    print(f"    ❌ Low confidence: {confidence} < {self.RULES['min_confidence']}")
                continue
            
            track_id = detection['track_id']
            bbox = detection['bbox']
            class_name = detection['class_name']
            
            if self.debug:
                print(f"    🎯 Processing {class_name} (ID: {track_id}, conf: {confidence:.2f})")
            
            # Update track position history
            self._update_track_history(track_id, bbox, current_time)
            
            # Check each bin zone
            bbox_center = self._get_bbox_center(bbox)
            if self.debug:
                print(f"    📍 Bbox center: {bbox_center}")
            
            in_any_zone = False
            for zone_name, zone_coords in self.BIN_ZONES.items():
                if self._point_in_polygon(bbox_center, zone_coords):
                    if self.debug:
                        print(f"    🏠 IN ZONE: {zone_name}")
                    in_any_zone = True
                    
                    # Check if this is a new entry for this track in this zone
                    if self._is_new_zone_entry(track_id, zone_name, current_time):
                        infraction = self._process_zone_entry(
                            track_id, zone_name, class_name, bbox, current_time
                        )
                        if infraction:
                            infractions.append(infraction)
                else:
                    # Object left this zone - record exit
                    self._record_zone_exit(track_id, zone_name, current_time)
            
            if self.debug and not in_any_zone:
                print(f"    ❌ Not in any zone")
        
        if self.debug:
            print(f"🎯 Summary: {target_detections} target detections, {len(infractions)} infractions")
        
        # Clean up old data
        self._cleanup_old_data(current_time)
        
        return infractions
    
    def _validate_detection(self, detection):
        """Validate detection has required fields"""
        required_fields = ['class_name', 'bbox']
        
        for field in required_fields:
            if field not in detection:
                if self.debug:
                    print(f"    ❌ Missing required field: {field}")
                return False
        
        # Generate track_id if missing (for non-tracking scenarios)
        if 'track_id' not in detection:
            # Use bbox center as simple track ID for stateless detection
            bbox = detection['bbox']
            center = self._get_bbox_center(bbox)
            detection['track_id'] = f"det_{center[0]}_{center[1]}"
            if self.debug:
                print(f"    🆔 Generated track_id: {detection['track_id']}")
        
        return True
    
    def _is_new_zone_entry(self, track_id, zone_name, timestamp):
        """Determine if this is a genuine new entry vs continuous presence"""
        
        # Check track zone history for this specific track and zone
        if track_id in self.track_zone_history:
            if zone_name in self.track_zone_history[track_id]:
                last_entry = self.track_zone_history[track_id][zone_name]
                time_since_last = (timestamp - last_entry).total_seconds()
                
                if time_since_last < self.RULES['entry_cooldown_seconds']:
                    if self.debug:
                        print(f"      ⏳ Cooldown active: {time_since_last:.1f}s < {self.RULES['entry_cooldown_seconds']}s")
                    return False
        
        # Check if we have exit data for this track/zone
        if (track_id in self.last_zone_exit and 
            zone_name in self.last_zone_exit[track_id]):
            
            last_exit = self.last_zone_exit[track_id][zone_name]
            time_since_exit = (timestamp - last_exit).total_seconds()
            
            # Must have been outside for cooldown period
            if time_since_exit >= self.RULES['entry_cooldown_seconds']:
                return True
            else:
                if self.debug:
                    print(f"      🚪 Too soon since exit: {time_since_exit:.1f}s")
                return False
        
        # This is a new track or first time in this zone
        return True
    
    def _process_zone_entry(self, track_id, zone_name, class_name, bbox, timestamp):
        """Process potential zone entry - track ALL DOORS combined"""
        
        # Use a single "all_doors" key instead of individual zones
        combined_zone = "all_doors"
        
        # Initialize combined tracking
        if combined_zone not in self.zone_entries:
            self.zone_entries[combined_zone] = []
        
        # Check if this is actually a new entry (add cooldown logic)
        is_new = True
        if self.zone_entries[combined_zone]:
            last_entry_time = self.zone_entries[combined_zone][-1]['timestamp']
            time_since_last = (timestamp - last_entry_time).total_seconds()
            if time_since_last < self.RULES['entry_cooldown_seconds']:
                is_new = False
        
        if is_new:
            # Record the entry with which specific door
            self.zone_entries[combined_zone].append({
                'timestamp': timestamp,
                'track_id': track_id,
                'class': class_name,
                'door': zone_name  # Track which door but count them all together
            })
            
            print(f"🪣 {class_name} (ID: {track_id}) entered {zone_name} - TOTAL entries: {len(self.zone_entries[combined_zone])}")
            
            # Clean old entries
            self._clean_old_zone_entries(combined_zone, timestamp)
            
            # Check for infraction based on TOTAL entries across all doors
            total_entries = len(self.zone_entries[combined_zone])
            if total_entries >= self.RULES['max_entries_threshold']:
                return self._create_infraction_event(
                    track_id, f"multiple_doors", class_name, bbox, timestamp, total_entries
                )

        return None
    
    def _clean_old_zone_entries(self, zone_name, current_time):
        """Remove entries outside the time window for a specific zone"""
        cutoff_time = current_time - timedelta(minutes=self.RULES['time_window_minutes'])
        
        if zone_name in self.zone_entries:
            old_count = len(self.zone_entries[zone_name])
            self.zone_entries[zone_name] = [
                entry for entry in self.zone_entries[zone_name]
                if entry['timestamp'] >= cutoff_time
            ]
            new_count = len(self.zone_entries[zone_name])
            
            if self.debug and old_count != new_count:
                print(f"    🧹 Cleaned {zone_name}: {old_count} -> {new_count} entries")
    
    def _cleanup_old_data(self, current_time):
        """Clean up old tracking data to prevent memory leaks"""
        cutoff_time = current_time - timedelta(hours=2)
        
        # Clean track positions
        for track_id in list(self.track_positions.keys()):
            if self.track_positions[track_id]:
                self.track_positions[track_id] = [
                    (bbox, timestamp) for bbox, timestamp in self.track_positions[track_id]
                    if timestamp >= cutoff_time
                ]
                
                # Remove empty tracks
                if not self.track_positions[track_id]:
                    del self.track_positions[track_id]
        
        # Clean track zone history
        for track_id in list(self.track_zone_history.keys()):
            for zone_name in list(self.track_zone_history[track_id].keys()):
                if self.track_zone_history[track_id][zone_name] < cutoff_time:
                    del self.track_zone_history[track_id][zone_name]
            
            # Remove empty track entries
            if not self.track_zone_history[track_id]:
                del self.track_zone_history[track_id]
    
    def _record_zone_exit(self, track_id, zone_name, timestamp):
        """Record when a track exits a zone"""
        if track_id not in self.last_zone_exit:
            self.last_zone_exit[track_id] = {}
        self.last_zone_exit[track_id][zone_name] = timestamp
    
    def _create_infraction_event(self, track_id, zone_name, class_name, bbox, timestamp, entry_count):
        """Create an infraction event"""
        
        # Check alert cooldown to prevent spam
        if zone_name in self.alert_cooldowns:
            time_since_last_alert = (timestamp - self.alert_cooldowns[zone_name]).total_seconds()
            if time_since_last_alert < self.RULES['alert_cooldown_minutes'] * 60:
                if self.debug:
                    print(f"    🔇 Alert cooldown active for {zone_name}")
                return None  # Still in cooldown
        
        # Determine severity
        severity = 'critical' if entry_count >= self.RULES['max_entries_threshold'] else 'warning'
        
        # Create infraction event
        infraction = InfractionEvent(
            track_id=track_id,
            zone=zone_name,
            object_class=class_name,
            entry_count=entry_count,
            timestamp=timestamp,
            bbox=bbox,
            severity=severity,
            clip_start_time=timestamp - timedelta(minutes=2)  # Start clip 2 min before
        )
        
        # Update cooldown
        self.alert_cooldowns[zone_name] = timestamp
        
        # Store active infraction
        self.active_infractions[track_id] = infraction
        
        print(f"🚨 INFRACTION DETECTED: {class_name} in {zone_name} - {entry_count} entries!")
        
        return infraction
    
    def _update_track_history(self, track_id, bbox, timestamp):
        """Update position history for track"""
        if track_id not in self.track_positions:
            self.track_positions[track_id] = []
        
        # Add current position
        self.track_positions[track_id].append((bbox, timestamp))
        
        # Keep only recent history (last 100 positions)
        if len(self.track_positions[track_id]) > 100:
            self.track_positions[track_id] = self.track_positions[track_id][-100:]
    
    def _point_in_polygon(self, point, polygon):
        """Check if point is inside polygon using ray casting"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _get_bbox_center(self, bbox):
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def get_active_infractions(self):
        """Get list of currently active infractions"""
        return list(self.active_infractions.values())
    
    def clear_infraction(self, track_id):
        """Clear/acknowledge an infraction"""
        if track_id in self.active_infractions:
            del self.active_infractions[track_id]
            print(f"✅ Infraction for track {track_id} cleared")
    
    def get_zone_statistics(self):
        """Get statistics for each zone"""
        stats = {}
        current_time = datetime.now()
        
        for zone_name in self.BIN_ZONES.keys():
            total_entries = 0
            active_infractions = 0
            
            # Count entries in this zone in last hour
            if zone_name in self.zone_entries:
                recent_entries = [
                    entry for entry in self.zone_entries[zone_name]
                    if (current_time - entry['timestamp']).total_seconds() <= 3600
                ]
                total_entries = len(recent_entries)
            
            # Check if this zone has active infractions
            for infraction in self.active_infractions.values():
                if infraction.zone == zone_name:
                    active_infractions += 1
            
            stats[zone_name] = {
                'total_entries_last_hour': total_entries,
                'active_infractions': active_infractions,
                'last_alert': self.alert_cooldowns.get(zone_name, None)
            }
        
        return stats
    
    def toggle_debug(self, enabled=None):
        """Enable/disable debug mode"""
        if enabled is None:
            self.debug = not self.debug
        else:
            self.debug = enabled
        print(f"🐛 Debug mode: {'ON' if self.debug else 'OFF'}")


# Integration with your detection pipeline
class BucketMonitoringPipeline:
    """Integration wrapper for your YOLO + ByteTrack pipeline"""
    
    def __init__(self, debug=False):
        self.infraction_detector = BucketInfractionDetector(debug=debug)
        self.video_recorder = None  # Your video recording component
        
    def process_frame(self, frame, detections, timestamp):
        """Process a single frame"""
        
        # Run infraction detection
        infractions = self.infraction_detector.process_detections(detections, timestamp)
        
        # Handle any new infractions
        for infraction in infractions:
            self._handle_infraction(infraction, frame, timestamp)
        
        return infractions
    
    def _handle_infraction(self, infraction, frame, timestamp):
        """Handle detected infraction"""
        
        # Log the infraction
        print(f"🚨 BUCKET INFRACTION: {infraction.object_class} in {infraction.zone}")
        print(f"   Entry count: {infraction.entry_count}")
        print(f"   Severity: {infraction.severity}")
        print(f"   Track ID: {infraction.track_id}")
        print(f"   Timestamp: {infraction.timestamp}")
        
        # Start recording clip if not already recording
        if self.video_recorder:
            clip_path = f"infractions/bucket_infraction_{infraction.track_id}_{timestamp}.mp4"
            self.video_recorder.start_clip(clip_path, duration=30)  # 30 second clip
        
        # Send alert (implement your notification system)
        self._send_alert(infraction)
    
    def _send_alert(self, infraction):
        """Send alert notification (email, SMS, dashboard update, etc.)"""
        # TODO: Implement your notification system here
        # Example:
        # - Send email
        # - Update dashboard
        # - Log to database
        # - Send SMS alert
        pass