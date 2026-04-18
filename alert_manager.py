"""
Alert Manager - Orchestrate alert logic and decision making
"""
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
import winsound
from config import SNAPSHOT_DIR, VIDEO_DIR, LOG_DIR
from video_buffer import VideoBuffer
from alert_sender import AlertSender, AlertLogger
from location_mapper import LocationMapper
from detection_refactored import PersonDetector, AlertRuleEngine


class AlertManager:
    """
    Manage alerts, video buffers, and alert notifications
    """
    
    def __init__(self, video_buffer_seconds: int = 30, fps: int = 30):
        """
        Initialize alert manager
        
        Args:
            video_buffer_seconds: Seconds of video to keep in buffer
            fps: Frames per second
        """
        # Detection
        self.detector = PersonDetector()
        self.rule_engine = AlertRuleEngine()
        
        # Video buffer
        self.video_buffer = VideoBuffer(video_buffer_seconds, fps)
        
        # Alert system
        self.alert_sender = AlertSender()
        self.alert_logger = AlertLogger()
        self.location_mapper = LocationMapper()
        
        # Alert throttling (prevent spam)
        self.last_alert_time = {}
        self.alert_cooldown = 10  # Seconds between alerts from same camera
        
        # Statistics
        self.alerts_triggered = 0
        self.alerts_sent = 0
    
    def process_frame(self, frame: np.ndarray, camera_id: str,
                     latitude: float, longitude: float) -> Dict:
        """
        Process frame and check for alert condition
        
        Args:
            frame: Video frame
            camera_id: Camera ID
            latitude: Camera latitude
            longitude: Camera longitude
        
        Returns:
            Dictionary with processing results
        """
        results = {
            'camera_id': camera_id,
            'timestamp': datetime.now().isoformat(),
            'alert_triggered': False,
            'male_count': 0,
            'female_count': 0,
            'detections': []
        }
        
        try:
            # Add frame to buffer
            self.video_buffer.add_frame(frame, datetime.now())
            
            # Detect persons
            detections, _ = self.detector.detect_persons(frame)
            results['detections'] = len(detections)
            
            # Get gender counts
            male_count, female_count = self.detector.get_gender_counts(detections)
            results['male_count'] = male_count
            results['female_count'] = female_count
            
            # Get gender centers
            male_centers, female_centers = self.detector.get_gender_centers(detections)
            
            # Check alert condition
            alert_condition = self.rule_engine.check_alert_condition(
                male_count, female_count, male_centers, female_centers
            )
            
            if alert_condition:
                results['alert_triggered'] = True
                self._handle_alert(
                    frame, camera_id, latitude, longitude,
                    male_count, female_count, detections
                )
            
            return results
        
        except Exception as e:
            print(f"❌ Error processing frame from {camera_id}: {e}")
            results['error'] = str(e)
            return results
    
    def _handle_alert(self, frame: np.ndarray, camera_id: str,
                     latitude: float, longitude: float,
                     male_count: int, female_count: int,
                     detections: list):
        """
        Handle triggered alert
        
        Args:
            frame: Current frame
            camera_id: Camera ID
            latitude: Camera latitude
            longitude: Camera longitude
            male_count: Number of males
            female_count: Number of females
            detections: Detection list
        """
        # Check cooldown
        current_time = time.time()
        if camera_id in self.last_alert_time:
            if current_time - self.last_alert_time[camera_id] < self.alert_cooldown:
                return  # Still in cooldown
        
        self.last_alert_time[camera_id] = current_time
        self.alerts_triggered += 1
        
        print(f"🚨 ALERT from {camera_id}: {female_count} Female(s), {male_count} Male(s)")
        
        # Play beep sound (Windows)
        try:
            winsound.Beep(1000, 500)  # 1000 Hz, 500ms
        except:
            pass  # Might not work on non-Windows
        
        # Save snapshot
        snapshot_path = self._save_snapshot(frame, camera_id)
        
        # Save video clip (last 30 seconds)
        video_path = self._save_video_clip(camera_id)
        
        # Add to location mapper
        self.location_mapper.add_alert(
            latitude, longitude,
            severity=self._calculate_severity(male_count, female_count),
            details={
                'camera_id': camera_id,
                'male_count': male_count,
                'female_count': female_count,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        # Send alerts across channels
        send_results = self.alert_sender.send_multi_channel_alert(
            camera_id, latitude, longitude, male_count, female_count,
            snapshot_path, video_path
        )
        
        self.alerts_sent += 1 if send_results['telegram_alert'] else 0
        
        # Log alert
        self.alert_logger.log_alert({
            'camera_id': camera_id,
            'latitude': latitude,
            'longitude': longitude,
            'male_count': male_count,
            'female_count': female_count,
            'severity': self._calculate_severity(male_count, female_count),
            'image_path': snapshot_path,
            'video_path': video_path,
            'timestamp': datetime.now().isoformat()
        })
    
    def _save_snapshot(self, frame: np.ndarray, camera_id: str) -> str:
        """
        Save snapshot of alert
        
        Args:
            frame: Video frame
            camera_id: Camera ID
        
        Returns:
            Path to saved snapshot
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{camera_id}_{timestamp}.jpg"
            filepath = Path(SNAPSHOT_DIR) / filename
            
            cv2.imwrite(str(filepath), frame)
            print(f"✅ Snapshot saved: {filepath}")
            
            return str(filepath)
        
        except Exception as e:
            print(f"❌ Error saving snapshot: {e}")
            return ""
    
    def _save_video_clip(self, camera_id: str) -> str:
        """
        Save video clip from buffer
        
        Args:
            camera_id: Camera ID
        
        Returns:
            Path to saved video
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{camera_id}_{timestamp}.mp4"
            filepath = Path(VIDEO_DIR) / filename
            
            success = self.video_buffer.save_video(str(filepath), fourcc_code='mp4v')
            
            if success:
                print(f"✅ Video clip saved: {filepath}")
                return str(filepath)
            
            return ""
        
        except Exception as e:
            print(f"❌ Error saving video: {e}")
            return ""
    
    def _calculate_severity(self, male_count: int, female_count: int) -> str:
        """
        Calculate alert severity based on gender counts
        
        Args:
            male_count: Number of males
            female_count: Number of females
        
        Returns:
            Severity level: 'low', 'medium', 'high'
        """
        ratio = male_count / max(female_count, 1)
        
        if ratio >= 3:
            return 'high'
        elif ratio >= 2:
            return 'medium'
        else:
            return 'low'
    
    def get_statistics(self) -> Dict:
        """Get alert statistics"""
        return {
            'total_alerts_triggered': self.alerts_triggered,
            'total_alerts_sent': self.alerts_sent,
            'buffer_status': self.video_buffer.get_info(),
            'location_stats': self.location_mapper.get_alert_statistics(),
            'hotspots': self.location_mapper.get_hotspots()
        }
    
    def generate_reports(self):
        """Generate heatmap and logs"""
        print("📊 Generating reports...")
        
        # Generate heatmap
        self.location_mapper.create_heatmap('heatmap.html')
        self.location_mapper.create_detailed_map('detailed_map.html')
        self.location_mapper.export_alerts_json('alerts_data.json')
        
        print("✅ Reports generated")
