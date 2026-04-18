"""
Multi-Camera Handler - Manage multiple camera streams concurrently
"""
import cv2
import threading
import time
from typing import Dict, Optional, Callable
from queue import Queue
import numpy as np
from config import CameraConfig, FRAME_RESIZE, SKIP_FRAMES


class CameraStream:
    """
    Handle individual camera stream
    """
    
    def __init__(self, camera_config: CameraConfig, frame_callback: Callable = None):
        """
        Initialize camera stream
        
        Args:
            camera_config: Camera configuration
            frame_callback: Optional callback function for each frame
        """
        self.config = camera_config
        self.cap = None
        self.frame_queue = Queue(maxsize=5)  # Keep last 5 frames
        self.is_running = False
        self.thread = None
        self.frame_callback = frame_callback
        self.last_frame = None
        self.frame_count = 0
        self.fps = 30
    
    def start(self) -> bool:
        """
        Start camera stream reading
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Open camera source
            if isinstance(self.config.source, int):
                # Webcam
                self.cap = cv2.VideoCapture(self.config.source)
            else:
                # Video file or IP camera
                self.cap = cv2.VideoCapture(self.config.source)
            
            if not self.cap.isOpened():
                print(f"❌ Failed to open camera {self.config.camera_id}: {self.config.source}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_RESIZE[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_RESIZE[1])
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            
            # Start reading thread
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            
            print(f"✅ Camera {self.config.camera_id} started")
            return True
        
        except Exception as e:
            print(f"❌ Error starting camera {self.config.camera_id}: {e}")
            return False
    
    def _read_loop(self):
        """Internal loop to read frames"""
        frame_skip_counter = 0
        
        while self.is_running and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    print(f"⚠️ Failed to read frame from {self.config.camera_id}")
                    break
                
                # Skip frames for performance
                if frame_skip_counter % SKIP_FRAMES != 0:
                    frame_skip_counter += 1
                    continue
                
                frame_skip_counter += 1
                
                # Resize for faster processing
                frame = cv2.resize(frame, FRAME_RESIZE)
                
                self.last_frame = frame
                self.frame_count += 1
                
                # Put frame in queue (discard if full)
                try:
                    self.frame_queue.put_nowait(frame)
                except:
                    pass  # Queue full, drop frame
                
                # Call callback if provided
                if self.frame_callback:
                    self.frame_callback(self._get_frame_data())
            
            except Exception as e:
                print(f"⚠️ Error in camera {self.config.camera_id} read loop: {e}")
                break
        
        self.stop()
    
    def _get_frame_data(self) -> Dict:
        """Get frame data for callback"""
        return {
            'camera_id': self.config.camera_id,
            'frame': self.last_frame.copy() if self.last_frame is not None else None,
            'timestamp': time.time(),
            'frame_count': self.frame_count,
            'latitude': self.config.latitude,
            'longitude': self.config.longitude
        }
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get latest frame from camera
        
        Returns:
            Frame as numpy array, or None if no frame available
        """
        try:
            return self.frame_queue.get_nowait()
        except:
            return self.last_frame
    
    def get_camera_info(self) -> Dict:
        """Get camera information"""
        return {
            'camera_id': self.config.camera_id,
            'name': self.config.name,
            'latitude': self.config.latitude,
            'longitude': self.config.longitude,
            'is_running': self.is_running,
            'frame_count': self.frame_count,
            'enabled': self.config.enabled
        }
    
    def stop(self):
        """Stop camera stream"""
        self.is_running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        print(f"✅ Camera {self.config.camera_id} stopped")


class MultiCameraManager:
    """
    Manage multiple camera streams
    """
    
    def __init__(self):
        self.cameras: Dict[str, CameraStream] = {}
        self.camera_configs: Dict[str, CameraConfig] = {}
    
    def add_camera(self, camera_config: CameraConfig, frame_callback: Callable = None) -> bool:
        """
        Add a camera
        
        Args:
            camera_config: Camera configuration
            frame_callback: Optional callback for each frame
        
        Returns:
            True if added successfully
        """
        try:
            camera = CameraStream(camera_config, frame_callback)
            self.cameras[camera_config.camera_id] = camera
            self.camera_configs[camera_config.camera_id] = camera_config
            print(f"✅ Camera {camera_config.camera_id} added to manager")
            return True
        except Exception as e:
            print(f"❌ Error adding camera: {e}")
            return False
    
    def start_all_cameras(self) -> Dict[str, bool]:
        """
        Start all cameras
        
        Returns:
            Dictionary with status for each camera
        """
        results = {}
        for camera_id, camera in self.cameras.items():
            results[camera_id] = camera.start()
        
        print(f"✅ Started {sum(results.values())}/{len(results)} cameras")
        return results
    
    def stop_all_cameras(self):
        """Stop all cameras"""
        for camera in self.cameras.values():
            camera.stop()
        print("✅ All cameras stopped")
    
    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """
        Get frame from specific camera
        
        Args:
            camera_id: Camera ID
        
        Returns:
            Frame as numpy array, or None
        """
        if camera_id in self.cameras:
            return self.cameras[camera_id].get_frame()
        return None
    
    def get_frames_all(self) -> Dict[str, Optional[np.ndarray]]:
        """
        Get current frame from all cameras
        
        Returns:
            Dictionary with camera_id -> frame mapping
        """
        frames = {}
        for camera_id, camera in self.cameras.items():
            frame = camera.get_frame()
            if frame is not None:
                frames[camera_id] = frame
        
        return frames
    
    def get_camera_info(self, camera_id: str = None) -> Dict:
        """
        Get camera information
        
        Args:
            camera_id: Specific camera ID, or None for all
        
        Returns:
            Camera information dictionary
        """
        if camera_id:
            if camera_id in self.cameras:
                return self.cameras[camera_id].get_camera_info()
            return {}
        
        # Return all cameras
        return {cid: cam.get_camera_info() for cid, cam in self.cameras.items()}
    
    def get_camera_location(self, camera_id: str) -> Dict:
        """
        Get camera GPS location
        
        Args:
            camera_id: Camera ID
        
        Returns:
            Dictionary with latitude and longitude
        """
        if camera_id in self.camera_configs:
            config = self.camera_configs[camera_id]
            return {
                'latitude': config.latitude,
                'longitude': config.longitude,
                'camera_id': camera_id,
                'name': config.name
            }
        return {}
    
    def is_camera_running(self, camera_id: str) -> bool:
        """Check if camera is running"""
        if camera_id in self.cameras:
            return self.cameras[camera_id].is_running
        return False
    
    def get_total_frames(self, camera_id: str) -> int:
        """Get total frames processed from camera"""
        if camera_id in self.cameras:
            return self.cameras[camera_id].frame_count
        return 0
