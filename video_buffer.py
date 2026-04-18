"""
Circular Video Buffer for storing last N seconds of video
"""
from collections import deque
from datetime import datetime
import cv2
import numpy as np
from config import BUFFER_SIZE, FPS


class VideoBuffer:
    """
    Circular buffer to store video frames
    Maintains last N seconds of video efficiently
    """
    
    def __init__(self, buffer_seconds: int = 30, fps: int = 30):
        """
        Initialize video buffer
        
        Args:
            buffer_seconds: Number of seconds to buffer
            fps: Frames per second
        """
        self.buffer_size = buffer_seconds * fps
        self.frames = deque(maxlen=self.buffer_size)
        self.fps = fps
        self.metadata = deque(maxlen=self.buffer_size)
    
    def add_frame(self, frame: np.ndarray, timestamp: datetime = None):
        """
        Add frame to buffer
        Oldest frame is automatically discarded when buffer is full
        
        Args:
            frame: Video frame (numpy array)
            timestamp: Frame timestamp (optional)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.frames.append(frame.copy())
        self.metadata.append({
            'timestamp': timestamp,
            'shape': frame.shape
        })
    
    def save_video(self, output_path: str, fourcc_code: str = 'mp4v'):
        """
        Save buffered frames to video file
        
        Args:
            output_path: Path to save video
            fourcc_code: Codec code (default: mp4v for .mp4)
        
        Returns:
            bool: True if successful, False otherwise
        """
        if len(self.frames) == 0:
            print("⚠️ Buffer is empty, cannot save video")
            return False
        
        try:
            # Get video properties
            first_frame = self.frames[0]
            height, width = first_frame.shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
            
            # Write all frames
            for frame in self.frames:
                # Ensure frame is proper size
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                out.write(frame)
            
            out.release()
            print(f"✅ Video saved: {output_path}")
            return True
        
        except Exception as e:
            print(f"❌ Error saving video: {e}")
            return False
    
    def get_snapshot(self) -> np.ndarray:
        """
        Get latest frame as snapshot
        
        Returns:
            numpy array of latest frame, or None if buffer is empty
        """
        if len(self.frames) > 0:
            return self.frames[-1].copy()
        return None
    
    def get_buffer_duration(self) -> float:
        """Get current buffer duration in seconds"""
        return len(self.frames) / self.fps
    
    def clear(self):
        """Clear the buffer"""
        self.frames.clear()
        self.metadata.clear()
    
    def is_full(self) -> bool:
        """Check if buffer is at maximum capacity"""
        return len(self.frames) == self.buffer_size
    
    def get_info(self) -> dict:
        """Get buffer information"""
        return {
            'current_frames': len(self.frames),
            'max_frames': self.buffer_size,
            'duration_seconds': self.get_buffer_duration(),
            'is_full': self.is_full(),
            'fps': self.fps
        }
