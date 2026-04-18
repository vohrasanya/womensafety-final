"""
Multi-Camera Safety Detection System - Main Orchestrator
Complete system that integrates all components
"""
import cv2
import time
import numpy as np
from typing import Dict
import threading
from config import CAMERAS, NUM_THREADS
from camera_handler import MultiCameraManager
from alert_manager import AlertManager


class SafetySystemOrchestrator:
    """
    Main orchestrator for multi-camera safety system
    """
    
    def __init__(self):
        print("🚀 Initializing Multi-Camera Safety Detection System...")
        
        self.camera_manager = MultiCameraManager()
        self.alert_manager = AlertManager()
        
        self.is_running = False
        self.process_threads = []
        
        print("✅ System initialized successfully!")
    
    def setup_cameras(self) -> bool:
        """
        Add all configured cameras to the manager
        
        Returns:
            True if all cameras added successfully
        """
        print(f"📷 Setting up {len(CAMERAS)} camera(s)...")
        
        for camera_config in CAMERAS:
            if camera_config.enabled:
                self.camera_manager.add_camera(camera_config)
            else:
                print(f"⚠️ Camera {camera_config.camera_id} disabled (skipped)")
        
        return len(self.camera_manager.cameras) > 0
    
    def start_system(self) -> bool:
        """
        Start the entire system
        
        Returns:
            True if system started successfully
        """
        print("▶️ Starting system...")
        
        # Start all cameras
        camera_results = self.camera_manager.start_all_cameras()
        
        if not any(camera_results.values()):
            print("❌ No cameras started successfully!")
            return False
        
        self.is_running = True
        
        # Start processing threads for each camera
        for camera_id in self.camera_manager.cameras.keys():
            thread = threading.Thread(
                target=self._process_camera_stream,
                args=(camera_id,),
                daemon=True
            )
            thread.start()
            self.process_threads.append(thread)
        
        print("✅ System started! Processing feeds...")
        return True
    
    def _process_camera_stream(self, camera_id: str):
        """
        Process stream from a single camera
        
        Args:
            camera_id: Camera ID
        """
        print(f"🎬 Processing thread started for {camera_id}")
        
        frame_count = 0
        fps_timer = time.time()
        fps = 0
        
        while self.is_running:
            try:
                # Get frame from camera
                frame = self.camera_manager.get_frame(camera_id)
                
                if frame is None:
                    time.sleep(0.01)  # Avoid busy waiting
                    continue
                
                # Get camera location
                location = self.camera_manager.get_camera_location(camera_id)
                
                # Process frame
                results = self.alert_manager.process_frame(
                    frame,
                    camera_id,
                    location['latitude'],
                    location['longitude']
                )
                
                # Draw frame (optional visualization)
                frame_with_info = self._annotate_frame(frame, results)
                
                # Display frame (optional)
                # cv2.imshow(f'{camera_id}', frame_with_info)
                
                frame_count += 1
                
                # Calculate FPS
                elapsed = time.time() - fps_timer
                if elapsed > 1:
                    fps = frame_count / elapsed
                    print(f"📊 {camera_id}: {fps:.1f} FPS - M:{results['male_count']} F:{results['female_count']}", end='\r')
                    frame_count = 0
                    fps_timer = time.time()
                
                # Check for exit (press 'q' in any window)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False
                    break
            
            except Exception as e:
                print(f"❌ Error processing {camera_id}: {e}")
                time.sleep(0.1)
    
    def _annotate_frame(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        Annotate frame with detection info
        
        Args:
            frame: Input frame
            results: Detection results
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Add text information
        info_text = [
            f"Camera: {results['camera_id']}",
            f"Males: {results['male_count']}",
            f"Females: {results['female_count']}",
            f"Detections: {results['detections']}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(
                annotated, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            y_offset += 30
        
        # Add alert indicator
        if results['alert_triggered']:
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], annotated.shape[0]),
                         (0, 0, 255), 5)
            cv2.putText(
                annotated, "🚨 ALERT 🚨", (10, annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3
            )
        
        return annotated
    
    def stop_system(self):
        """Stop the system gracefully"""
        print("\n⏹️ Stopping system...")
        
        self.is_running = False
        
        # Wait for threads to finish
        for thread in self.process_threads:
            thread.join(timeout=5)
        
        # Stop all cameras
        self.camera_manager.stop_all_cameras()
        
        # Generate reports
        print("📊 Generating reports...")
        self.alert_manager.generate_reports()
        
        # Print statistics
        stats = self.alert_manager.get_statistics()
        print("\n" + "="*60)
        print("📈 SYSTEM STATISTICS")
        print("="*60)
        print(f"Total Alerts Triggered: {stats['total_alerts_triggered']}")
        print(f"Total Alerts Sent: {stats['total_alerts_sent']}")
        print(f"Location Stats: {stats['location_stats']}")
        print("="*60)
        
        print("✅ System stopped successfully!")
        print("📁 Generated files:")
        print("   - heatmap.html")
        print("   - detailed_map.html")
        print("   - alerts_data.json")
        print("   - alerts/alerts.log")
        print("   - alerts/snapshots/")
        print("   - alerts/videos/")
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'is_running': self.is_running,
            'cameras': self.camera_manager.get_camera_info(),
            'statistics': self.alert_manager.get_statistics(),
            'threads_alive': sum(1 for t in self.process_threads if t.is_alive())
        }


def main():
    """
    Main entry point
    """
    # Initialize orchestrator
    orchestrator = SafetySystemOrchestrator()
    
    # Setup cameras
    if not orchestrator.setup_cameras():
        print("❌ Failed to setup cameras")
        return
    
    # Start system
    if not orchestrator.start_system():
        print("❌ Failed to start system")
        return
    
    try:
        # Keep system running
        print("\n✅ System running. Press Ctrl+C to stop...")
        while orchestrator.is_running:
            time.sleep(1)
            
            # Optionally print status periodically
            # status = orchestrator.get_system_status()
            # print(f"Active threads: {status['threads_alive']}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Keyboard interrupt received")
    
    finally:
        orchestrator.stop_system()


if __name__ == "__main__":
    main()
