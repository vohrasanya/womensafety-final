"""
Example Usage Scenarios for Multi-Camera Safety System
Demonstrates different use cases and integration patterns
"""

# ==========================================
# Example 1: Basic Multi-Camera Setup
# ==========================================

def example_basic_setup():
    """Start system with default configuration"""
    from multi_camera_main import SafetySystemOrchestrator
    
    # Initialize
    orchestrator = SafetySystemOrchestrator()
    
    # Setup cameras from config
    orchestrator.setup_cameras()
    
    # Start processing
    orchestrator.start_system()
    
    # Keep running
    try:
        while orchestrator.is_running:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        orchestrator.stop_system()


# ==========================================
# Example 2: Custom Camera Setup
# ==========================================

def example_custom_cameras():
    """Add cameras programmatically"""
    from multi_camera_main import SafetySystemOrchestrator
    from config import CameraConfig
    
    orchestrator = SafetySystemOrchestrator()
    
    # Add cameras manually
    camera1 = CameraConfig(
        camera_id="MAIN_GATE",
        source=0,  # Webcam
        latitude=40.7128,
        longitude=-74.0060,
        name="Main Gate"
    )
    
    camera2 = CameraConfig(
        camera_id="BACK_GATE",
        source="http://192.168.1.100:8080/video",  # IP camera
        latitude=40.7130,
        longitude=-74.0065,
        name="Back Gate"
    )
    
    orchestrator.camera_manager.add_camera(camera1)
    orchestrator.camera_manager.add_camera(camera2)
    
    orchestrator.start_system()


# ==========================================
# Example 3: Get System Status
# ==========================================

def example_system_status():
    """Monitor system status"""
    from multi_camera_main import SafetySystemOrchestrator
    import time
    
    orchestrator = SafetySystemOrchestrator()
    orchestrator.setup_cameras()
    orchestrator.start_system()
    
    # Print status every 5 seconds
    try:
        while orchestrator.is_running:
            status = orchestrator.get_system_status()
            
            print("\n" + "="*60)
            print("SYSTEM STATUS")
            print("="*60)
            print(f"Running: {status['is_running']}")
            print(f"Active Threads: {status['threads_alive']}")
            print(f"Alert Stats: {status['statistics']['total_alerts_triggered']} total")
            
            for cam_id, cam_info in status['cameras'].items():
                print(f"  {cam_id}: {'✅ Running' if cam_info['is_running'] else '❌ Stopped'}")
                print(f"    Frames: {cam_info['frame_count']}")
            
            time.sleep(5)
    
    except KeyboardInterrupt:
        orchestrator.stop_system()


# ==========================================
# Example 4: Custom Alert Rules
# ==========================================

def example_custom_alert_rules():
    """Modify alert rules dynamically"""
    from alert_manager import AlertRuleEngine
    
    # Create engine
    engine = AlertRuleEngine()
    
    # Modify thresholds
    engine.min_females = 1
    engine.min_males = 3
    engine.alert_radius = 150
    
    # Test condition
    male_count = 3
    female_count = 1
    male_centers = [(100, 100), (150, 120), (200, 110)]
    female_centers = [(110, 105)]
    
    alert = engine.check_alert_condition(
        male_count, female_count, male_centers, female_centers
    )
    
    print(f"Alert Triggered: {alert}")


# ==========================================
# Example 5: Direct Frame Processing
# ==========================================

def example_direct_frame_processing():
    """Process frames without full system"""
    import cv2
    from detection_refactored import PersonDetector, AlertRuleEngine
    
    detector = PersonDetector()
    rule_engine = AlertRuleEngine()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect persons
        detections, _ = detector.detect_persons(frame)
        
        # Get counts
        male_count, female_count = detector.get_gender_counts(detections)
        male_centers, female_centers = detector.get_gender_centers(detections)
        
        # Check alert
        alert = rule_engine.check_alert_condition(
            male_count, female_count, male_centers, female_centers
        )
        
        # Draw bounding boxes
        for detection in detections:
            x, y, w, h = detection['bbox']
            color = (0, 255, 0) if detection['gender'] == 'Female' else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, detection['gender'], (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Alert indicator
        if alert:
            cv2.putText(frame, f"ALERT: {female_count}F {male_count}M",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# ==========================================
# Example 6: Alert History Analysis
# ==========================================

def example_alert_analysis():
    """Analyze alert history"""
    import json
    from datetime import datetime
    
    # Load alerts
    with open('alerts_data.json', 'r') as f:
        alerts = json.load(f)
    
    # Statistics
    print(f"Total Alerts: {len(alerts)}")
    
    # By severity
    high = sum(1 for a in alerts if a.get('severity') == 'high')
    medium = sum(1 for a in alerts if a.get('severity') == 'medium')
    low = sum(1 for a in alerts if a.get('severity') == 'low')
    
    print(f"High Severity: {high}")
    print(f"Medium Severity: {medium}")
    print(f"Low Severity: {low}")
    
    # By camera
    from collections import Counter
    cameras = [a.get('details', {}).get('camera_id', 'Unknown') for a in alerts]
    camera_counts = Counter(cameras)
    
    print(f"\nAlerts by Camera:")
    for cam_id, count in camera_counts.most_common():
        print(f"  {cam_id}: {count}")
    
    # Average m/f ratio
    if alerts:
        avg_males = sum(a.get('details', {}).get('male_count', 0) for a in alerts) / len(alerts)
        avg_females = sum(a.get('details', {}).get('female_count', 0) for a in alerts) / len(alerts)
        
        print(f"\nAverage per Alert:")
        print(f"  Males: {avg_males:.1f}")
        print(f"  Females: {avg_females:.1f}")


# ==========================================
# Example 7: Video Buffer Demo
# ==========================================

def example_video_buffer():
    """Demonstrate video buffer functionality"""
    import cv2
    import time
    from video_buffer import VideoBuffer
    
    buffer = VideoBuffer(buffer_seconds=5, fps=30)
    
    cap = cv2.VideoCapture(0)
    
    print("Recording for 10 seconds...")
    start_time = time.time()
    
    while time.time() - start_time < 10:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add to buffer
        buffer.add_frame(frame)
        
        # Display buffer status
        info = buffer.get_info()
        print(f"Buffer: {info['current_frames']}/{info['max_frames']} "
              f"({info['duration_seconds']:.1f}s)", end='\r')
    
    cap.release()
    
    print("\n\nBuffer Status:")
    print(buffer.get_info())
    
    # Save buffer
    buffer.save_video("buffer_output.mp4")


# ==========================================
# Example 8: Heatmap Generation
# ==========================================

def example_heatmap():
    """Generate heatmaps from alert data"""
    from location_mapper import LocationMapper
    
    mapper = LocationMapper(
        center_lat=40.7128,
        center_lon=-74.0060,
        zoom=14
    )
    
    # Add sample alerts
    locations = [
        (40.7128, -74.0060, 'high'),
        (40.7135, -74.0065, 'medium'),
        (40.7120, -74.0055, 'low'),
        (40.7130, -74.0062, 'medium'),
    ]
    
    for lat, lon, severity in locations:
        mapper.add_alert(
            lat, lon,
            severity=severity,
            details={
                'camera_id': 'TEST_CAM',
                'male_count': 2,
                'female_count': 1
            }
        )
    
    # Generate maps
    mapper.create_heatmap('example_heatmap.html')
    mapper.create_detailed_map('example_detailed_map.html')
    
    print("✅ Heatmaps generated!")
    print(f"Hotspots: {mapper.get_hotspots()}")


# ==========================================
# Example 9: Alert Sending
# ==========================================

def example_alerts():
    """Send test alerts"""
    from alert_sender import AlertSender, AlertLogger
    
    sender = AlertSender()
    
    # Send Telegram alert
    sender.send_telegram_alert(
        camera_id="TEST_CAMERA",
        latitude=40.7128,
        longitude=-74.0060,
        male_count=2,
        female_count=1
    )
    
    # Log alert
    logger = AlertLogger()
    logger.log_alert({
        'camera_id': 'TEST_CAMERA',
        'latitude': 40.7128,
        'longitude': -74.0060,
        'male_count': 2,
        'female_count': 1,
        'severity': 'medium',
        'timestamp': '2024-04-15 14:30:00'
    })


# ==========================================
# Run Examples
# ==========================================

if __name__ == "__main__":
    import sys
    
    examples = {
        '1': ('Basic Setup', example_basic_setup),
        '2': ('Custom Cameras', example_custom_cameras),
        '3': ('System Status', example_system_status),
        '4': ('Custom Alert Rules', example_custom_alert_rules),
        '5': ('Direct Frame Processing', example_direct_frame_processing),
        '6': ('Alert Analysis', example_alert_analysis),
        '7': ('Video Buffer', example_video_buffer),
        '8': ('Heatmap Generation', example_heatmap),
        '9': ('Send Alerts', example_alerts),
    }
    
    print("\n" + "="*60)
    print("🎯 MULTI-CAMERA SYSTEM - EXAMPLE SCENARIOS")
    print("="*60)
    
    for key, (name, _) in examples.items():
        print(f"{key}. {name}")
    
    print("\n0. Exit")
    
    choice = input("\nSelect example (0-9): ").strip()
    
    if choice in examples:
        print(f"\n▶️ Running: {examples[choice][0]}\n")
        examples[choice][1]()
    else:
        print("Invalid choice")
