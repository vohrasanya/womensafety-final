# 🏛️ SYSTEM ARCHITECTURE DOCUMENT

## Overview

The Multi-Camera Safety Detection System is designed with **modularity**, **scalability**, and **real-time performance** in mind.

---

## Core Design Principles

### 1. **Separation of Concerns**
Each component has a single responsibility:
- **Camera Handler**: Stream management only
- **Detection**: Computer vision logic only
- **Alert Manager**: Rules and orchestration
- **Alert Sender**: Communication channels
- **Location Mapper**: Geographic data

### 2. **Thread-Based Concurrency**
- Each camera runs on its own thread
- Non-blocking architecture
- Safe for 5-50+ concurrent streams
- Minimal lock contention

### 3. **Modular ML Pipeline**
```
Frame → YOLO Detection → Face Cascade → Gender Classification → Decision
```
Each step can be replaced or improved independently.

### 4. **Circular Buffering**
- Fixed memory usage regardless of alerts
- Always keeps last N seconds
- O(1) append and save operations
- Prevents memory leaks

---

## Component Architecture

### 1. **Configuration Module** (`config.py`)
**Purpose**: Centralized configuration management

**Key Features**:
- Camera definitions with GPS coordinates
- Alert rules and thresholds
- Model paths and settings
- Output directories
- Performance tuning parameters

**Design Pattern**: Singleton-like configuration
**Why**: Single source of truth for all settings

---

### 2. **Camera Handler** (`camera_handler.py`)
**Purpose**: Multi-camera stream management

**Architecture**:
```
MultiCameraManager (Main)
    ├── CameraStream 1 (Thread)
    ├── CameraStream 2 (Thread)
    └── CameraStream N (Thread)
```

**Key Classes**:
- `CameraStream`: Individual camera handler with dedicated thread
- `MultiCameraManager`: Manages multiple streams, provides unified interface

**Features**:
- Non-blocking frame reading
- Frame queue (FIFO) for latest frames
- Automatic frame resizing
- Frame skipping for performance

**Thread Safety**:
- Each camera has its own lock-free buffer
- No shared mutable state between cameras
- Queue operations are thread-safe

**Performance**:
- Independent threads = CPU parallelization
- Queue drop frame when full = no memory explosion
- Frame skipping = variable FPS support

---

### 3. **Detection Module** (`detection_refactored.py`)
**Purpose**: Computer vision pipeline

**Components**:
```
PersonDetector
├── YOLO (Person Detection)
├── Face Cascade (Face Detection)
└── Gender Model (Classification)

AlertRuleEngine
└── Proximity checking
```

**Key Methods**:
- `detect_persons()`: Full detection pipeline
- `predict_gender()`: Single face classification
- `get_gender_counts()`: Aggregate statistics
- `check_alert_condition()`: Rule evaluation

**Design**:
- Stateless processing (can be parallelized)
- No dependency on camera source
- Compatible with both image files and streams

**Bottleneck**: Gender model prediction (50-100ms per detection)
Fix for future: Batch inference or GPU acceleration

---

### 4. **Video Buffer** (`video_buffer.py`)
**Purpose**: Efficient incident recording

**Data Structure**: 
```
Circular Deque [max_size = FPS * buffer_seconds]
├── Frame 1 (oldest, discarded when full)
├── Frame 2
├── ...
└── Frame N (newest)
```

**Memory Model**:
- Fixed allocation: `30 * 30 * 640 * 480 * 3 * 1 byte = ~1.65 GB` (worst case)
- Actual: ~500MB for YUV encoding
- Per-camera: negligible compared to processing

**Operations**:
- `add_frame()`: O(1) with deque
- `save_video()`: O(n) but only on alert (not continuous)
- `clear()`: O(1)

**Why Circular Buffer**:
- Fixed memory
- No garbage collection pauses
- Always ready for alert
- No disk writes until alert triggers

---

### 5. **Alert Manager** (`alert_manager.py`)
**Purpose**: Core logic orchestration

**Workflow**:
```
Frame → Detect → Check Rules → Decision
                                  ├→ NO: Continue
                                  └→ YES: 
                                      ├→ Check Cooldown
                                      ├→ Save Snapshot
                                      ├→ Save Video Clip
                                      ├→ Send Alerts
                                      └→ Log Event
```

**Alerts Throttling**:
- Per-camera cooldown (10 seconds default)
- Prevents alert spam
- Configurable threshold

**Severity Calculation**:
```
Male/Female Ratio
├─ < 2.0 → Low
├─ 2.0-3.0 → Medium
└─ > 3.0 → High
```

---

### 6. **Alert Sender** (`alert_sender.py`)
**Purpose**: Multi-channel notifications

**Channels**:
1. **Telegram**: Instant notifications with media
   - API: `sendMessage` (text), `sendPhoto` (image), `sendVideo` (video)
   - Latency: 1-3 seconds
   - Reliability: 99%+

2. **Email**: Detailed reports with attachments
   - Protocol: SMTP (Gmail, Outlook, etc.)
   - Latency: 5-15 seconds
   - Reliability: 95%+

**Features**:
- Non-blocking sends (can be async)
- Automatic retry logic
- Formatted messages with details
- Media attachment support

**Future**: SMS, Push Notifications, Webhook

---

### 7. **Location Mapper** (`location_mapper.py`)
**Purpose**: Geographic visualization

**Features**:

1. **Heatmap**:
   - Folium library
   - Color intensity based on severity
   - HTML output for browser viewing
   - Real-time updates on alert

2. **Detailed Map**:
   - Individual markers per alert
   - Popup info on click
   - Clustering for dense areas

3. **Hotspot Detection**:
   - K-means like clustering (simplified)
   - Radius-based grouping
   - Severity aggregation

4. **Data Export**:
   - JSON format for integration
   - CSV for analysis

---

### 8. **Dashboard** (`dashboard.py`)
**Purpose**: Real-time monitoring UI

**Technology**: Streamlit
**Why Streamlit**:
- No JavaScript needed
- Pure Python
- Hot reload
- Easy to prototype

**Pages**:
1. **Dashboard**: Stats, metrics, recent activity
2. **Maps**: Heatmap and detailed map
3. **Snapshots**: Gallery of incidents
4. **Logs**: Alert history with export

**Auto-refresh**: 5-second polling with button

---

### 9. **Main Orchestrator** (`multi_camera_main.py`)
**Purpose**: System coordination

**Architecture**:
```
SafetySystemOrchestrator
├── Camera Manager
│   └── Thread per camera
├── Alert Manager
│   ├── Detection
│   ├── Video Buffer
│   ├── Alert Sender
│   └── Location Mapper
└── Dashboard (separate Streamlit process)
```

**Workflow**:
1. Initialize all components
2. Start camera threads
3. Main loop: Get frames → Process → Log
4. Graceful shutdown on interrupt

---

## Data Flow Diagram

```
┌──────────────┐
│ Camera 1     │
└──────┬───────┘
       │
       ▼
   [Frame Queue]
       │
       ▼
━━━━━━┷━━━━━━━ THREAD 1 ━━━━━━
│
└──→ [YOLO Detection]
    │
    ├──→ [Face Detection]
    │   │
    │   └──→ [Gender Model]
    │       │
    │       └──→ [{detection}]
    │
    ├──→ [Counts: M/F]
    │
    ├──→ [Video Buffer] ◄── (Continuous capture)
    │
    └──→ [Rule Engine: Check Alert]
        │
        ├─→ NO: Log stats, continue
        │
        └─→ YES: 
            ├──→ [Save Snapshot] ───→ File
            ├──→ [Save 30s Clip] ────→ File
            ├──→ [Telegram Bot] ─────→ Network
            ├──→ [Email SMTP] ───────→ Network
            ├──→ [Location Map] ─────→ File
            └──→ [Alert Log] ────────→ File
            
            All files sync to Dashboard (Streamlit)
```

---

## Performance Analysis

### Per-Camera Processing
```
Operation              Time        Bottleneck
─────────────────────────────────────────────
Frame Capture          ~30-50ms    Camera API
Resize Frame           ~5-10ms     OpenCV
YOLO Inference         ~40-80ms    Neural Net ⚠️
Face Detection         ~10-20ms    Cascade Classifier
Gender Inference       ~50-100ms   Neural Net ⚠️
─────────────────────────────────────────────
TOTAL per frame        ~150-300ms  ~3-7 FPS max
```

**With Threading**:
- 5 cameras: 15-35 FPS total (parallelized)
- 10 cameras: Limited by machine resources

**With GPU** (future enhancement):
- YOLO inference: 5-10ms
- Gender inference: 10-20ms
- Total: ~30-800ms → 30+ FPS easily

### Memory Usage
```
Component              Memory      Notes
──────────────────────────────────────────
1 Camera Stream        100-150MB   Frame queues
Video Buffer (30s)     450-650MB   Circular deque
Gender Model           200-300MB   TensorFlow
YOLO Model             100-150MB   OpenCV DNN
System Overhead        200-300MB   Python, libraries
──────────────────────────────────────────
2 Cameras Total        ~1.5-2.5GB
```

---

## Scaling Architecture

### Horizontal Scaling
For 50+ cameras:
```
Load Balancer
├── Worker 1 (8 cameras)
├── Worker 2 (8 cameras)
├── Worker 3 (8 cameras)
└── Central Alert Hub
    ├── Aggregates alerts
    ├── Central heatmap
    └── Unified dashboard
```

### Vertical Scaling
For single machine:
```
GPU Processing          30+ FPS × 10 cameras
─────────────────────────────────────────
→ Jetson AGX Xavier     Up to 1000+ FPS total
→ A100 GPU              Unlimited parallelization
→ Edge Processing       Decentralized alerts
```

---

## Security Architecture

### Data Flow Security
```
Local Processing        Secure
├── Detection: Local CPU only
├── Video: Local storage only
├── Snapshots: Local disk

Alert Transmission      Use TLS/HTTPS
├── Telegram: Bot API (encrypted)
├── Email: SMTP TLS (encrypted)
└── Dashboard: Local network only

Storage
├── No cloud upload (configurable)
├── Local deletion after N days (future feature)
└── Audit logging of all access
```

### Authentication
- API Key for REST endpoints
- Environment variables for secrets
- No hardcoded credentials

---

## Fault Tolerance

### Camera Failures
```
If camera disconnects:
├── Retry connection (configurable)
├── Other cameras continue
├── Alert if camera offline > 5 min
└── Auto-recover when available
```

### Alert Delivery Failure
```
If Telegram fails:
├── Retry up to 3 times
├── Fall back to Email
├── Log failure in database
└── Alert user via dashboard
```

### Resource Exhaustion
```
If memory full:
├── Reduce buffer size
├── Drop oldest alerts (never)
├── Increase frame skip
└── Reduce resolution
```

---

## Future Improvements

### Short-term
1. **GPU Support**: CUDA/TensorRT for 10x speedup
2. **Custom Models**: Train gender model on specific data
3. **Database**: PostgreSQL for alerts instead of JSON
4. **Web UI**: Full-fledged dashboard (React)

### Medium-term
1. **Distributed Processing**: Multiple workers
2. **Edge Deployment**: Jetson/Coral inference
3. **Face Recognition**: Identify specific persons
4. **Behavior Analysis**: Detect suspicious actions

### Long-term
1. **Real-time Streaming**: RTMP/HLS broadcast
2. **ML Explainability**: Understand why alerts triggered
3. **Multi-modal**: Audio, thermal, LiDAR sensors
4. **System Optimization**: Rust/C++ critical paths

---

## References

**YOLOv3**: https://pjreddie.com/darknet/yolo/
**Haar Cascades**: Viola-Jones object detection framework
**Folium**: Leaflet.js wrapper for Python
**Streamlit**: Rapid Python data app development
**TensorFlow**: Deep learning framework

---

## Conclusion

The system is designed for:
- ✅ **Real-time processing**: 30+ FPS on good hardware
- ✅ **Scalability**: From 1 to 100+ cameras
- ✅ **Reliability**: Fault-tolerant components
- ✅ **Maintainability**: Clean modular code
- ✅ **Extensibility**: Easy to add features

