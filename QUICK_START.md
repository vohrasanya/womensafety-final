## 🚀 QUICK START GUIDE

### 5-Minute Setup

#### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 2: Get Telegram Bot Token
1. Message @BotFather on Telegram
2. Type `/newbot`
3. Copy your token

#### Step 3: Configuration
```bash
# Copy template
copy .env.example .env

# Edit .env
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

Get your chat ID:
```
https://api.telegram.org/bot{TOKEN}/getUpdates
```

#### Step 4: Edit Camera Config
In `config.py`, update:
```python
CAMERAS = [
    CameraConfig(
        camera_id="CAM_01",
        source=0,  # 0 = webcam
        latitude=40.7128,
        longitude=-74.0060,
        name="Main Entrance"
    ),
]
```

#### Step 5: Run System
```bash
python multi_camera_main.py
```

#### Step 6: View Dashboard (Optional)
```bash
streamlit run dashboard.py
```
Visit: http://localhost:8501

---

### Testing on Single Webcam

If you just have one webcam, simplest setup:

1. **config.py:**
```python
CAMERAS = [
    CameraConfig(
        camera_id="CAM_01",
        source=0,  # Webcam
        latitude=40.7128,
        longitude=-74.0060,
        name="Test Camera"
    ),
]
```

2. **.env:**
```
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_id
```

3. **Run:**
```bash
python multi_camera_main.py
```

---

### Expected Output

```
🚀 Initializing Multi-Camera Safety Detection System...
✅ System initialized successfully!
📷 Setting up 1 camera(s)...
✅ Camera CAM_01 added to manager
▶️ Starting system...
✅ Camera CAM_01 started
✅ System started! Processing feeds...
🎬 Processing thread started for CAM_01
📊 CAM_01: 28.5 FPS - M:0 F:0
```

When alert triggers:
```
🚨 ALERT from CAM_01: 1 Female(s), 2 Male(s)
✅ Snapshot saved: alerts/snapshots/CAM_01_20240415_143025.jpg
✅ Video clip saved: alerts/videos/CAM_01_20240415_143025.mp4
✅ Telegram alert sent from CAM_01
✅ Alert logged to alerts/alerts.log
```

---

### Generated Files

After running, you'll see:
- `heatmap.html` - Incident heatmap (open in browser)
- `detailed_map.html` - Alert map with markers
- `alerts_data.json` - Structured alert data
- `alerts/` folder with snapshots and videos

---

### Troubleshooting Telegram

**Test Telegram Bot:**
```python
import requests

token = "YOUR_TOKEN"
chat_id = "YOUR_CHAT_ID"
message = "Test message"

url = f"https://api.telegram.org/bot{token}/sendMessage"
data = {"chat_id": chat_id, "text": message}

response = requests.post(url, json=data)
print(response.status_code)  # Should be 200
```

**Send Image:**
```python
url = f"https://api.telegram.org/bot{token}/sendPhoto"
files = {"photo": open("test.jpg", "rb")}
data = {"chat_id": chat_id, "caption": "Test photo"}

response = requests.post(url, data=data, files=files)
print(response.status_code)  # Should be 200
```

---

### What Happens Next?

1. **Real-Time Processing**: Your camera(s) start streaming
2. **Person Detection**: YOLO detects people
3. **Gender Classification**: Model classifies gender
4. **Local Alert Checking**: Checks if alert condition met (1F + 2M in radius)
5. **Alert Trigger**: If condition met:
   - Saves snapshot
   - Saves last 30 seconds video
   - Sends Telegram/Email alert
   - Logs event to JSON and text file
   - Maps location on heatmap

---

### Next Steps

- Modify alert rules in `config.py`
- Add more cameras
- Train custom gender model
- Integrate with other systems
- Deploy on edge device (Raspberry Pi, Jetson)

Get started now! 🚀
