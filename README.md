# Object Detection and Gender Classification using YOLOv3 and TensorFlow

# 👩‍🦰 Women Safety Detection System 

This project is a **real-time Women Safety Detection System** that detects people using a webcam, classifies their gender (Male/Female), counts them, and raises an **alert when a woman is surrounded by multiple men**.

The system is built using **Computer Vision and Deep Learning**.

---

## 🚀 Features

- 📷 Real-time webcam detection
- 🧍 Person detection using **YOLOv3-Tiny**
- 👨👩 Gender classification using a trained CNN model
- 🔢 Live Male and Female count
- ⚠️ Safety alert when:
  - Only **1 female** is detected
  - **2 or more males** are nearby
- 🔊 Sound alert (beep)
- 🚨 Visual alert banner with red border
- 🛑 Stop using `q` key

---

## 🧠 Working Principle


## Project Structure

gender_1/
│
├── main.py
├── README.md
│
├── model/
│ └── gender_model.h5
│
├── yolov3/
│ ├── yolov3-tiny.cfg
│ ├── yolov3-tiny.weights
│ └── coco.names
│
├── data/ # Used only for training
│ ├── train/
│ └── val/
│
├── venv/
└── requirements.txt


---

## ⚙️ Requirements

- Python **3.10**
- Windows OS (for sound alert)
- Webcam

### Required Libraries


tensorflow
opencv-python
numpy
scipy
pillow


---

## 🧪 Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd gender_1

2️⃣ Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Download YOLOv3-Tiny files

Place the following inside the yolov3/ folder:

yolov3-tiny.cfg

yolov3-tiny.weights

coco.names

(Source: Official Darknet repository)

5️⃣ Ensure trained model exists
model/gender_model.h5


This model was trained using a Kaggle gender classification dataset.

▶️ Run the Project
python main.py

🛑 Stop the Program

Press q in the camera window

OR press Ctrl + C in terminal

⚠️ Alert Condition

An alert is triggered when:

Exactly 1 Female is detected

2 or more Males are detected nearby

On alert:

🔊 Beep sound plays

🚨 Red border appears

⚠ Alert banner is shown

📊 Dataset Used

Gender Classification Dataset

Source: Kaggle
https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset

🎓 Use Cases

Women safety monitoring

Academic / college projects

Computer vision demonstrations

AI ethics discussion

⚠️ Disclaimer

This project is developed only for educational purposes.
It is a risk alert system, not a crime detection or surveillance tool.






- YOLOv3: [YOLO Website](https://pjreddie.com/darknet/yolo/)
- TensorFlow: [TensorFlow Documentation](https://www.tensorflow.org/)
- OpenCV: [OpenCV Documentation](https://opencv.org/)

## Contact

For any questions or suggestions keep it to  yourself.
