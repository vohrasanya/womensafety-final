# Object Detection and Gender Classification using YOLOv3 and TensorFlow

This project integrates YOLOv3 and TensorFlow for real-time object detection and gender classification using OpenCV and a webcam. The application can identify objects (like people, cars, etc.) and classify the gender (male or female) of any person detected. An optional Flask web interface is also provided for running the system via a web browser.

## Project Structure

```
object-detection-gender-classification/
├── yolov3/
│   ├── yolov3.cfg
│   ├── yolov3.weights
│   ├── coco.names
├── model/
│   ├── gender_model.h5
│   ├── train_gender_model.py
├── app/
│   ├── __init__.py
│   ├── routes.py
│   ├── camera.py
├── static/
│   ├── css/
│   ├── js/
├── templates/
│   ├── index.html
├── README.md
├── requirements.txt
├── main.py
├── run.sh
└── Dockerfile
```

## Features

- Real-time object detection using YOLOv3
- Gender classification using a pre-trained TensorFlow model
- Live webcam feed for detection
- Optional Flask web interface for easy use via a browser

## Prerequisites

- Python 3.6 or later
- TensorFlow 2.x
- OpenCV
- Flask (optional for web interface)

## Setup and Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/object-detection-gender-classification.git
    cd object-detection-gender-classification
    ```

2. **Download the YOLOv3 weights:**

    Download the `yolov3.weights` file from the following link and place it in the `yolov3/` directory:

    ```
    https://pjreddie.com/media/files/yolov3.weights
    ```

3. **Install dependencies:**

    Install the required Python packages using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

4. **Download or create the gender classification model:**

    Place the pre-trained `gender_model.h5` in the `model/` directory. You can optionally train your own model using the `train_gender_model.py` script.

## Running the Application

### Option 1: Run in the Terminal

Run the main script to perform object detection and gender classification directly in the terminal:

```bash
python main.py
```

This will open your webcam and display the live video feed with detections.

### Option 2: Run with Flask Web Interface

Run the Flask application to use the web interface:

```bash
python app/routes.py
```

Open your browser and visit `http://127.0.0.1:5000` to see the live stream with object detection and gender classification.

## Project Explanation

1. **YOLOv3 for Object Detection:**
   - YOLO (You Only Look Once) is a state-of-the-art object detection algorithm that is fast and accurate.
   - The model is configured using `yolov3.cfg` and weights are loaded from `yolov3.weights`. Object class names (like person, car, etc.) are loaded from `coco.names`.

2. **Gender Classification with TensorFlow:**
   - A custom-trained TensorFlow model is used to classify the gender (male or female) of detected persons.
   - The model takes detected faces, resizes them, and performs classification based on pre-trained weights.

3. **Real-Time Detection and Classification:**
   - The `main.py` script captures frames from the webcam, applies YOLOv3 to detect objects, and uses the gender model to classify detected persons.

## Files and Directories

- **`yolov3/`:** Contains YOLOv3 configuration files and weights.
- **`model/`:** Holds the gender classification model and optional training script.
- **`app/`:** Includes Flask application code for the web interface.
- **`templates/`:** HTML templates for the Flask web interface.
- **`main.py`:** The main script to run object detection and classification.
- **`requirements.txt`:** List of required Python packages.
- **`run.sh`:** Shell script to automate the running of the application (optional).
- **`Dockerfile`:** Docker configuration to containerize the project (optional).

## Deployment Options

You can deploy this application in various ways:
- Running locally with Python
- Docker container (use the included `Dockerfile`)
- Cloud platforms like Heroku or AWS (with Flask for web deployment)

## Contributing

Feel free to contribute to this project by creating issues or submitting pull requests. Make sure to follow the [contribution guidelines](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- YOLOv3: [YOLO Website](https://pjreddie.com/darknet/yolo/)
- TensorFlow: [TensorFlow Documentation](https://www.tensorflow.org/)
- OpenCV: [OpenCV Documentation](https://opencv.org/)

## Contact

For any questions or suggestions, feel free to reach out via GitHub or [LinkedIn](https://www.linkedin.com/in/enthusiastyuwe/).
```