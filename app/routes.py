from flask import Blueprint, render_template, Response
import cv2
import tensorflow as tf
import numpy as np

# Initialize the blueprint for the routes
app_routes = Blueprint('app_routes', __name__)

# Load the pre-trained gender classification model
gender_model = tf.keras.models.load_model("model/gender_model.h5")

# Load the YOLOv3 model for object detection
net = cv2.dnn.readNet("yolov3/yolov3.weights", "yolov3/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def detect_objects(frame):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return indices, class_ids, boxes

def classify_gender(frame):
    resized_frame = cv2.resize(frame, (64, 64))
    img_array = np.array(resized_frame) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = gender_model.predict(img_array)
    return "Male" if prediction[0][0] > 0.5 else "Female"

def gen():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        indices, class_ids, boxes = detect_objects(frame)
        for i in indices:
            i = i[0]
            box = boxes[i]
            x, y, w, h = box
            label = classify_gender(frame[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app_routes.route('/')
def index():
    return render_template('index.html')

@app_routes.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
