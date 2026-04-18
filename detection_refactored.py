"""
Refactored Detection Module - YOLO + Gender Classification
"""
import cv2
import numpy as np
import tensorflow as tf
import math
from typing import List, Tuple, Dict
from config import (
    YOLO_WEIGHTS, YOLO_CFG, YOLO_NAMES,
    GENDER_MODEL_PATH, ALERT_RULES
)


class PersonDetector:
    """
    Detects persons using YOLOv3 and classifies gender
    """
    
    def __init__(self, model_type: str = 'tiny'):
        """
        Initialize detector
        
        Args:
            model_type: 'tiny' for YOLOv3-tiny or 'full' for full YOLOv3
        """
        self.model_type = model_type
        
        # Load YOLO
        self.net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Get layer names
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Load class names
        with open(YOLO_NAMES, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Load gender model
        self.gender_model = tf.keras.models.load_model(GENDER_MODEL_PATH)
        
        # Face cascade for preprocessing
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        self.gender_threshold = ALERT_RULES.get('gender_threshold', 0.55)
    
    def predict_gender(self, face: np.ndarray) -> str:
        """
        Predict gender from face
        
        Args:
            face: Face image (as numpy array)
        
        Returns:
            'Male' or 'Female'
        """
        if face is None or face.size == 0:
            return "Unknown"
        
        try:
            # Resize and normalize
            face_resized = cv2.resize(face, (64, 64))
            face_normalized = face_resized / 255.0
            face_expanded = np.expand_dims(face_normalized, axis=0)
            
            # Predict
            prediction = self.gender_model.predict(face_expanded, verbose=0)[0][0]
            
            # Threshold-based classification
            return "Female" if prediction < self.gender_threshold else "Male"
        
        except Exception as e:
            print(f"⚠️ Error predicting gender: {e}")
            return "Unknown"
    
    def detect_persons(self, frame: np.ndarray) -> Tuple[List[Dict], List[np.ndarray]]:
        """
        Detect persons and classify gender
        
        Args:
            frame: Input video frame
        
        Returns:
            Tuple of (detections list, boxes list)
            Each detection: {
                'bbox': (x, y, w, h),
                'center': (cx, cy),
                'gender': 'Male'/'Female'/'Unknown',
                'confidence': float
            }
        """
        height, width = frame.shape[:2]
        detections = []
        
        # Prepare blob for YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences_list = []
        class_ids = []
        
        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Only person class (class_id = 0)
                if confidence > ALERT_RULES.get('min_confidence', 0.5) and class_id == 0:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences_list.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply NMS (Non-Maximum Suppression)
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences_list, 
            ALERT_RULES.get('min_confidence', 0.5), 
            0.4
        )
        
        # Extract persons and predict gender
        for i in indices:
            idx = i if isinstance(i, int) else i[0]
            x, y, w, h = boxes[idx]
            
            # Extract person region
            person_region = frame[max(0, y):y+h, max(0, x):x+w]
            
            # Try to detect face for gender classification
            gender = "Unknown"
            try:
                if person_region.size > 0:
                    # Detect faces in person region
                    faces = self.face_cascade.detectMultiScale(
                        person_region, 1.1, 4, minSize=(20, 20)
                    )
                    
                    if len(faces) > 0:
                        # Use largest face
                        largest_face = max(faces, key=lambda f: f[2] * f[3])
                        fx, fy, fw, fh = largest_face
                        face_region = person_region[fy:fy+fh, fx:fx+fw]
                        gender = self.predict_gender(face_region)
            except:
                pass
            
            detection = {
                'bbox': (x, y, w, h),
                'center': (x + w//2, y + h//2),
                'gender': gender,
                'confidence': confidences_list[idx]
            }
            detections.append(detection)
        
        return detections, boxes
    
    def get_gender_counts(self, detections: List[Dict]) -> Tuple[int, int]:
        """
        Get counts of males and females
        
        Args:
            detections: List of detections from detect_persons()
        
        Returns:
            Tuple of (male_count, female_count)
        """
        male_count = sum(1 for d in detections if d['gender'] == 'Male')
        female_count = sum(1 for d in detections if d['gender'] == 'Female')
        
        return male_count, female_count
    
    def get_gender_centers(self, detections: List[Dict]) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Get center coordinates grouped by gender
        
        Args:
            detections: List of detections from detect_persons()
        
        Returns:
            Tuple of (male_centers, female_centers)
        """
        male_centers = [d['center'] for d in detections if d['gender'] == 'Male']
        female_centers = [d['center'] for d in detections if d['gender'] == 'Female']
        
        return male_centers, female_centers


class AlertRuleEngine:
    """
    Determine if alert condition is met
    """
    
    def __init__(self):
        self.min_females = ALERT_RULES.get('min_females', 1)
        self.min_males = ALERT_RULES.get('min_males', 2)
        self.alert_radius = ALERT_RULES.get('alert_radius', 200)
    
    def check_alert_condition(self, male_count: int, female_count: int,
                            male_centers: List[Tuple], female_centers: List[Tuple]) -> bool:
        """
        Check if alert condition is met
        
        Args:
            male_count: Number of males detected
            female_count: Number of females detected
            male_centers: Centers of males
            female_centers: Centers of females
        
        Returns:
            True if alert should be triggered
        """
        # Condition 1: Enough females
        if female_count < self.min_females:
            return False
        
        # Condition 2: Enough males
        if male_count < self.min_males:
            return False
        
        # Condition 3: Proximity check (males near females)
        if self._check_proximity(male_centers, female_centers):
            return True
        
        return False
    
    def _check_proximity(self, male_centers: List[Tuple], female_centers: List[Tuple]) -> bool:
        """
        Check if any male is within alert radius of any female
        """
        for male_center in male_centers:
            for female_center in female_centers:
                distance = self._euclidean_distance(male_center, female_center)
                if distance <= self.alert_radius:
                    return True
        
        return False
    
    @staticmethod
    def _euclidean_distance(p1: Tuple, p2: Tuple) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
