from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import mysql.connector
from datetime import datetime
import threading
import time
import os
import base64
import json
from collections import defaultdict

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

class ObjectDetectionSystem:
    def __init__(self):
        # Initialize variables
        self.cap = None
        self.is_running = False
        self.detection_thread = None
        
        # YOLO model variables - Load on demand instead of init
        self.net = None
        self.classes = []
        self.colors = None
        self.output_layers = []
        
        # Reduce memory usage with smaller buffers
        self.frame_buffer_size = 1  # Reduced from default
        self.detection_interval = 0.1  # Process every 100ms
        
        # Tracking variables
        self.tracked_objects = {}
        self.entry_line_y = None
        self.exit_threshold = 50
        self.object_id_counter = 0
        
        # Configuration
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
        # Database config
        self.db_config = {
            'host': 'aiml-surveillance.czoey2saeds6.ap-south-1.rds.amazonaws.com',
            'user': 'admin',
            'password': 'vtpl1234',
            'database': 'CV_surveillance'
        }
        
        # Statistics
        self.total_entries = 0
        self.total_exits = 0
        self.current_frame = None
        
    def load_yolo_model(self):
        """Load YOLO model with memory optimization"""
        try:
            if self.net is not None:
                return True  # Model already loaded
                
            weights_path = 'yolov4-tiny.weights'  # Using tiny weights
            cfg_path = 'yolov4-tiny.cfg'  # Using tiny config
            names_path = 'coco.names'
            
            if not all(os.path.exists(f) for f in [weights_path, cfg_path, names_path]):
                print("Downloading required files...")
                # Add download logic here if needed
                return False
            
            # Load YOLO with memory optimization
            self.net = cv2.dnn.readNet(weights_path, cfg_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # Load classes with minimal memory
            with open(names_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Generate colors only for needed classes
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            
            # Get output layers efficiently
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            print("YOLO model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            return False
            
    def detect_objects(self, frame):
        """Memory-optimized object detection"""
        if self.net is None:
            if not self.load_yolo_model():
                return []
                
        # Resize frame to reduce memory usage
        frame = cv2.resize(frame, (416, 416))
        height, width = frame.shape[:2]
        
        # Create blob with memory optimization
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        del blob  # Free memory
        
        # Get detections
        outputs = self.net.forward(self.output_layers)
        
        # Process detections with minimal memory usage
        detected_objects = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    detected_objects.append({
                        'class_name': self.classes[class_id],
                        'confidence': confidence,
                        'bbox': [x, y, w, h],
                        'center': [center_x, center_y]
                    })
        
        return detected_objects

    # ... (rest of the class methods remain the same)

# Initialize with reduced worker threads
detection_system = ObjectDetectionSystem()

if __name__ == '__main__':
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)