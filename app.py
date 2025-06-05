# from flask import Flask, render_template, request, jsonify, Response
# from flask_socketio import SocketIO, emit
# import cv2
# import numpy as np
# import mysql.connector
# from datetime import datetime
# import threading
# import time
# import os
# import base64
# from collections import defaultdict
# import json

# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'your-secret-key-here'
# socketio = SocketIO(app, cors_allowed_origins="*")

# class ObjectDetectionSystem:
#     def __init__(self):
#         # Initialize variables
#         self.cap = None
#         self.is_running = False
#         self.detection_thread = None
        
#         # YOLO model variables
#         self.net = None
#         self.classes = []
#         self.colors = None
#         self.output_layers = []
        
#         # Tracking variables
#         self.tracked_objects = {}
#         self.entry_line_y = None
#         self.exit_threshold = 50
#         self.object_id_counter = 0
        
#         # Configuration
#         self.confidence_threshold = 0.5
#         self.nms_threshold = 0.4
        
#         # Database config
#         self.db_config = {
#             'host': 'localhost',
#             'user': 'root',
#             'password': '1234',
#             'database': 'nrkindex_api'
#         }
        
#         # Statistics
#         self.total_entries = 0
#         self.total_exits = 0
        
#         self.load_yolo_model()
        
#     def load_yolo_model(self):
#         """Load YOLO model and classes"""
#         try:
#             weights_path = 'yolov4.weights'
#             cfg_path = 'yolov4.cfg'
#             names_path = 'coco.names'
            
#             if not os.path.exists(weights_path):
#                 raise FileNotFoundError(f"YOLO weights file not found: {weights_path}")
#             if not os.path.exists(cfg_path):
#                 raise FileNotFoundError(f"YOLO config file not found: {cfg_path}")
#             if not os.path.exists(names_path):
#                 raise FileNotFoundError(f"YOLO names file not found: {names_path}")
            
#             # Load YOLO
#             self.net = cv2.dnn.readNet(weights_path, cfg_path)
            
#             # Load classes
#             with open(names_path, 'r') as f:
#                 self.classes = [line.strip() for line in f.readlines()]
            
#             # Generate colors
#             self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            
#             # Get output layers
#             layer_names = self.net.getLayerNames()
#             unconnected_out_layers = self.net.getUnconnectedOutLayers()
            
#             if len(unconnected_out_layers.shape) == 1:
#                 self.output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
#             else:
#                 self.output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
            
#             print("YOLO model loaded successfully!")
#             return True
            
#         except Exception as e:
#             print(f"Failed to load YOLO model: {e}")
#             return False
    
#     def start_camera(self):
#         """Start camera capture"""
#         try:
#             if self.cap is None or not self.cap.isOpened():
#                 self.cap = cv2.VideoCapture(0)
#                 if not self.cap.isOpened():
#                     raise Exception("Could not open camera")
#             return True
#         except Exception as e:
#             print(f"Failed to start camera: {e}")
#             return False
    
#     def stop_camera(self):
#         """Stop camera capture"""
#         if self.cap is not None:
#             self.cap.release()
#             self.cap = None
    
#     def start_detection(self):
#         """Start object detection"""
#         if self.cap is None or not self.cap.isOpened():
#             return False, "Please start camera first!"
        
#         if self.net is None:
#             return False, "YOLO model not loaded!"
            
#         self.is_running = True
#         self.detection_thread = threading.Thread(target=self.detection_loop)
#         self.detection_thread.daemon = True
#         self.detection_thread.start()
        
#         return True, "Object detection started!"
    
#     def stop_detection(self):
#         """Stop object detection"""
#         self.is_running = False
#         return "Object detection stopped!"
    
#     def detection_loop(self):
#         """Main detection loop"""
#         while self.is_running and self.cap is not None and self.cap.isOpened():
#             ret, frame = self.cap.read()
#             if not ret:
#                 break
            
#             # Detect objects
#             detected_objects = self.detect_objects(frame)
            
#             # Track entry/exit
#             self.track_entry_exit(detected_objects)
            
#             # Draw detections
#             frame = self.draw_detections(frame, detected_objects)
            
#             # Convert frame to base64 for web transmission
#             _, buffer = cv2.imencode('.jpg', frame)
#             frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
#             # Emit to all connected clients
#             socketio.emit('video_frame', {
#                 'frame': frame_base64,
#                 'detections': self.format_detections_for_web(detected_objects),
#                 'stats': {
#                     'total_entries': self.total_entries,
#                     'total_exits': self.total_exits,
#                     'current_objects': len(detected_objects),
#                     'tracked_objects': len([obj for obj in self.tracked_objects.values() if obj['status'] == 'present'])
#                 }
#             })
            
#             time.sleep(0.03)  # ~30 FPS
    
#     def detect_objects(self, frame):
#         """Detect objects in frame using YOLO"""
#         height, width, channels = frame.shape
        
#         if self.entry_line_y is None:
#             self.entry_line_y = height // 2
        
#         # Prepare image for YOLO
#         blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#         self.net.setInput(blob)
#         outputs = self.net.forward(self.output_layers)
        
#         class_ids = []
#         confidences = []
#         boxes = []
        
#         for output in outputs:
#             for detection in output:
#                 scores = detection[5:]
#                 class_id = np.argmax(scores)
#                 confidence = scores[class_id]
                
#                 if confidence > self.confidence_threshold:
#                     center_x = int(detection[0] * width)
#                     center_y = int(detection[1] * height)
#                     w = int(detection[2] * width)
#                     h = int(detection[3] * height)
                    
#                     x = int(center_x - w / 2)
#                     y = int(center_y - h / 2)
                    
#                     boxes.append([x, y, w, h])
#                     confidences.append(float(confidence))
#                     class_ids.append(class_id)
        
#         # Apply Non-Maximum Suppression
#         indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
#         detected_objects = []
#         if len(indices) > 0:
#             for i in indices.flatten():
#                 x, y, w, h = boxes[i]
#                 class_name = self.classes[class_ids[i]]
#                 confidence = confidences[i]
                
#                 center_x = x + w // 2
#                 center_y = y + h // 2
                
#                 detected_objects.append({
#                     'class_name': class_name,
#                     'confidence': confidence,
#                     'bbox': [x, y, w, h],
#                     'center': [center_x, center_y]
#                 })
        
#         return detected_objects
    
#     def track_entry_exit(self, detected_objects):
#         """Track entry and exit of objects"""
#         current_time = datetime.now()
#         current_objects = set()
        
#         for obj in detected_objects:
#             class_name = obj['class_name']
#             center_x, center_y = obj['center']
#             confidence = obj['confidence']
            
#             obj_key = f"{class_name}_{center_x//80}_{center_y//80}"
#             current_objects.add(obj_key)
            
#             if obj_key not in self.tracked_objects:
#                 self.tracked_objects[obj_key] = {
#                     'class_name': class_name,
#                     'last_position': [center_x, center_y],
#                     'status': 'present',
#                     'entry_time': current_time,
#                     'last_seen': current_time,
#                     'confidence': confidence,
#                     'db_id': None
#                 }
                
#                 db_id = self.log_to_database(class_name, current_time, status='entered')
#                 self.tracked_objects[obj_key]['db_id'] = db_id
                
#                 self.total_entries += 1
                
#                 # Emit entry event
#                 socketio.emit('object_event', {
#                     'type': 'entry',
#                     'object': class_name,
#                     'confidence': confidence,
#                     'time': current_time.strftime("%H:%M:%S")
#                 })
                
#             else:
#                 tracked_obj = self.tracked_objects[obj_key]
#                 tracked_obj['last_position'] = [center_x, center_y]
#                 tracked_obj['last_seen'] = current_time
#                 tracked_obj['confidence'] = max(tracked_obj['confidence'], confidence)
        
#         # Check for exits
#         current_timestamp = current_time.timestamp()
#         objects_to_remove = []
        
#         for obj_key, obj_data in self.tracked_objects.items():
#             time_since_seen = current_timestamp - obj_data['last_seen'].timestamp()
            
#             if obj_key not in current_objects and time_since_seen > 2.0:
#                 if obj_data['status'] == 'present':
#                     obj_data['status'] = 'exited'
#                     obj_data['exit_time'] = current_time
                    
#                     self.update_exit_in_database(obj_data['db_id'], current_time)
#                     self.total_exits += 1
                    
#                     duration = (current_time - obj_data['entry_time']).total_seconds()
                    
#                     # Emit exit event
#                     socketio.emit('object_event', {
#                         'type': 'exit',
#                         'object': obj_data['class_name'],
#                         'duration': f"{duration:.1f}s",
#                         'time': current_time.strftime("%H:%M:%S")
#                     })
                
#                 if time_since_seen > 5.0:
#                     objects_to_remove.append(obj_key)
        
#         for obj_key in objects_to_remove:
#             del self.tracked_objects[obj_key]
    
#     def draw_detections(self, frame, detected_objects):
#         """Draw bounding boxes and labels on frame"""
#         height, width = frame.shape[:2]
        
#         # Draw detection area border
#         cv2.rectangle(frame, (10, 10), (width-10, height-10), (0, 255, 0), 2)
#         cv2.putText(frame, "Detection Area", (15, 35), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
#         for obj in detected_objects:
#             x, y, w, h = obj['bbox']
#             class_name = obj['class_name']
#             confidence = obj['confidence']
            
#             color = self.colors[self.classes.index(class_name)]
            
#             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
#             center_x, center_y = obj['center']
#             cv2.circle(frame, (center_x, center_y), 5, color, -1)
            
#             label = f"{class_name}: {confidence:.2f}"
#             label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
#             cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
#             cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
#         # Display statistics
#         stats_y = height - 60
#         cv2.putText(frame, f"Entries: {self.total_entries}", (10, stats_y), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#         cv2.putText(frame, f"Exits: {self.total_exits}", (10, stats_y + 20), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#         cv2.putText(frame, f"Current: {len(detected_objects)}", (10, stats_y + 40), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
#         return frame
    
#     def format_detections_for_web(self, detected_objects):
#         """Format detection data for web display"""
#         formatted = []
#         for obj in detected_objects:
#             formatted.append({
#                 'class_name': obj['class_name'],
#                 'confidence': round(obj['confidence'], 2),
#                 'center': obj['center']
#             })
#         return formatted
    
#     def log_to_database(self, obj_name, timestamp, status='present'):
#         """Log object detection to MySQL database"""
#         try:
#             conn = mysql.connector.connect(**self.db_config)
#             cursor = conn.cursor()
            
#             if status == 'entered':
#                 query = """
#                 INSERT INTO object_tracking (objname, dateandtime, entrytime, status)
#                 VALUES (%s, %s, %s, %s)
#                 """
#                 cursor.execute(query, (obj_name, timestamp, timestamp, 'present'))
#                 record_id = cursor.lastrowid
                
#             conn.commit()
#             cursor.close()
#             conn.close()
            
#             return record_id
            
#         except Exception as e:
#             print(f"Database error: {e}")
#             return None
    
#     def update_exit_in_database(self, record_id, exit_time):
#         """Update database record with exit time"""
#         if record_id is None:
#             return
            
#         try:
#             conn = mysql.connector.connect(**self.db_config)
#             cursor = conn.cursor()
            
#             query = """
#             UPDATE object_tracking 
#             SET exittime = %s, status = %s 
#             WHERE id = %s
#             """
#             cursor.execute(query, (exit_time, 'exited', record_id))
            
#             conn.commit()
#             cursor.close()
#             conn.close()
            
#         except Exception as e:
#             print(f"Database update error: {e}")
    
#     def test_database_connection(self):
#         """Test database connection and create table if needed"""
#         try:
#             conn = mysql.connector.connect(**self.db_config)
#             cursor = conn.cursor()
            
#             create_table_query = """
#             CREATE TABLE IF NOT EXISTS object_tracking (
#                 id INT AUTO_INCREMENT PRIMARY KEY,
#                 objname VARCHAR(50) NOT NULL,
#                 dateandtime DATETIME NOT NULL,
#                 entrytime DATETIME,
#                 exittime DATETIME,
#                 status ENUM('entered', 'exited', 'present') DEFAULT 'present',
#                 INDEX idx_objname (objname),
#                 INDEX idx_datetime (dateandtime)
#             )
#             """
#             cursor.execute(create_table_query)
#             conn.commit()
#             cursor.close()
#             conn.close()
            
#             return True, "Database connection successful!"
            
#         except Exception as e:
#             return False, f"Database connection failed: {e}"
    
#     def get_database_records(self, limit=100):
#         """Get recent database records"""
#         try:
#             conn = mysql.connector.connect(**self.db_config)
#             cursor = conn.cursor(dictionary=True)
#             cursor.execute("SELECT * FROM object_tracking ORDER BY dateandtime DESC LIMIT %s", (limit,))
#             records = cursor.fetchall()
            
#             # Convert datetime objects to strings
#             for record in records:
#                 for key, value in record.items():
#                     if isinstance(value, datetime):
#                         record[key] = value.strftime("%Y-%m-%d %H:%M:%S")
            
#             cursor.close()
#             conn.close()
            
#             return records
            
#         except Exception as e:
#             print(f"Database query error: {e}")
#             return []

# # Initialize the detection system
# detection_system = ObjectDetectionSystem()

# # Routes
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/api/start_camera', methods=['POST'])
# def start_camera():
#     success = detection_system.start_camera()
#     return jsonify({
#         'success': success,
#         'message': 'Camera started successfully!' if success else 'Failed to start camera'
#     })

# @app.route('/api/stop_camera', methods=['POST'])
# def stop_camera():
#     detection_system.stop_camera()
#     return jsonify({
#         'success': True,
#         'message': 'Camera stopped successfully!'
#     })

# @app.route('/api/start_detection', methods=['POST'])
# def start_detection():
#     success, message = detection_system.start_detection()
#     return jsonify({
#         'success': success,
#         'message': message
#     })

# @app.route('/api/stop_detection', methods=['POST'])
# def stop_detection():
#     message = detection_system.stop_detection()
#     return jsonify({
#         'success': True,
#         'message': message
#     })

# @app.route('/api/update_settings', methods=['POST'])
# def update_settings():
#     data = request.json
#     detection_system.confidence_threshold = data.get('confidence', 0.5)
#     detection_system.exit_threshold = data.get('threshold', 50)
    
#     return jsonify({
#         'success': True,
#         'message': 'Settings updated successfully!'
#     })

# @app.route('/api/test_database', methods=['POST'])
# def test_database():
#     data = request.json
#     if data:
#         detection_system.db_config.update(data)
    
#     success, message = detection_system.test_database_connection()
#     return jsonify({
#         'success': success,
#         'message': message
#     })

# @app.route('/api/database_records')
# def get_database_records():
#     limit = request.args.get('limit', 100, type=int)
#     records = detection_system.get_database_records(limit)
#     return jsonify(records)

# @app.route('/api/stats')
# def get_stats():
#     return jsonify({
#         'total_entries': detection_system.total_entries,
#         'total_exits': detection_system.total_exits,
#         'tracked_objects': len([obj for obj in detection_system.tracked_objects.values() if obj['status'] == 'present'])
#     })

# # WebSocket events
# @socketio.on('connect')
# def handle_connect():
#     print('Client connected')

# @socketio.on('disconnect')
# def handle_disconnect():
#     print('Client disconnected')

# if __name__ == '__main__':
#     socketio.run(app, debug=True, host='0.0.0.0', port=5000)

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
        
        # YOLO model variables
        self.net = None
        self.classes = []
        self.colors = None
        self.output_layers = []
        
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
            'host': 'localhost',
            'user': 'root',
            'password': '1234',
            'database': 'nrkindex_api'
        }
        
        # Statistics
        self.total_entries = 0
        self.total_exits = 0
        self.current_frame = None
        
        # Load YOLO model on initialization
        self.load_yolo_model()
        
    def load_yolo_model(self):
        """Load YOLO model and classes"""
        try:
            # Check if files exist
            weights_path = 'yolov4.weights'
            cfg_path = 'yolov4.cfg'
            names_path = 'coco.names'
            
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"YOLO weights file not found: {weights_path}")
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"YOLO config file not found: {cfg_path}")
            if not os.path.exists(names_path):
                raise FileNotFoundError(f"YOLO names file not found: {names_path}")
            
            # Load YOLO
            self.net = cv2.dnn.readNet(weights_path, cfg_path)
            
            # Load classes
            with open(names_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Generate colors
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            
            # Get output layers
            layer_names = self.net.getLayerNames()
            unconnected_out_layers = self.net.getUnconnectedOutLayers()
            
            # Handle both old and new OpenCV versions
            if len(unconnected_out_layers.shape) == 1:
                self.output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
            else:
                self.output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
            
            print("YOLO model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            return False
            
    def start_camera(self):
        """Start camera capture"""
        try:
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise Exception("Could not open camera")
            return True
        except Exception as e:
            print(f"Failed to start camera: {e}")
            return False
            
    def stop_camera(self):
        """Stop camera capture"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        return True
        
    def start_detection(self):
        """Start object detection"""
        if self.cap is None or not self.cap.isOpened():
            return False, "Please start camera first!"
            
        if self.net is None:
            return False, "YOLO model not loaded!"
            
        self.is_running = True
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        return True, "Object detection started!"
        
    def stop_detection(self):
        """Stop object detection"""
        self.is_running = False
        return True, "Object detection stopped!"
        
    def detection_loop(self):
        """Main detection loop running in separate thread"""
        while self.is_running and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Detect objects
            detected_objects = self.detect_objects(frame)
            
            # Track entry/exit
            self.track_entry_exit(detected_objects)
            
            # Draw detections
            frame = self.draw_detections(frame, detected_objects)
            
            # Store current frame
            self.current_frame = frame
            
            # Emit updates to web clients
            self.emit_updates(detected_objects)
            
            time.sleep(0.03)  # ~30 FPS
            
    def detect_objects(self, frame):
        """Detect objects in frame using YOLO"""
        height, width, channels = frame.shape
        
        # Set entry line if not set
        if self.entry_line_y is None:
            self.entry_line_y = height // 2
        
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Information to show on screen
        class_ids = []
        confidences = []
        boxes = []
        
        # Extract information from outputs
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        detected_objects = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                class_name = self.classes[class_ids[i]]
                confidence = confidences[i]
                
                center_x = x + w // 2
                center_y = y + h // 2
                
                detected_objects.append({
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [x, y, w, h],
                    'center': [center_x, center_y]
                })
        
        return detected_objects
        
    def track_entry_exit(self, detected_objects):
        """Track entry and exit of objects based on appearance/disappearance"""
        current_time = datetime.now()
        current_objects = set()
        
        # Process each detected object
        for obj in detected_objects:
            class_name = obj['class_name']
            center_x, center_y = obj['center']
            confidence = obj['confidence']
            
            # Create a unique identifier for tracking based on position and class
            obj_key = f"{class_name}_{center_x//80}_{center_y//80}"
            current_objects.add(obj_key)
            
            if obj_key not in self.tracked_objects:
                # NEW OBJECT APPEARED - Log as ENTRY
                self.tracked_objects[obj_key] = {
                    'class_name': class_name,
                    'last_position': [center_x, center_y],
                    'status': 'present',
                    'entry_time': current_time,
                    'last_seen': current_time,
                    'confidence': confidence,
                    'db_id': None  # Will store database record ID
                }
                
                # Log entry to database
                db_id = self.log_to_database(class_name, current_time, status='entered')
                self.tracked_objects[obj_key]['db_id'] = db_id
                
                self.total_entries += 1
                
            else:
                # UPDATE EXISTING OBJECT
                tracked_obj = self.tracked_objects[obj_key]
                tracked_obj['last_position'] = [center_x, center_y]
                tracked_obj['last_seen'] = current_time
                tracked_obj['confidence'] = max(tracked_obj['confidence'], confidence)
        
        # Check for objects that disappeared (EXIT)
        current_timestamp = current_time.timestamp()
        objects_to_remove = []
        
        for obj_key, obj_data in self.tracked_objects.items():
            # If object hasn't been seen for 2 seconds, consider it as exited
            time_since_seen = current_timestamp - obj_data['last_seen'].timestamp()
            
            if obj_key not in current_objects and time_since_seen > 2.0:
                # OBJECT DISAPPEARED - Log as EXIT
                if obj_data['status'] == 'present':
                    obj_data['status'] = 'exited'
                    obj_data['exit_time'] = current_time
                    
                    # Update database record with exit time
                    self.update_exit_in_database(obj_data['db_id'], current_time)
                    
                    self.total_exits += 1
                
                # Mark for removal after logging exit
                if time_since_seen > 5.0:  # Keep in memory for 5 seconds after exit
                    objects_to_remove.append(obj_key)
        
        # Remove old tracked objects
        for obj_key in objects_to_remove:
            del self.tracked_objects[obj_key]
            
    def draw_detections(self, frame, detected_objects):
        """Draw bounding boxes and labels on frame"""
        height, width = frame.shape[:2]
        
        # Draw detection area border
        cv2.rectangle(frame, (10, 10), (width-10, height-10), (0, 255, 0), 2)
        cv2.putText(frame, "Detection Area", (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw detected objects
        for obj in detected_objects:
            x, y, w, h = obj['bbox']
            class_name = obj['class_name']
            confidence = obj['confidence']
            
            # Get color for this class
            color = self.colors[self.classes.index(class_name)]
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw center point
            center_x, center_y = obj['center']
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
            
            # Draw label with background
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display current statistics on frame
        stats_y = height - 60
        cv2.putText(frame, f"Entries: {self.total_entries}", (10, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Exits: {self.total_exits}", (10, stats_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Current: {len(detected_objects)}", (10, stats_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
        
    def emit_updates(self, detected_objects):
        """Emit updates to web clients via SocketIO"""
        # Prepare data for web client
        detection_data = []
        for obj in detected_objects:
            detection_data.append({
                'class_name': obj['class_name'],
                'confidence': round(obj['confidence'], 2),
                'bbox': obj['bbox']
            })
        
        # Prepare tracking data
        tracking_data = []
        present_objects = 0
        for obj_key, obj_data in self.tracked_objects.items():
            if obj_data['status'] == 'present':
                duration = (datetime.now() - obj_data['entry_time']).total_seconds()
                tracking_data.append({
                    'class_name': obj_data['class_name'],
                    'status': 'PRESENT',
                    'duration': round(duration, 0)
                })
                present_objects += 1
        
        # Emit to all connected clients
        socketio.emit('detection_update', {
            'detections': detection_data,
            'tracking': tracking_data,
            'stats': {
                'total_entries': self.total_entries,
                'total_exits': self.total_exits,
                'present_objects': present_objects
            }
        })
        
    def get_current_frame_base64(self):
        """Get current frame as base64 string for web display"""
        if self.current_frame is not None:
            # Resize frame for web display
            frame = cv2.resize(self.current_frame, (640, 480))
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            return frame_base64
        return None
        
    def log_to_database(self, obj_name, timestamp, status='present'):
        """Log object detection to MySQL database and return the record ID"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            if status == 'entered':
                query = """
                INSERT INTO object_tracking (objname, dateandtime, entrytime, status)
                VALUES (%s, %s, %s, %s)
                """
                cursor.execute(query, (obj_name, timestamp, timestamp, 'present'))
                record_id = cursor.lastrowid
                
            conn.commit()
            cursor.close()
            conn.close()
            
            return record_id
            
        except Exception as e:
            print(f"Database error: {e}")
            return None
            
    def update_exit_in_database(self, record_id, exit_time):
        """Update database record with exit time"""
        if record_id is None:
            return
            
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            query = """
            UPDATE object_tracking 
            SET exittime = %s, status = %s 
            WHERE id = %s
            """
            cursor.execute(query, (exit_time, 'exited', record_id))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Database update error: {e}")
            
    def test_db_connection(self):
        """Test database connection and create table if needed"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            
            # Create table if not exists
            cursor = conn.cursor()
            create_table_query = """
            CREATE TABLE IF NOT EXISTS object_tracking (
                id INT AUTO_INCREMENT PRIMARY KEY,
                objname VARCHAR(50) NOT NULL,
                dateandtime DATETIME NOT NULL,
                entrytime DATETIME,
                exittime DATETIME,
                status ENUM('entered', 'exited', 'present') DEFAULT 'present',
                INDEX idx_objname (objname),
                INDEX idx_datetime (dateandtime)
            )
            """
            cursor.execute(create_table_query)
            conn.commit()
            cursor.close()
            conn.close()
            
            return True, "Database connection successful!"
            
        except Exception as e:
            return False, f"Database connection failed: {e}"
            
    def get_database_records(self, limit=100):
        """Get database records for display"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM object_tracking ORDER BY dateandtime DESC LIMIT %s", (limit,))
            records = cursor.fetchall()
            
            # Convert to list of dictionaries
            columns = ['id', 'objname', 'dateandtime', 'entrytime', 'exittime', 'status']
            result = []
            for record in records:
                record_dict = {}
                for i, value in enumerate(record):
                    if isinstance(value, datetime):
                        record_dict[columns[i]] = value.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        record_dict[columns[i]] = value
                result.append(record_dict)
                
            cursor.close()
            conn.close()
            
            return True, result
            
        except Exception as e:
            return False, f"Failed to load database: {e}"

# Initialize the detection system
detection_system = ObjectDetectionSystem()

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start_camera', methods=['POST'])
def start_camera():
    success = detection_system.start_camera()
    return jsonify({'success': success, 'message': 'Camera started' if success else 'Failed to start camera'})

@app.route('/api/stop_camera', methods=['POST'])
def stop_camera():
    success = detection_system.stop_camera()
    return jsonify({'success': success, 'message': 'Camera stopped'})

@app.route('/api/start_detection', methods=['POST'])
def start_detection():
    success, message = detection_system.start_detection()
    return jsonify({'success': success, 'message': message})

@app.route('/api/stop_detection', methods=['POST'])
def stop_detection():
    success, message = detection_system.stop_detection()
    return jsonify({'success': success, 'message': message})

@app.route('/api/set_config', methods=['POST'])
def set_config():
    data = request.json
    if 'confidence' in data:
        detection_system.confidence_threshold = float(data['confidence'])
    if 'threshold' in data:
        detection_system.exit_threshold = int(data['threshold'])
    return jsonify({'success': True, 'message': 'Configuration updated'})

@app.route('/api/test_db', methods=['POST'])
def test_db():
    data = request.json
    if data:
        detection_system.db_config.update(data)
    success, message = detection_system.test_db_connection()
    return jsonify({'success': success, 'message': message})

@app.route('/api/get_db_records')
def get_db_records():
    success, data = detection_system.get_database_records()
    return jsonify({'success': success, 'data': data if success else [], 'message': data if not success else ''})

# @app.route('/video_feed')
# def video_feed():
#     def generate():
#         while True:
#             frame_b64 = detection_system.get_current_frame_base64()
#             if frame_b64:
#                 yield f"data:image/jpeg;base64,{frame_b64}\n\n"
#             time.sleep(0.1)
    
#     return Response(generate(), mimetype='text/plain')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame_b64 = detection_system.get_current_frame_base64()
            if frame_b64:
                yield f"data:image/jpeg;base64,{frame_b64}"
                break  # Send one frame per request
            else:
                yield "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                break  # Send a 1x1 transparent pixel if no frame
    
    return Response(generate(), mimetype='text/plain')
# SocketIO Events
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=8080)