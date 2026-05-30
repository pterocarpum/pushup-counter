import cv2
import mediapipe as mp
import time
from analysis import AdaptivePushupAnalyzer
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

DEBUG=False

class PushupCounter:
    def __init__(self, model_path='pose_landmarker_heavy.task'):
        self.analyzer = AdaptivePushupAnalyzer("Shoulder")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        
        self.LEFT_SHOULDER, self.RIGHT_SHOULDER = 11, 12
        self.LEFT_ELBOW = 13
        self.LEFT_WRIST, self.RIGHT_WRIST = 15, 16
        self.LEFT_HIP, self.RIGHT_HIP = 23, 24
        
        self.start_time = time.time()
        self.last_timestamp_ms = -1
        
    def process_frame(self, frame, current_time=None):
        if current_time is None:
            current_time = time.time() - self.start_time
            
        timestamp_ms = int(current_time * 1000)
        
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms

        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        try:
            detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print("MediaPipe Detection Error:", e)
            return frame

        annotated_image = frame.copy()
        
        if detection_result.pose_landmarks:
            pose_landmarks = detection_result.pose_landmarks[0]
            
            # Draw landmarks
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=pose_landmarks,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style()
            )
            
            try:
                # Calculate shoulder height
                wrist_r_y = pose_landmarks[self.RIGHT_WRIST].y
                wrist_l_y = pose_landmarks[self.LEFT_WRIST].y
                ground_y = (wrist_l_y + wrist_r_y) / 2
                
                shoulder_y = (pose_landmarks[self.LEFT_SHOULDER].y + pose_landmarks[self.RIGHT_SHOULDER].y) / 2
                
                shoulder_height_px = max(0, (ground_y - shoulder_y) * h)
                
                # Pass to analyzer
                self.analyzer.process_point(current_time, shoulder_height_px)
            except IndexError:
                pass
                
        # Overlay counts and status
        # DEBUGGING PURPOSES
        if DEBUG:
            overlay = annotated_image.copy()
            cv2.rectangle(overlay, (10, 10), (280, 130), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, annotated_image, 0.3, 0, annotated_image)
            
            # Draw border
            cv2.rectangle(annotated_image, (10, 10), (280, 130), (100, 200, 255), 2)
            
            # Text overlay
            cv2.putText(annotated_image, 'PUSH-UPS (LIVE)', (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            
            # Rep count text
            cv2.putText(annotated_image, str(self.analyzer.rep_count), (25, 105), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (100, 255, 100), 4, cv2.LINE_AA)
        
        # Display analysis state
        state_str = self.analyzer.state
        if state_str == 'LOOKING_FOR_MIN':
            state_str = "DOWN PHASE"
            state_color = (100, 100, 255) # Reddish
        elif state_str == 'LOOKING_FOR_MAX':
            state_str = "UP PHASE"
            state_color = (255, 200, 100) # Bluish
        else:
            state_color = (200, 200, 200) # Gray

        if DEBUG: cv2.putText(annotated_image, state_str, (120, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 1, cv2.LINE_AA)
        
        return annotated_image

    def close(self):
        if hasattr(self, 'landmarker'):
            self.landmarker.close()
