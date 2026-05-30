import cv2
import mediapipe as mp
import time
from analysis import AdaptivePushupAnalyzer
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles


def calculate_angle(a, b, c):
    # Function removed as per request to avoid angle checks
    pass

class PushupCounter:
    def __init__(self, model_path='pose_landmarker_heavy.task'):
        self.analyzer = AdaptivePushupAnalyzer()
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        
        self.LEFT_SHOULDER, self.RIGHT_SHOULDER = 11, 12
        self.LEFT_ELBOW, self.RIGHT_ELBOW = 13, 14
        self.LEFT_WRIST, self.RIGHT_WRIST = 15, 16
        self.LEFT_HIP, self.RIGHT_HIP = 23, 24
        self.LEFT_KNEE, self.RIGHT_KNEE = 25, 26
        self.LEFT_ANKLE, self.RIGHT_ANKLE = 27, 28
        
        self.debug = True
        
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
        
        feedback_msg = ""
        
        try:
            detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print("MediaPipe Detection Error:", e)
            return frame, feedback_msg

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
                # Extract required landmarks
                wrist_r = pose_landmarks[self.RIGHT_WRIST]
                wrist_l = pose_landmarks[self.LEFT_WRIST]
                shoulder_r = pose_landmarks[self.RIGHT_SHOULDER]
                shoulder_l = pose_landmarks[self.LEFT_SHOULDER]
                waist_r = pose_landmarks[self.RIGHT_HIP]
                waist_l = pose_landmarks[self.LEFT_HIP]
                elbow_r = pose_landmarks[self.RIGHT_ELBOW]
                elbow_l = pose_landmarks[self.LEFT_ELBOW]
                knee_r = pose_landmarks[self.RIGHT_KNEE]
                knee_l = pose_landmarks[self.LEFT_KNEE]
                ankle_r = pose_landmarks[self.RIGHT_ANKLE]
                ankle_l = pose_landmarks[self.LEFT_ANKLE]
                
                # Check visibility
                min_vis = min([
                    wrist_r.visibility, wrist_l.visibility, 
                    shoulder_r.visibility, shoulder_l.visibility, 
                    waist_r.visibility, waist_l.visibility, 
                    elbow_r.visibility, elbow_l.visibility,
                    knee_r.visibility, knee_l.visibility,
                    ankle_r.visibility, ankle_l.visibility
                ])
                
                if min_vis == 0:
                    feedback_msg = "Ensure your full body is visible."
                else:
                    # Convert coordinates to NumPy arrays for geometric math (X, Y) in pixel coordinates
                    ls = np.array([shoulder_l.x * w, shoulder_l.y * h])
                    rs = np.array([shoulder_r.x * w, shoulder_r.y * h])
                    lh = np.array([waist_l.x * w, waist_l.y * h])
                    rh = np.array([waist_r.x * w, waist_r.y * h])
                    lk = np.array([knee_l.x * w, knee_l.y * h])
                    rk = np.array([knee_r.x * w, knee_r.y * h])
                    la = np.array([ankle_l.x * w, ankle_l.y * h])
                    ra = np.array([ankle_r.x * w, ankle_r.y * h])
                    
                    # Calculate midpoints
                    shoulder_mid = (ls + rs) / 2
                    hip_mid = (lh + rh) / 2
                    knee_mid = (lk + rk) / 2
                    ankle_mid = (la + ra) / 2
                    
                    body_length = np.linalg.norm(shoulder_mid - ankle_mid)
                    
                    if body_length == 0:
                        feedback_msg = "Adjust your stance."
                    else:
                        body_incline_ratio = abs(shoulder_mid[1] - ankle_mid[1]) / body_length
                        
                        is_not_standing = body_incline_ratio < 0.65 
                        is_feet_on_floor = ankle_mid[1] > (knee_mid[1] - 0.05 * body_length)
                        
                        if not is_not_standing:
                            feedback_msg = "Invalid: Standing/Steep Incline"
                        elif not is_feet_on_floor:
                            feedback_msg = "Invalid: Knee Push-up / Feet Lifted"
                        else:
                            feedback_msg = "Good position!"
                            
                            # Use wrists as your ground/floor anchor baseline
                            ground_y_px = ((wrist_l.y + wrist_r.y) / 2) * h
                            shoulder_y_px = shoulder_mid[1]
                            
                            # Proceed with analyzing points
                            shoulder_height_px = max(0, ground_y_px - shoulder_y_px)
                            
                            self.analyzer.process_point(current_time, shoulder_height_px)
            except IndexError:
                pass
        else:
            feedback_msg = "No person detected."
                
        # Overlay counts and status
        if self.debug:
            # Create a semi-transparent overlay for the HUD
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
                
            cv2.putText(annotated_image, state_str, (120, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 1, cv2.LINE_AA)
        
        return annotated_image, feedback_msg

    def close(self):
        if hasattr(self, 'landmarker'):
            self.landmarker.close()
