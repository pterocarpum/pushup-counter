import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Helper function to calculate angles geometrically
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

# Landmark Indices
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# Set up MediaPipe Tasks with the HEAVY model
base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

cap = cv2.VideoCapture('videos/pushup3.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30
frame_index = 0

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2] # Get frame dimensions to correct aspect ratio distortion
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        timestamp_ms = int(frame_index * 1000 / fps)
        frame_index += 1
        
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # UI Overlay Background Box (Height increased for more metrics)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (620, 200), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            
            # Extract coordinates mapped to actual image pixels (fixes angle distortion)
            ls = np.array([landmarks[L_SHOULDER].x * w, landmarks[L_SHOULDER].y * h])
            rs = np.array([landmarks[R_SHOULDER].x * w, landmarks[R_SHOULDER].y * h])
            lh = np.array([landmarks[L_HIP].x * w, landmarks[L_HIP].y * h])
            rh = np.array([landmarks[R_HIP].x * w, landmarks[R_HIP].y * h])
            lk = np.array([landmarks[L_KNEE].x * w, landmarks[L_KNEE].y * h])
            rk = np.array([landmarks[R_KNEE].x * w, landmarks[R_KNEE].y * h])
            la = np.array([landmarks[L_ANKLE].x * w, landmarks[L_ANKLE].y * h])
            ra = np.array([landmarks[R_ANKLE].x * w, landmarks[R_ANKLE].y * h])
            
            # Calculate midpoints for robust tracking
            shoulder_mid = (ls + rs) / 2
            hip_mid = (lh + rh) / 2
            knee_mid = (lk + rk) / 2
            ankle_mid = (la + ra) / 2
            
            # 1. Angles for crouching and bent-leg checks
            hip_angle = calculate_angle(shoulder_mid, hip_mid, knee_mid)
            knee_angle = calculate_angle(hip_mid, knee_mid, ankle_mid)
            
            # 2. Body dimensions
            body_length = np.linalg.norm(shoulder_mid - ankle_mid)
            
            if body_length > 0:
                body_incline_ratio = abs(shoulder_mid[1] - ankle_mid[1]) / body_length
                
                # --- Strict Plank / Push-up Position Validations ---
                
                # Check 1: Must be roughly horizontal (rejects standing)
                is_not_standing = body_incline_ratio < 0.65 
                
                # Check 2: Torso must be straight (rejects crouching / downward dog)
                is_torso_straight = hip_angle > 130.0 
                
                # Check 3: Legs must be straight (rejects knee push-ups with feet on ground)
                is_leg_straight = knee_angle > 130.0
                
                # Check 4: Feet must touch floor (rejects knee push-ups with feet lifted)
                # In pixel coords, higher Y means lower on screen. Ankle should be lower than/level with Knee.
                is_feet_on_floor = ankle_mid[1] > (knee_mid[1] - 0.05 * body_length)
                
                # Evaluate hierarchy of failures
                if not is_not_standing:
                    orientation = "Invalid: Standing/Steep Incline"
                    status_color = (0, 0, 255)
                elif not is_feet_on_floor:
                    orientation = "Invalid: Knee Push-up / Feet Lifted"
                    status_color = (0, 165, 255)
                elif not is_torso_straight:
                    orientation = "Invalid: Crouching / Hips Bent"
                    status_color = (0, 165, 255)
                elif not is_leg_straight:
                    orientation = "Invalid: Legs Bent"
                    status_color = (0, 165, 255)
                else:
                    orientation = "Valid Push-up Position"
                    status_color = (0, 255, 0)

                # --- HUD Rendering ---
                cv2.putText(frame, f"STATE: {orientation}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
                
                cv2.putText(frame, f"1. Body Incline Ratio: {body_incline_ratio:.2f} (Target < 0.65)", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240) if is_not_standing else (0,0,255), 1, cv2.LINE_AA)
                
                cv2.putText(frame, f"2. Torso Straightness: {hip_angle:.1f} deg (Target > 130)", (20, 115), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240) if is_torso_straight else (0,0,255), 1, cv2.LINE_AA)
                
                cv2.putText(frame, f"3. Leg Straightness: {knee_angle:.1f} deg (Target > 130)", (20, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240) if is_leg_straight else (0,0,255), 1, cv2.LINE_AA)
                
                cv2.putText(frame, f"4. Feet touching floor: {'Yes' if is_feet_on_floor else 'No'}", (20, 185), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240) if is_feet_on_floor else (0,0,255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "STATE: No Target Detected", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Strict Push-up Validator', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()