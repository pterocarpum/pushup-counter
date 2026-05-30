import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Landmark Indices
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24

# Set up MediaPipe Tasks with the HEAVY model
base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

cap = cv2.VideoCapture(0)

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # UI Overlay Background Box
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 160), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            
            # Extract 2D vector coordinates
            ls = np.array([landmarks[L_SHOULDER].x, landmarks[L_SHOULDER].y])
            rs = np.array([landmarks[R_SHOULDER].x, landmarks[R_SHOULDER].y])
            lh = np.array([landmarks[L_HIP].x, landmarks[L_HIP].y])
            rh = np.array([landmarks[R_HIP].x, landmarks[R_HIP].y])
            
            # Calculate midpoints and distance-invariant baseline
            shoulder_mid = (ls + rs) / 2
            hip_mid = (lh + rh) / 2
            torso_length = np.linalg.norm(shoulder_mid - hip_mid)
            
            if torso_length > 0:
                # Calculate your customized ratios
                vertical_ratio = abs(shoulder_mid[1] - hip_mid[1]) / torso_length
                shoulder_ratio = abs(ls[0] - rs[0]) / torso_length
                
                # --- Classification Logic using your calibrated thresholds ---
                # Check horizontal first to avoid false-positive sideways flags while lying down
                if vertical_ratio < 0.4:
                    orientation = "Horizontally Positioned"
                    status_color = (0, 165, 255)  # Orange
                elif shoulder_ratio < 0.3:
                    orientation = "Positioned Sideways"
                    status_color = (255, 0, 255)  # Magenta
                else:
                    orientation = "Vertically Positioned"
                    status_color = (0, 255, 0)  # Green

                # --- HUD Rendering ---
                cv2.putText(frame, f"STATE: {orientation}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
                
                cv2.putText(frame, f"Vertical Drop Ratio: {vertical_ratio:.2f} (Target < 0.4)", (20, 85), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
                
                cv2.putText(frame, f"Shoulder Width Ratio: {shoulder_ratio:.2f} (Target < 0.3)", (20, 125), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "STATE: No Target Detected", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Calibrated Pose Orientation Classifier', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()