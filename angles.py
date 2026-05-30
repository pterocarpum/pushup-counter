import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os
import csv
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

# --- 1. Auto-Download the Pose Landmarker Model ---
model_path = 'pose_landmarker_heavy.task'
if not os.path.exists(model_path):
    print("Downloading the Pose Landmarker model (~30MB)...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete.")

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# --- 2. Core Processing Function ---
def process_pushup_video(video_name, video_dir='videos', show_video=False):
    """
    Processes a single push-up video, counts reps, records arm angles,
    saves stats to a CSV, and exports a trend plot.
    """
    video_source = f'{video_dir}/{video_name}'
    video_name = 'angles_'+video_name
    
    # Ensure output directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('stats', exist_ok=True)
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_source}")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30
    frame_count = 0

    counter = 0 
    stage = "UNKNOWN"

    time_steps, left_arm_angles, right_arm_angles = [], [], []
    csv_data = []

    # Initialize Tasks API Pose Landmarker
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )

    # Relevant landmarks for arm angles
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_ELBOW, RIGHT_ELBOW = 13, 14
    LEFT_WRIST, RIGHT_WRIST = 15, 16

    print(f"\n--- Starting processing for: {video_name} ---")
    print(f"Video View is {'ON' if show_video else 'OFF'}")
    start_time = time.time()

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # End of video
                
            frame_count += 1
            current_time = frame_count / fps
            timestamp_ms = int((frame_count * 1000) / fps)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if show_video:
                annotated_image = frame.copy()
            
            if detection_result.pose_landmarks:
                pose_landmarks = detection_result.pose_landmarks[0]
                
                if show_video:
                    drawing_utils.draw_landmarks(
                        image=annotated_image,
                        landmark_list=pose_landmarks,
                        connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                        landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style()
                    )
                
                try:
                    # Extract left arm coordinates
                    shoulder_l = [pose_landmarks[LEFT_SHOULDER].x, pose_landmarks[LEFT_SHOULDER].y]
                    elbow_l    = [pose_landmarks[LEFT_ELBOW].x, pose_landmarks[LEFT_ELBOW].y]
                    wrist_l    = [pose_landmarks[LEFT_WRIST].x, pose_landmarks[LEFT_WRIST].y]
                    
                    # Extract right arm coordinates
                    shoulder_r = [pose_landmarks[RIGHT_SHOULDER].x, pose_landmarks[RIGHT_SHOULDER].y]
                    elbow_r    = [pose_landmarks[RIGHT_ELBOW].x, pose_landmarks[RIGHT_ELBOW].y]
                    wrist_r    = [pose_landmarks[RIGHT_WRIST].x, pose_landmarks[RIGHT_WRIST].y]
                    
                    # Calculate both angles
                    left_angle = calculate_angle(shoulder_l, elbow_l, wrist_l)
                    right_angle = calculate_angle(shoulder_r, elbow_r, wrist_r)
                    
                    # Use left angle for standard rep counting
                    if left_angle > 160:
                        stage = "UP"
                    if left_angle < 90 and stage == 'UP':
                        stage = "DOWN"
                        counter += 1
                        
                    # Save the angle data for CSV and plotting
                    time_steps.append(current_time)
                    left_arm_angles.append(left_angle)
                    right_arm_angles.append(right_angle)
                    csv_data.append([round(current_time, 2), round(left_angle, 2), round(right_angle, 2)])

                    if show_video:
                        cv2.rectangle(annotated_image, (0,0), (225,73), (245,117,16), -1)
                        cv2.putText(annotated_image, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        cv2.putText(annotated_image, str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                        cv2.putText(annotated_image, 'STAGE', (100,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        cv2.putText(annotated_image, stage, (100,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                    
                except IndexError:
                    pass
            
            if show_video:
                cv2.imshow('Push-up Counter & Tracker', annotated_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if frame_count % 50 == 0:
                    percent = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    print(f"Processing... {frame_count} frames done ({percent:.1f}%). Reps counted: {counter}")

    cap.release()
    cv2.destroyAllWindows()

    end_time = time.time()
    print(f"Processing finished in {round(end_time - start_time, 2)} seconds!")
    print(f"Total Reps Counted: {counter}")

    # Save the Data to a CSV File
    if len(csv_data) > 0:
        csv_filename = f"data/{video_name[:-4]}.csv"
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time(s)", "Left(deg)", "Right(deg)"])
            writer.writerows(csv_data)
        print(f"Data successfully saved to '{csv_filename}'.")

    # Plotting the Data
    if len(time_steps) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(time_steps, left_arm_angles, label='Left Arm', color='blue', linewidth=2)
        plt.plot(time_steps, right_arm_angles, label='Right Arm', color='red', linewidth=2, alpha=0.7)
        
        plt.title(f'Arm Angles Over Time - {video_name[:-4]}')
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Angle (Degrees)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'stats/{video_name[:-4]}.png')
        plt.close() # Closes figure window to prevent memory overlay issues in a loop
        print(f"Plot saved to 'stats/{video_name[:-4]}.png'.")
    else:
        print("No data was collected.")

if __name__ == "__main__":
    video_list =  [f for f in os.listdir('videos') if os.path.isfile(os.path.join('videos', f))]
    
    # Run the loop
    for video in video_list:
        process_pushup_video(video_name=video, video_dir='videos', show_video=False)