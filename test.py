import cv2
import sys
from counter import PushupCounter

def main(video_source):
    counter = PushupCounter()
    counter.debug = True # ensure overlay is on for testing
    
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        sys.exit()

    print(f"Processing video {video_source}... Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # Video has ended
        
        # Process the frame through your class
        annotated_frame, feedback = counter.process_frame(frame)
        
        # Display the output window
        cv2.imshow('Push-Up Counter', annotated_frame)
        
        # Break loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 3. Clean up
    cap.release()
    counter.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main('videos/pushup6.mp4')