import cv2
import sys
from counter import PushupCounter

def main():
    counter = PushupCounter()
    video_source = "videos/pushup5.mp4" 
    
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        sys.exit()

    print("Processing video... Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # Video has ended
        
        # Process the frame through your class
        annotated_frame = counter.process_frame(frame)
        
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
    main()