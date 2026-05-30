from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import cv2
import numpy as np
import base64
from counter import PushupCounter
import os

app = FastAPI()

# Make sure static folder exists
os.makedirs("static", exist_ok=True)

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get():
    with open("static/index.html", "r") as f:
        html = f.read()
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    pushup_counter = PushupCounter()
    
    try:
        while True:
            # Receive frame from client (base64 encoded string)
            data = await websocket.receive_text()
            
            # The client sends "data:image/jpeg;base64,..."
            if data.startswith('data:image/'):
                img_data = base64.b64decode(data.split(',')[1])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Process frame
                    annotated_frame, feedback = pushup_counter.process_frame(frame)
                    
                    # Encode back to base64
                    # We can use a lower quality to improve speed
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                    _, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
                    base64_str = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send annotated frame, rep count, and feedback
                    response_data = {
                        "image": f"data:image/jpeg;base64,{base64_str}",
                        "count": pushup_counter.analyzer.rep_count,
                        "feedback": feedback
                    }
                    await websocket.send_json(response_data)
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pushup_counter.close()

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
