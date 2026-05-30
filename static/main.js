const startBtn = document.getElementById('start-btn');
const video = document.getElementById('webcam-video');
const canvas = document.getElementById('capture-canvas');
const ctx = canvas.getContext('2d');
const processedFeed = document.getElementById('processed-feed');
const cameraOverlay = document.getElementById('camera-overlay');
const repCount = document.getElementById('rep-count');
const connectionStatus = document.getElementById('connection-status');
const connectionText = document.getElementById('connection-text');

let ws;
let isStreaming = false;

// Setup WebSocket connection
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        connectionStatus.classList.remove('disconnected');
        connectionStatus.classList.add('connected');
        connectionText.textContent = 'Connected';
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // Update image
        if (data.image) {
            processedFeed.src = data.image;
        }
        
        // Update count
        if (data.count !== undefined) {
            repCount.textContent = data.count;
        }
    };

    ws.onclose = () => {
        connectionStatus.classList.remove('connected');
        connectionStatus.classList.add('disconnected');
        connectionText.textContent = 'Disconnected';
        
        // Try to reconnect if still streaming
        if (isStreaming) {
            setTimeout(connectWebSocket, 1000);
        }
    };
    
    ws.onerror = (err) => {
        console.error('WebSocket Error:', err);
    };
}

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        });
        
        video.srcObject = stream;
        
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            cameraOverlay.style.display = 'none';
            processedFeed.style.display = 'block';
            
            isStreaming = true;
            connectWebSocket();
            
            // Start sending frames
            sendFrames();
        };
    } catch (err) {
        console.error('Error accessing webcam:', err);
        alert('Could not access webcam. Please ensure permissions are granted.');
    }
}

function sendFrames() {
    if (!isStreaming) return;
    
    // Check if websocket is open
    if (ws && ws.readyState === WebSocket.OPEN) {
        // Draw video frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Get base64 string
        // We use JPEG for better performance over network
        const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
        
        // Send to server
        ws.send(dataUrl);
    }
    
    // Request next frame (throttled to ~15 fps to reduce load)
    setTimeout(() => {
        requestAnimationFrame(sendFrames);
    }, 1000 / 15);
}

startBtn.addEventListener('click', startCamera);
