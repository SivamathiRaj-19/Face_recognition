import cv2

RTSP_URL = "rtsp://admin:Triton123@192.168.1.103/stream1"

FRAME_RESIZE = (640, 640)

def stream_camera(rtsp_url: str):
    global frame
    print(f"Trying camera: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    print(f"Connection established: {rtsp_url}")
    if not cap.isOpened():
        print(f"❌ Failed to connect to camera {rtsp_url}")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("⚠️ Failed to grab frame. Stream might have disconnected.")
            break

        frame = cv2.resize(frame, FRAME_RESIZE)

        cv2.imshow("Triton Smart Locker", frame)
        

    cap.release()

pipelin_str = f"""
    rtspsrc location = {RTSP_URL} latency = 100 protocols = tcp !
    queue !
    rtph264depay !
    h264parse !
    """