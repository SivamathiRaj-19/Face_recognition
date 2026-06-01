import cv2

rtsp_url = "rtsp://admin:Tiandy%40123@192.168.1.103/stream1"

def get_camera(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"❌ Failed to connect to camera {rtsp_url}")
        return None
    return cap

def get_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    frame = cv2.resize(frame,(320, 320))
    return frame
