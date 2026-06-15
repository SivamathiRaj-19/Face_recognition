import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import threading
import queue
import numpy as np
import cv2
import signal
import sys

# Hailo Python bindings
import hailo

RTSP_URL = "rtsp://admin:Triton123@192.168.1.103/stream1"
HEF1_PATH = "./models/scrfd_500m.hef"
SO1_PATH = "/path/to/libyolo10_post.so"
CROPPER_PATH = "/path/to/libwhole_buffer.so"
SO2_PATH = "/path/to/libface_recognition_post.so"
HEF2_PATH = "./models/arcface_mobilefacenet-2.hef"
Gst.init(None)

###############################################################################
# Thread-safe queues
###############################################################################

embedding_queue = queue.Queue(maxsize=256)

###############################################################################
# GStreamer Pipeline
###############################################################################

PIPELINE = f"""
rtspsrc location = {RTSP_URL} latency = 100 protocols = tcp !
decodebin !
videoconvert !
videoscale !
video/x-raw,format=RGB,width=640,height=640,pixel-aspect-ratio=1/1 !
queue max-size-buffers=5 leaky=downstream !
hailonet hef-path={HEF1_PATH} batch-size=1 scheduling-algorithm=1 !
queue max-size-buffers=5 !
hailofilter so-path={SO1_PATH} !
queue max-size-buffers=5 !
hailotracker keep-new-frames=5 keep-tracked-frames=5 class-id=0 !
queue max-size-buffers=5 !
hailocropper name=facecropper
    so-path={CROPPER_PATH}
    internal-offset=true !

queue max-size-buffers=10 !
hailonet hef-path={HEF2_PATH} batch-size=1 scheduling-algorithm=1 !
queue max-size-buffers=10 !
hailofilter so-path={SO2_PATH} !
queue max-size-buffers=20 !

appsink
    name=appsink_embeddings
    emit-signals=true
    sync=false
    max-buffers=30
    drop=true
"""

###############################################################################
# Create Pipeline
###############################################################################

pipeline = Gst.parse_launch(PIPELINE)

appsink = pipeline.get_by_name("appsink_embeddings")

###############################################################################
# Helper Functions
###############################################################################

def gst_buffer_to_numpy(buffer, caps):
    """
    Convert GstBuffer -> numpy frame safely
    """

    structure = caps.get_structure(0)

    width = structure.get_value("width")
    height = structure.get_value("height")

    success, map_info = buffer.map(Gst.MapFlags.READ)

    if not success:
        return None

    try:
        frame = np.ndarray(
            shape=(height, width, 3),
            dtype=np.uint8,
            buffer=map_info.data
        )

        return frame.copy()

    finally:
        buffer.unmap(map_info)

###############################################################################
# Metadata Extraction
###############################################################################

def extract_embeddings(gst_buffer):

    embeddings_output = []

    roi = hailo.get_roi_from_buffer(gst_buffer)

    if roi is None:
        return embeddings_output

    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    for detection in detections:

        bbox = detection.get_bbox()

        track_id = -1

        tracking_objects = detection.get_objects_typed(
            hailo.HAILO_UNIQUE_ID
        )

        if len(tracking_objects) > 0:
            track_id = tracking_objects[0].get_id()

        matrices = detection.get_objects_typed(hailo.HAILO_MATRIX)

        for matrix in matrices:

            embedding = np.array(
                matrix.get_data(),
                dtype=np.float32
            )

            embedding = embedding.flatten()

            embeddings_output.append({
                "track_id": track_id,
                "bbox": {
                    "xmin": bbox.xmin(),
                    "ymin": bbox.ymin(),
                    "width": bbox.width(),
                    "height": bbox.height()
                },
                "embedding": embedding
            })

    return embeddings_output

###############################################################################
# AppSink Callback
###############################################################################

def on_new_sample(sink):

    sample = sink.emit("pull-sample")

    if sample is None:
        return Gst.FlowReturn.ERROR

    buffer = sample.get_buffer()
    caps = sample.get_caps()

    frame = gst_buffer_to_numpy(buffer, caps)

    if frame is None:
        return Gst.FlowReturn.OK

    embeddings = extract_embeddings(buffer)

    try:
        embedding_queue.put_nowait({
            "frame": frame,
            "embeddings": embeddings
        })

    except queue.Full:
        pass

    return Gst.FlowReturn.OK

appsink.connect("new-sample", on_new_sample)

###############################################################################
# Worker Thread
###############################################################################

running = True

def embedding_worker():

    while running:

        try:
            item = embedding_queue.get(timeout=1)

        except queue.Empty:
            continue

        frame = item["frame"]
        embeddings = item["embeddings"]

        for emb_data in embeddings:

            track_id = emb_data["track_id"]
            embedding = emb_data["embedding"]

            print(
                f"[TRACK {track_id}] "
                f"Embedding shape: {embedding.shape}"
            )

            #
            # Example:
            # send to FAISS / Qdrant / database
            #

        embedding_queue.task_done()

###############################################################################
# Bus Handler
###############################################################################

loop = GLib.MainLoop()

def bus_call(bus, message, loop):

    t = message.type

    if t == Gst.MessageType.EOS:
        loop.quit()

    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"ERROR: {err}")
        print(debug)
        loop.quit()

    return True

bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", bus_call, loop)

###############################################################################
# Shutdown Handling
###############################################################################

def shutdown(sig, frame):

    global running

    running = False

    pipeline.send_event(Gst.Event.new_eos())

signal.signal(signal.SIGINT, shutdown)

###############################################################################
# Start Threads
###############################################################################

worker_thread = threading.Thread(
    target=embedding_worker,
    daemon=True
)

worker_thread.start()

###############################################################################
# Start Pipeline
###############################################################################

pipeline.set_state(Gst.State.PLAYING)

try:
    loop.run()

except KeyboardInterrupt:
    pass

pipeline.set_state(Gst.State.NULL)