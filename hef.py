import gi
gi.require_version('Gst','1.0')
gi.require_version('GObject','2.0')
from gi.repository import Gst, GLib

Gst.init(None)

RTSP_URL = "rtsp://admin:Tiandy%40123@192.168.1.103:554/Streaming/Channels/101"

# PATHS (Ensure these paths match your environment installations)
DET_HEF = "./models/retinaface_mobilenet_v1.hef" 
DET_PP  = "./models/libface_detection_post.so" # Added face detection post-process
REC_HEF = "./models/arcface_mobilefacenet-2.hef"
REC_PP  = "./models/libface_recognition_post.so"

pipeline_str = f"""
rtspsrc location={RTSP_URL} latency=100 protocols=udp !
rtph264depay !
decodebin !
videoconvert !
videoscale !
video/x-raw,width=640,height=640 !
queue name=prim_convert_q_sink !

hailonet hef-path={DET_HEF} nms-score-threshold=0.01 nms-iou-threshold=0.03 output-format-type=HAILO_FORMAT_TYPE_FLOAT32 !
queue !
hailofilter so-path={DET_PP} qos=false !
queue !

hailocropper name=cropper 
    hailomuxer name=muxer

cropper.src_0 ! 
    queue name=hailo_face_rec_q ! 
    hailonet hef-path={REC_HEF} nms-score-threshold=0.01 nms-iou-threshold=0.03 output-format-type=HAILO_FORMAT_TYPE_FLOAT32 !
    queue ! 
    hailofilter so-path={REC_PP} qos=false ! 
    queue ! 
    hailoaggregator ! 
    muxer.sink_0

cropper.src_1 ! 
    queue name=hailo_bypass_q ! 
    muxer.sink_1

muxer.src ! 
    queue !
    hailooverlay !
    videoconvert !
    autovideosink sync=false
"""

pipeline = Gst.parse_launch(pipeline_str)
mainLoop = GLib.MainLoop()
pipeline.set_state(Gst.State.PLAYING)
print("Pipeline stream is running safely...")

try:
    mainLoop.run()
except KeyboardInterrupt:
    mainLoop.quit()
finally:
    pipeline.set_state(Gst.State.NULL)