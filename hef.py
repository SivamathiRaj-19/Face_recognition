import gi
gi.require_version('Gst','1.0')
gi.require_version('GObject','2.0')
from gi.repository import Gst, GLib

Gst.init(None)

RTSP_URL="rtsp://admin:Tiandy%40123@192.168.1.103:554/Streaming/Channels/101"
HEF_PATH = "./models/yolov11n.hef" 
PP_PATH = "./models/libyolo_hailortpp_postprocess.so"
F_EXTRACTION= "./models/arcface_mobilefacenet-2.hef"



pipeline_str = f"""
rtspsrc location={RTSP_URL} latency=100 protocols=udp !
rtph264depay !
decodebin !
videoconvert !
videoscale !
video/x-raw,width=640,height=640 !
queue !
hailonet hef-path={HEF_PATH} nms-score-threshold=0.01 nms-iou-threshold=0.03 output-format-type=HAILO_FORMAT_TYPE_FLOAT32 !
queue !
hailofilter so-path={PP_PATH} qos=false !
queue !
hailooverlay !
videoconvert !
autovideosink  sync=false
"""

pipeline=Gst.parse_launch(pipeline_str)
mainLoop=GLib.MainLoop()
pipeline.set_state(Gst.State.PLAYING)
print("pipeline stream is running")
try:
    mainLoop.run()
except KeyboardInterrupt:
    mainLoop.quit()
finally:
    pipeline.set_state(Gst.State.NULL)
