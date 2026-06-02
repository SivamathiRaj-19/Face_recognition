import gi
gi.require_version('Gst','1.0')
gi.require_version('GObject','2.0')
from gi.repository import Gst, GLib

Gst.init(None)

RTSP_URL = "rtsp://admin:Triton123@192.168.1.103:554/Streaming/Channels/101"
HEF_PATH = "/home/mandlacx/intern/Face_recognition/models/yolov11n-face.hef"
PP_PATH ="/home/mandlacx/intern/Face_recognition/models/libyolo_hailortpp_postprocess.so"
HEF2_PATH = ""
PP2_PATH =""

pipeline_str = f"""
rtspsrc location={RTSP_URL} latency=100 protocols=tcp !
rtph264depay !
decodebin !
videoconvert !
videoscale !
video/x-raw, width=640, height=640, format=RGB  !
queue !
hailonet hef-path={HEF_PATH} !
queue !
hailofilter so-path={PP_PATH} qos=false !
queue !
hailooverlay !
videoconvert !
autovideosink  sync=false
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
