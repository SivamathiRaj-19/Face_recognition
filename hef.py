import gi
gi.require_version('Gst','1.0')
gi.require_version('GObject','2.0')
from gi.repository import Gst, GLib

Gst.init(None)

RTSP_URL="rtsp://admin:Tiandy%40123@192.168.1.103:554/Streaming/Channels/101" 
PP_PATH = "./models/libscrfd.so"
HEF_PATH = "./models/scrfd_50m.hef"
FE_PATH = "./models/arcface_mobilefacenet-2.hef"
PO_PATH = "./models/libface_recognition_post.so"
CROPPPER_PATH = "./models/libvms_croppers.so"


pipeline_str = f"""
rtspsrc location={RTSP_URL} latency=100 protocols=tcp !
rtph264depay ! decodebin ! videoconvert ! videoscale !
video/x-raw, width=640, height=640, format=RGB ! 
queue !
hailonet hef-path={HEF_PATH} ! 
queue !
hailofilter so-path={PP_PATH} qos=false ! 
queue !
hailocropper name=crop so-path={CROPPPER_PATH} function-name=Z19algorithm_face_cropjjRK9HailoBBoxffuse-letterbox=true internal-offset=true ! 
hailoaggregator name=agg ! 
queue ! 
hailooverlay ! videoconvert ! autovideosink sync=false
crop. ! queue ! 
video/x-raw, width=112, height=112, format=RGB ! 
hailonet hef-path={FE_PATH} ! 
queue ! 
hailofilter so-path={PO_PATH} qos=false ! 
queue ! 
agg.
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
