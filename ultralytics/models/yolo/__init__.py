# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models.yolo import classify, detect, obb, pose, segment, world, yoloe

from .model import YOLO, YOLOE, YOLOWorld

# Import YOLOMM conditionally
try:
    from .model import YOLOMM
    _YOLOMM_AVAILABLE = True
except ImportError:
    _YOLOMM_AVAILABLE = False

if _YOLOMM_AVAILABLE:
    __all__ = "classify", "segment", "detect", "pose", "obb", "world", "yoloe", "YOLO", "YOLOWorld", "YOLOE", "YOLOMM"
else:
    __all__ = "classify", "segment", "detect", "pose", "obb", "world", "yoloe", "YOLO", "YOLOWorld", "YOLOE"
