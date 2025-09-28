# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "8.3.163"

import os

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

# Import standard models
from ultralytics.models import NAS, RTDETR, SAM, YOLO, YOLOE, FastSAM, YOLOWorld

# Import YOLOMM (conditional)
try:
    from ultralytics.models.yolo.model import YOLOMM
    _YOLOMM_AVAILABLE = True
except ImportError:
    _YOLOMM_AVAILABLE = False
    YOLOMM = None  # Set to None if not available

# Import RTDETRMM (conditional)
try:
    from ultralytics.models.rtdetr.model import RTDETRMM
    _RTDETRMM_AVAILABLE = True
except ImportError:
    _RTDETRMM_AVAILABLE = False
    RTDETRMM = None  # Set to None if not available
from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

settings = SETTINGS

# Dynamic __all__ based on availability
__all__ = [
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "YOLOE",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
]

if _YOLOMM_AVAILABLE:
    __all__.append("YOLOMM")
    
if _RTDETRMM_AVAILABLE:
    __all__.append("RTDETRMM")

__all__ = tuple(__all__)
