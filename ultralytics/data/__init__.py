# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .base import BaseDataset
from .build import build_dataloader, build_grounding, build_yolo_dataset, load_inference_source
from .dataset import (
    ClassificationDataset,
    GroundingDataset,
    SemanticDataset,
    YOLOConcatDataset,
    YOLODataset,
    YOLOMultiModalDataset,
    YOLOMultiModalImageDataset,
)
from .multimodal_augment import (
    BaseMultiModalTransform,
    MultiModalMixUp,
    MultiModalMosaic,
    MultiModalRandomFlip,
    MultiModalRandomHSV,
    create_multimodal_transforms,
)

__all__ = (
    # Base and standard datasets
    "BaseDataset",
    "ClassificationDataset",
    "SemanticDataset",
    "YOLODataset",
    "YOLOConcatDataset",
    "GroundingDataset",

    # Multi-modal datasets
    "YOLOMultiModalDataset",           # Text multi-modal dataset
    "YOLOMultiModalImageDataset",      # RGB+X image multi-modal dataset

    # Multi-modal data augmentation
    "BaseMultiModalTransform",         # Multi-modal transform base class
    "MultiModalRandomHSV",             # Multi-modal HSV augmentation
    "MultiModalRandomFlip",            # Multi-modal flip augmentation
    "MultiModalMosaic",                # Multi-modal mosaic augmentation
    "MultiModalMixUp",                 # Multi-modal mixup augmentation
    "create_multimodal_transforms",    # Multi-modal transform factory

    # Dataset building functions
    "build_yolo_dataset",              # Enhanced with multi_modal_image support
    "build_grounding",
    "build_dataloader",
    "load_inference_source",
)
