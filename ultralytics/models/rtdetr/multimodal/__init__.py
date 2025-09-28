# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
RT-DETR MultiModal module for multi-modal object detection.

This module provides multi-modal extensions for RT-DETR, including specialized
trainers, validators, and predictors for handling RGB+X modality inputs.
"""

from .predict import RTDETRMMPredictor
from .train import RTDETRMMTrainer
from .val import RTDETRMMValidator

__all__ = ["RTDETRMMTrainer", "RTDETRMMValidator", "RTDETRMMPredictor"]