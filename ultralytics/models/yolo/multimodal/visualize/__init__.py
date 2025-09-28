# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Visualization module for YOLO multimodal models.

This module provides comprehensive tools for model interpretability and visualization,
including heatmap generation and feature map visualization capabilities. It helps users
understand how multimodal YOLO models process and fuse information from different input
modalities (RGB and depth/thermal/other).

Key Components:
    - VisualizationManager: Central orchestrator for all visualization tasks
    - HeatmapVisualizer: Generates Grad-CAM based heatmaps for model interpretability
    - FeatureMapVisualizer: Visualizes intermediate feature maps from model layers
    - HeatmapResult: Data class containing heatmap visualization results
    - FeatureMapResult: Data class containing feature map visualization results

Example:
    ```python
    from ultralytics.models.yolo.multimodal.visualize import VisualizationManager
    from ultralytics import YOLOMM
    
    # Initialize model and visualization manager
    model = YOLOMM('yolo11n-mm.yaml')
    model.load('path/to/weights.pt')
    vis_manager = VisualizationManager(model.model)
    
    # Generate heatmap for an image
    heatmap_result = vis_manager.generate_heatmap(
        'path/to/image.jpg',
        method='gradcam',
        target_layer='model.model.9'
    )
    
    # Visualize feature maps
    feature_result = vis_manager.visualize_features(
        'path/to/image.jpg',
        layers=['model.model.4', 'model.model.6']
    )
    ```
"""

from .exceptions import EmptyLayersError, InvalidLayerIndexError, LayerNotSpecifiedError
from .feature import FeatureMapVisualizer
from .heatmap import HeatmapVisualizer
from .manager import FeatureMapResult, HeatmapResult, VisualizationManager

__all__ = (
    "VisualizationManager",
    "HeatmapVisualizer",
    "FeatureMapVisualizer",
    "HeatmapResult",
    "FeatureMapResult",
    "LayerNotSpecifiedError",
    "EmptyLayersError",
    "InvalidLayerIndexError",
)