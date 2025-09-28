# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Interface for Baidu's RT-DETR, a Vision Transformer-based real-time object detector.

RT-DETR offers real-time performance and high accuracy, excelling in accelerated backends like CUDA with TensorRT.
It features an efficient hybrid encoder and IoU-aware query selection for enhanced detection accuracy.

References:
    https://arxiv.org/pdf/2304.08069.pdf
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.models.yolo.multimodal.visualize.exceptions import (
    LayerNotSpecifiedError, EmptyLayersError, InvalidLayerIndexError
)
from ultralytics.models.yolo.multimodal.visualize.utils import load_image
from ultralytics.models.yolo.multimodal.modal_filling import generate_modality_filling

from .cocoval import RTDETRMMCOCOValidator
from .predict import RTDETRPredictor
from .train import RTDETRTrainer
from .val import RTDETRValidator


class RTDETR(Model):
    """
    Interface for Baidu's RT-DETR model, a Vision Transformer-based real-time object detector.

    This model provides real-time performance with high accuracy. It supports efficient hybrid encoding, IoU-aware
    query selection, and adaptable inference speed.

    Attributes:
        model (str): Path to the pre-trained model.

    Methods:
        task_map: Return a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

    Examples:
        Initialize RT-DETR with a pre-trained model
        >>> from ultralytics import RTDETR
        >>> model = RTDETR("rtdetr-l.pt")
        >>> results = model("image.jpg")
    """

    def __init__(self, model: str = "rtdetr-l.pt") -> None:
        """
        Initialize the RT-DETR model with the given pre-trained model file.

        Args:
            model (str): Path to the pre-trained model. Supports .pt, .yaml, and .yml formats.
        """
        super().__init__(model=model, task="detect")

    @property
    def task_map(self) -> dict:
        """
        Return a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

        Returns:
            (dict): A dictionary mapping task names to Ultralytics task classes for the RT-DETR model.
        """
        return {
            "detect": {
                "predictor": RTDETRPredictor,
                "validator": RTDETRValidator,
                "trainer": RTDETRTrainer,
                "model": RTDETRDetectionModel,
            }
        }

    def cocoval(self, **kwargs):
        """
        Run COCO evaluation using RTDETRMMCOCOValidator.

        Args:
            **kwargs: Additional arguments to pass to the validator.

        Returns:
            dict: Validation metrics from the COCO evaluation.
        """
        from .cocoval import RTDETRMMCOCOValidator
        
        self._check_is_pytorch_model()
        
        args = {**self.overrides, **{"rect": True, "conf": 0.05}, **kwargs, **{"mode": "val"}}
        validator = RTDETRMMCOCOValidator(
            dataloader=None,
            save_dir=None,
            pbar=None,
            args=args,
            _callbacks=self.callbacks
        )
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics


class RTDETRMM(RTDETR):
    """
    RT-DETR MultiModal (RTDETRMM) object detection model.

    RTDETRMM extends the RT-DETR architecture to support multi-modal input (RGB + X modality) for enhanced
    object detection performance. It supports flexible channel configurations and automatic modality
    routing for RGB, X, and Dual modality inputs.

    Attributes:
        model: The loaded RTDETRMM model instance.
        task: The task type (detect).
        overrides: Configuration overrides for the model.
        input_channels: Number of input channels (3 for RGB-only, 6 for RGB+X).
        modality_config: Configuration for supported modalities.
        is_multimodal: Whether the model is configured for multi-modal operation.

    Methods:
        __init__: Initialize RTDETRMM model with multi-modal configuration.
        task_map: Map tasks to their corresponding multi-modal model, trainer, validator, and predictor classes.
        validate_input_channels: Validate input channels against model configuration.
        get_modality_info: Get information about supported modalities.

    Examples:
        Load a RTDETRMM detection model
        >>> model = RTDETRMM("rtdetr-r18-mm.yaml")

        Load with specific channel configuration
        >>> model = RTDETRMM("rtdetr-r34-mm.yaml", ch=6)  # RGB+X modality

        RGB-only mode
        >>> model = RTDETRMM("rtdetr-r50-mm.yaml", ch=3)  # RGB-only
    """

    def __init__(self, model: Union[str, Path] = "rtdetr-r18-mm.pt", ch: Optional[int] = None, 
                 verbose: bool = False) -> None:
        """
        Initialize RTDETRMM multi-modal model.

        Args:
            model (str | Path): Model name or path to model file, i.e. 'rtdetr-r18-mm.yaml', 'rtdetr-r50-mm.pt'.
            ch (int, optional): Number of input channels. If None, auto-detected from model config.
                Supported values: 3 (RGB-only), 6 (RGB+X).
            verbose (bool): Display model info on load.

        Examples:
            >>> model = RTDETRMM("rtdetr-r18-mm.yaml")  # Auto-detect channels
            >>> model = RTDETRMM("rtdetr-r34-mm.yaml", ch=6)  # RGB+X modality
            >>> model = RTDETRMM("rtdetr-r50-mm.yaml", ch=3)  # RGB-only mode
        """
        # Store multi-modal specific attributes
        self.input_channels = ch
        self.modality_config = {}
        # 强制启用多模态模式 - RTDETRMM设计为专用多模态模型
        self.is_multimodal = True
        
        # Check model file name for warning (but don't affect is_multimodal)
        path = Path(model if isinstance(model, (str, Path)) else "")
        if "-mm" not in path.stem:
            if verbose:
                print(f"⚠️ 警告: 模型文件 '{path.stem}' 不包含 '-mm' 后缀，但RTDETRMM已强制启用多模态模式")
                print(f"   请确保使用的是多模态训练的权重文件，否则可能出现维度不匹配")
        elif verbose:
            print(f"✅ 检测到多模态RTDETR模型: {path.stem}")
        
        # Initialize base RTDETR model
        super().__init__(model=model)
        
        # Configure multi-modal settings (always configure for RTDETRMM)
        self._configure_multimodal_settings(verbose)
        
        # Ensure mm_router exists for multimodal operation
        self._ensure_mm_router(verbose)
    
    def _configure_multimodal_settings(self, verbose: bool = False) -> None:
        """
        Configure multi-modal settings based on model configuration.

        Args:
            verbose (bool): Display configuration info.
        """
        try:
            # Get model configuration
            if hasattr(self.model, 'yaml') and self.model.yaml:
                model_yaml = self.model.yaml
                
                # Check for multimodal layers in configuration
                has_multimodal_layers = self._detect_multimodal_layers(model_yaml)
                
                # Determine input channels from model configuration
                model_channels = model_yaml.get('ch', model_yaml.get('channels', 3))
                
                # If multimodal layers detected, determine channel count
                if has_multimodal_layers:
                    # Check for Dual modality layers (6 channels)
                    has_dual_layers = self._has_dual_modality_layers(model_yaml)
                    if has_dual_layers:
                        model_channels = 6
                    else:
                        model_channels = 3  # RGB or X only
                
                # Validate input channels
                if self.input_channels is None:
                    self.input_channels = model_channels
                    if verbose:
                        print(f"Auto-detected input channels: {self.input_channels}")
                elif self.input_channels != model_channels:
                    if verbose:
                        print(f"Warning: Specified channels ({self.input_channels}) differ from model config ({model_channels})")
                
                # Validate channel configuration
                self.validate_input_channels()
                
                # Configure modality information based on detected multimodal layers
                if has_multimodal_layers:
                    if self.input_channels == 6:
                        self.modality_config.update({
                            'rgb_channels': [0, 1, 2],
                            'x_channels': [3, 4, 5],
                            'supported_modalities': ['RGB', 'X', 'Dual'],
                            'default_modality': 'Dual'
                        })
                    else:
                        self.modality_config.update({
                            'rgb_channels': [0, 1, 2],
                            'x_channels': [3, 4, 5],
                            'supported_modalities': ['RGB', 'X'],
                            'default_modality': 'RGB'
                        })
                else:
                    self.modality_config.update({
                        'rgb_channels': [0, 1, 2],
                        'x_channels': [],
                        'supported_modalities': ['RGB'],
                        'default_modality': 'RGB'
                    })
                
                if verbose and self.modality_config:
                    print(f"RTDETRMM configured: {self.input_channels} channels, "
                          f"modalities: {self.modality_config.get('supported_modalities', [])}")
                    
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to configure multi-modal settings: {e}")
            # Set default configuration
            self.input_channels = self.input_channels or 3
            self.modality_config = {
                'supported_modalities': ['RGB'],
                'default_modality': 'RGB'
            }
    
    def _detect_multimodal_layers(self, model_yaml: dict) -> bool:
        """
        Detect if the model configuration contains multimodal layers.

        Args:
            model_yaml (dict): Model YAML configuration

        Returns:
            bool: True if multimodal layers detected
        """
        all_layers = model_yaml.get('backbone', []) + model_yaml.get('head', [])
        
        for layer_config in all_layers:
            if len(layer_config) >= 5:
                input_source = layer_config[4]
                if input_source in ['RGB', 'X', 'Dual']:
                    return True
        return False
    
    def _has_dual_modality_layers(self, model_yaml: dict) -> bool:
        """
        Check if the model configuration has Dual modality layers.

        Args:
            model_yaml (dict): Model YAML configuration

        Returns:
            bool: True if Dual modality layers found
        """
        all_layers = model_yaml.get('backbone', []) + model_yaml.get('head', [])
        
        for layer_config in all_layers:
            if len(layer_config) >= 5:
                input_source = layer_config[4]
                if input_source == 'Dual':
                    return True
        return False
    
    def validate_input_channels(self) -> None:
        """
        Validate input channels against supported configurations.

        Raises:
            ValueError: If input channels are not supported.
        """
        supported_channels = [3, 6]
        if self.input_channels not in supported_channels:
            raise ValueError(
                f"Unsupported input channels: {self.input_channels}. "
                f"Supported channels: {supported_channels} "
                f"(3=RGB-only, 6=RGB+X)"
            )
    
    def _ensure_mm_router(self, verbose: bool = False) -> None:
        """
        Ensure the model has mm_router for multimodal operation.
        
        This is necessary because models loaded from weights may not have mm_router initialized.
        
        Args:
            verbose (bool): Display debug information.
        """
        if hasattr(self, 'model'):
            # Check if the RTDETRDetectionModel has mm_router
            if not hasattr(self.model, 'mm_router') or self.model.mm_router is None:
                try:
                    from ultralytics.nn.mm import MultiModalRouter
                    # Create mm_router with model configuration
                    config_dict = getattr(self.model, 'yaml', None)
                    self.model.mm_router = MultiModalRouter(config_dict, verbose=verbose)
                    if verbose:
                        print(f"✅ RTDETRMM: 为RTDETRDetectionModel创建了MultiModalRouter")
                except Exception as e:
                    if verbose:
                        print(f"⚠️ RTDETRMM: 无法创建MultiModalRouter: {e}")
                        print(f"   这可能会影响多模态推理功能")
    
    def get_modality_info(self) -> Dict[str, Any]:
        """
        Get information about supported modalities and configuration.

        Returns:
            dict: Modality configuration information.
        """
        return {
            'input_channels': self.input_channels,
            'modality_config': self.modality_config.copy(),
            'model_type': 'RTDETRMM',
            'task': getattr(self, 'task', 'detect'),
            'is_multimodal': self.is_multimodal
        }
    
    @property
    def task_map(self) -> dict:
        """
        Return a task map for RTDETRMM, associating tasks with corresponding Ultralytics classes.

        Returns:
            (dict): A dictionary mapping task names to Ultralytics task classes for the RTDETRMM model.
        """
        if self.is_multimodal:
            try:
                # Import multi-modal components (only if available)
                from ultralytics.models.rtdetr.multimodal import (
                    RTDETRMMTrainer,
                    RTDETRMMValidator,
                    RTDETRMMPredictor
                )
                
                return {
                    "detect": {
                        "predictor": RTDETRMMPredictor,
                        "validator": RTDETRMMValidator,
                        "trainer": RTDETRMMTrainer,
                        "model": RTDETRDetectionModel,  # Use standard model with multi-modal routing
                    }
                }
            except ImportError as e:
                # If multi-modal components are not available, fall back to standard RTDETR
                if hasattr(self, 'verbose') and self.verbose:
                    print(f"Warning: Multi-modal components not available ({e}), using standard RTDETR components")
                return super().task_map
        else:
            # For non-multimodal models, use parent's task_map
            return super().task_map
    
    def vis(self, 
            source: Union[str, List[str], np.ndarray, List[np.ndarray]], 
            method: str = "heatmap",
            layers: Optional[List[int]] = None,  # 新增一级参数
            alg: str = 'gradcam',
            modality: Optional[str] = None,
            save: bool = True,
            project: str = "runs/visualize",
            name: str = "exp",
            **kwargs) -> Union['VisualizationResult', List['VisualizationResult']]:
        """
        Performs visualization tasks on a multi-modal model, supporting various input formats.

        This method serves as a unified API endpoint for all visualization tasks,
        aligning with the input style of the `predict` method for consistency. It
        pre-processes dual-modal and single-modal inputs into a standardized
        dictionary format before passing them to the backend visualization manager.

        Args:
            source (Union[str, List[str], np.ndarray, List[np.ndarray]]): The input source for visualization. Can be:
                - A list of two paths for dual-modal input, e.g., `['path/to/rgb.jpg', 'path/to/x.jpg']`.
                - A single path for single-modal input, e.g., `'path/to/image.jpg'`. Requires `modality`.
            method (str): The visualization method, e.g., 'heatmap', 'feature_map'. Defaults to 'heatmap'.
            layers (List[int]): List of layer indices to visualize. Must be specified.
            alg (str): The specific algorithm for the method, e.g., 'gradcam'. Defaults to 'gradcam'.
            modality (str, optional): Specifies the modality for single-source inputs.
                Must be 'rgb' or 'x'. Required when `source` is a single path. Defaults to None.
            save (bool): Whether to save visualization results. Defaults to True.
            project (str): The project directory to save results. Defaults to 'runs/visualize'.
            name (str): The experiment name for the save directory. Defaults to 'exp'.
            **kwargs: Additional arguments for the visualization backend.

        Returns:
            Union[VisualizationResult, List[VisualizationResult]]: The result from the visualization backend.
                Returns a list for multi-layer visualization, single result for single layer.

        Raises:
            ValueError: If the input arguments are inconsistent or unsupported.
            TypeError: If layers parameter is not a list of integers.
        """
        # Import visualization exceptions
        from ultralytics.models.yolo.multimodal.visualize.exceptions import (
            LayerNotSpecifiedError, EmptyLayersError, InvalidLayerIndexError
        )
        
        # --- Layers Parameter Validation ---
        # 检查layers参数
        if layers is None:
            # 使用专门的错误类，提供模型层数信息
            raise LayerNotSpecifiedError(model_layers=len(self.model.model))
        
        if not isinstance(layers, list):
            raise TypeError(f"layers must be a list of integers, got {type(layers)}")
        
        if len(layers) == 0:
            # 使用专门的空列表错误类
            raise EmptyLayersError()
        
        # 验证并处理层索引
        valid_layers = []
        invalid_indices = []
        
        for idx in layers:
            if not isinstance(idx, int):
                raise TypeError(f"Layer index must be integer, got {type(idx)} for {idx}")
            if idx < 0 or idx >= len(self.model.model):
                invalid_indices.append(idx)
            else:
                # 检查重复
                if idx not in valid_layers:
                    valid_layers.append(idx)
                else:
                    from ultralytics.utils import LOGGER
                    LOGGER.warning(f"Duplicate layer index {idx} will be ignored")
        
        # 如果有无效索引，抛出错误
        if invalid_indices:
            raise InvalidLayerIndexError(
                invalid_indices=invalid_indices, 
                valid_range=(0, len(self.model.model)-1)
            )
        
        # 检查层类型并发出警告（可选）
        for idx in valid_layers:
            layer_type = self.model.model[idx].__class__.__name__
            if layer_type in ['BatchNorm2d', 'Dropout', 'Upsample']:
                from ultralytics.utils import LOGGER
                LOGGER.warning(f"Layer {idx} ({layer_type}) may not produce meaningful heatmaps")
        
        # 将索引转换为内部使用的层名称格式
        layer_names = [str(idx) for idx in valid_layers]

        # --- Input Preprocessor Logic ---
        processed_source = None

        # 1. Handle dual-modal input (list of two paths)
        if isinstance(source, list):
            if len(source) == 2:
                source_dict = {'rgb': source[0], 'x': source[1]}
                # Load images from paths
                processed_source = {k: load_image(v) for k, v in source_dict.items()}
            elif len(source) == 1 and modality is not None:
                # Single-modal input with list format
                single_path = source[0]
                if not Path(single_path).exists():
                    raise FileNotFoundError(f"The specified image source '{single_path}' does not exist.")
                    
                # Load the provided image
                real_image_np = load_image(single_path)
                
                # Convert to tensor for the filler function
                real_image_tensor = torch.from_numpy(real_image_np).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

                # Generate the filled modality
                if modality == 'rgb':
                    filled_image_tensor = generate_modality_filling(real_image_tensor, 'rgb', 'x')
                    processed_source = {'rgb': real_image_np, 'x': filled_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()}
                elif modality == 'x':
                    filled_image_tensor = generate_modality_filling(real_image_tensor, 'x', 'rgb')
                    processed_source = {'rgb': filled_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy(), 'x': real_image_np}
                else:
                    raise ValueError(f"Unsupported modality '{modality}'. Please choose 'rgb' or 'x'.")
            else:
                if len(source) == 1 and modality is None:
                    raise ValueError("For single-modal input (list with one element), 'modality' parameter must be specified.")
                else:
                    raise ValueError(f"Source list must contain 1 or 2 paths, but got {len(source)} elements.")

        # 2. Handle single-modal input (single path with modality specified)
        elif isinstance(source, str) and modality is not None:
            if not Path(source).exists():
                raise FileNotFoundError(f"The specified image source '{source}' does not exist.")
                
            # Load the provided image
            real_image_np = load_image(source)
            h, w, _ = real_image_np.shape
            
            # Convert to tensor for the filler function
            real_image_tensor = torch.from_numpy(real_image_np).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

            # Generate the filled modality
            if modality == 'rgb':
                filled_image_tensor = generate_modality_filling(real_image_tensor, 'rgb', 'x')
                processed_source = {'rgb': real_image_np, 'x': filled_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()}
            elif modality == 'x':
                filled_image_tensor = generate_modality_filling(real_image_tensor, 'x', 'rgb')
                processed_source = {'rgb': filled_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy(), 'x': real_image_np}
            else:
                raise ValueError(f"Unsupported modality '{modality}'. Please choose 'rgb' or 'x'.")
        
        # 3. Handle other direct inputs (like a pre-made dictionary) or raise error
        elif isinstance(source, dict):
            # Check if values are paths that need loading
            if all(isinstance(v, str) for v in source.values()):
                processed_source = {k: load_image(v) for k, v in source.items()}
            else:
                processed_source = source  # Assume values are already loaded images
        else:
            raise ValueError(
                "Invalid input for `vis`. Please provide either:\n"
                "1. A list of two image paths for dual-modal visualization.\n"
                "2. A single image path (string) with the `modality` argument ('rgb' or 'x')."
            )

        # --- Backend Call ---
        self.model.eval()
        
        # Dynamically import to avoid circular dependencies
        from ultralytics.models.yolo.multimodal.visualize import VisualizationManager

        # Create VisualizationManager with project and name
        manager = VisualizationManager(model=self.model, project=project, name=name)
        
        # Pass all visualization parameters explicitly
        return manager.visualize(
            source=processed_source, 
            method=method,
            layers=layer_names,  # 使用转换后的层名称
            alg=alg,
            save=save,
            **kwargs
        )
