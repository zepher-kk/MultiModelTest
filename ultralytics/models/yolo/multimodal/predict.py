# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import numpy as np
import cv2
from pathlib import Path
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr, ops
from ultralytics.data.build import load_inference_source
from ultralytics.utils.plotting import Annotator, colors
from copy import deepcopy

try:
    from .modal_filling import ModalityFiller
except ImportError:
    from ultralytics.models.yolo.multimodal.modal_filling import ModalityFiller


class MultiModalDetectionPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a multimodal detection model.
    
    Supports RGB+X dual-modal inference (best performance), RGB single-modal inference (with X-modal filling),
    and X-modal single-modal inference (with RGB filling) for flexible multimodal object detection.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.multimodal import MultiModalDetectionPredictor

        # Dual-modal inference (best performance)
        args = dict(model="yolo11n-mm.pt", source=[ASSETS / "bus.jpg", ASSETS / "bus_depth.jpg"])
        predictor = MultiModalDetectionPredictor(overrides=args)
        predictor.predict_cli()
        
        # RGB single-modal inference
        args = dict(model="yolo11n-mm.pt", source=ASSETS / "bus.jpg", modality="rgb")
        predictor = MultiModalDetectionPredictor(overrides=args)
        predictor.predict_cli()
        
        # X-modal single-modal inference
        args = dict(model="yolo11n-mm.pt", source=ASSETS / "bus_depth.jpg", modality="depth")
        predictor = MultiModalDetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the MultiModalDetectionPredictor with the provided configuration, overrides, and callbacks.
        
        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
            _callbacks (dict, optional): Dictionary of callback functions. Defaults to None.
        """
        super().__init__(cfg, overrides, _callbacks)
        
        # Get modality parameter from standard cfg system (now natively supported by ultralytics)
        # Modality validation is handled by cfg system, no local validation needed
        self.modality = getattr(self.args, 'modality', None)
        
        # Initialize multimodal-specific attributes
        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None
        
        # Track input sources for multi-modal visualization
        self.rgb_source = None
        self.x_source = None
        self.input_mode = None  # 'dual', 'single_rgb', 'single_x'
        
        # Log initialization
        # 多模态推理器初始化完成

    
    def _parse_inference_input(self, source):
        """
        Parse and validate inference input from YOLOMM.predict() method.
        
        Handles various input formats and validates them against the current modality settings.
        Provides detailed logging about input sources and formats.
        
        Args:
            source: Input source from YOLOMM.predict() method. Can be:
                - Single file path (str/Path): for single-modal inference
                - List of 2 paths: [rgb_path, x_path] for dual-modal inference
                - PIL.Image or np.ndarray: for single image
                - List of PIL.Image/np.ndarray: for batch inference
                - torch.Tensor: preprocessed tensor
                
        Returns:
            tuple: (parsed_source, input_format_info) where:
                - parsed_source: Validated and normalized input source
                - input_format_info: Dict with input analysis information
                
        Raises:
            ValueError: If input format is invalid or incompatible with modality settings
            TypeError: If input type is not supported
        """
        import numpy as np
        from PIL import Image
        from pathlib import Path
        
        # Initialize input format analysis
        input_info = {
            'input_type': type(source).__name__,
            'is_batch': False,
            'source_count': 1,
            'modality_mode': 'dual' if self.is_dual_modal else f'single_{self.modality}',
            'inference_format': None,
            'validation_passed': False
        }
        
        try:
            # Log initial input analysis
            LOGGER.debug(f"解析推理输入: 类型={input_info['input_type']}, 模态模式={input_info['modality_mode']}")
            
            # Case 1: Tensor input (already preprocessed)
            if isinstance(source, torch.Tensor):
                input_info['inference_format'] = 'preprocessed_tensor'
                input_info['tensor_shape'] = list(source.shape)
                
                if source.dim() == 4 and source.shape[1] == 6:
                    LOGGER.debug("检测到6通道预处理tensor，直接使用")
                    input_info['validation_passed'] = True
                    return source, input_info
                else:
                    LOGGER.warning(f"Tensor维度不符合预期: {source.shape}，将重新处理")
            
            # Case 2: List/Tuple input
            elif isinstance(source, (list, tuple)):
                input_info['source_count'] = len(source)
                
                if len(source) == 2 and self.is_dual_modal:
                    # Dual-modal format: [rgb_source, x_source]
                    input_info['inference_format'] = 'dual_modal_list'
                    rgb_source, x_source = source
                    
                    # Validate individual sources
                    rgb_info = self._analyze_single_source(rgb_source, 'rgb')
                    x_info = self._analyze_single_source(x_source, 'x_modal')
                    
                    input_info['rgb_source'] = rgb_info
                    input_info['x_source'] = x_info
                    input_info['validation_passed'] = True
                    
                    # 双模态输入解析完成
                    return source, input_info
                    
                elif len(source) == 1 and self.is_single_modal:
                    # Single-modal with list wrapper
                    input_info['inference_format'] = 'single_modal_list'
                    single_source = source[0]
                    LOGGER.debug(f"单模态输入(列表包装): {type(single_source)}")
                    
                    # Analyze the single source
                    source_info = self._analyze_single_source(single_source, self.modality)
                    input_info.update(source_info)
                    input_info['validation_passed'] = True
                    
                    # 单模态输入解析完成
                    return single_source, input_info
                elif len(source) > 2:
                    # Batch inference support
                    input_info['inference_format'] = 'batch_inference'
                    input_info['is_batch'] = True
                    
                    if self.is_dual_modal:
                        # For dual-modal batch, expect pairs of sources
                        if len(source) % 2 != 0:
                            raise ValueError(f"双模态批量推理需要偶数个输入源，但接收到{len(source)}个")
                        
                        # Parse pairs
                        pairs = [(source[i], source[i+1]) for i in range(0, len(source), 2)]
                        input_info['batch_size'] = len(pairs)
                        LOGGER.info(f"双模态批量推理: {input_info['batch_size']}对图像")
                        input_info['validation_passed'] = True
                        return pairs, input_info
                    else:
                        # Single-modal batch
                        input_info['batch_size'] = len(source)
                        LOGGER.info(f"单模态批量推理: {input_info['batch_size']}张图像")
                        input_info['validation_passed'] = True
                        return source, input_info
                        
                else:
                    # Invalid list format
                    if self.is_dual_modal:
                        raise ValueError(f"双模态推理需要2个输入源，但接收到{len(source)}个")
                    else:
                        # Use first element for single-modal
                        single_source = source[0]
                        LOGGER.warning(f"单模态推理接收到{len(source)}个输入，使用第一个: {single_source}")
                        return self._parse_inference_input(single_source)
            
            # Case 3: Single source input
            else:
                if self.is_dual_modal:
                    raise ValueError(
                        f"双模态推理需要列表格式输入 [rgb_source, x_source]，"
                        f"但接收到单个源: {type(source)}"
                    )
                
                # Single-modal input validation
                input_info['inference_format'] = 'single_modal_source'
                source_info = self._analyze_single_source(source, self.modality)
                input_info.update(source_info)
                input_info['validation_passed'] = True
                
                # 单模态输入解析完成
                return source, input_info
                
        except Exception as e:
            input_info['validation_passed'] = False
            input_info['error'] = str(e)
            LOGGER.error(f"输入解析失败: {e}")
            raise
        
        finally:
            # Log final input analysis
            self._log_input_analysis(input_info)
    
    def _analyze_single_source(self, source, modality_hint=None):
        """
        Analyze a single input source and determine its characteristics.
        
        Args:
            source: Single input source (path, PIL.Image, np.ndarray, etc.)
            modality_hint (str): Hint about expected modality type
            
        Returns:
            dict: Analysis information about the source
        """
        import numpy as np
        from PIL import Image
        from pathlib import Path
        
        analysis = {
            'source_type': 'unknown',
            'path': None,
            'exists': False,
            'format': None,
            'modality_hint': modality_hint
        }
        
        if isinstance(source, (str, Path)):
            # File path
            path = Path(source)
            analysis['source_type'] = 'file_path'
            analysis['path'] = str(path)
            analysis['exists'] = path.exists()
            analysis['format'] = path.suffix.lower() if path.suffix else 'no_extension'
            
            if not analysis['exists']:
                raise FileNotFoundError(f"输入文件不存在: {path}")
                
        elif isinstance(source, Image.Image):
            # PIL Image
            analysis['source_type'] = 'pil_image'
            analysis['format'] = source.format or 'unknown'
            analysis['mode'] = source.mode
            analysis['size'] = source.size
            
        elif isinstance(source, np.ndarray):
            # Numpy array
            analysis['source_type'] = 'numpy_array'
            analysis['shape'] = source.shape
            analysis['dtype'] = str(source.dtype)
            
        elif isinstance(source, torch.Tensor):
            # Tensor
            analysis['source_type'] = 'torch_tensor'
            analysis['shape'] = list(source.shape)
            analysis['dtype'] = str(source.dtype)
            analysis['device'] = str(source.device)
            
        else:
            analysis['source_type'] = f'unsupported_{type(source).__name__}'
            
        return analysis
    
    def _log_input_analysis(self, input_info):
        """
        Log detailed input analysis information.
        
        Args:
            input_info (dict): Input analysis information
        """
        LOGGER.debug("=== 输入解析分析报告 ===")
        LOGGER.debug(f"输入类型: {input_info['input_type']}")
        LOGGER.debug(f"推理格式: {input_info['inference_format']}")
        LOGGER.debug(f"模态模式: {input_info['modality_mode']}")
        LOGGER.debug(f"源数量: {input_info['source_count']}")
        LOGGER.debug(f"批量推理: {input_info['is_batch']}")
        LOGGER.debug(f"验证通过: {input_info['validation_passed']}")
        
        if 'batch_size' in input_info:
            LOGGER.debug(f"批量大小: {input_info['batch_size']}")
            
        if 'rgb_source' in input_info:
            LOGGER.debug(f"RGB源信息: {input_info['rgb_source']}")
            
        if 'x_source' in input_info:
            LOGGER.debug(f"X模态源信息: {input_info['x_source']}")
            
        if 'error' in input_info:
            LOGGER.debug(f"错误信息: {input_info['error']}")
            
        LOGGER.debug("=== 分析报告结束 ===")

    def preprocess(self, im):
        """
        Prepares multimodal input images before inference.
        
        Handles dual-modal input (RGB + X-modal) and single-modal input with intelligent filling.
        Ensures output is always 6-channel tensor compatible with trained multimodal models.
        
        Args:
            im (torch.Tensor | List | str): Input images. Can be:
                - List of 2 paths: [rgb_path, x_path] for dual-modal
                - Single path: rgb_path or x_path for single-modal
                - torch.Tensor: preprocessed tensor
                
        Returns:
            torch.Tensor: 6-channel tensor [X, X, X, RGB, RGB, RGB] format
            
        Raises:
            ValueError: If input format is invalid or incompatible with modality settings
            FileNotFoundError: If image files cannot be found
            RuntimeError: If tensor processing fails
        """
        try:
            # 使用新的输入解析方法
            LOGGER.debug(f"开始多模态预处理: modality={self.modality}, input_type={type(im)}")
            
            # 解析和验证输入
            parsed_source, input_info = self._parse_inference_input(im)
            
            # 快速路径：如果输入已经是正确格式的6通道tensor
            if isinstance(parsed_source, torch.Tensor) and parsed_source.dim() == 4 and parsed_source.shape[1] == 6:
                LOGGER.debug("输入已为6通道tensor，进行格式验证后直接返回")
                return self._finalize_tensor(parsed_source)
            
            # 根据解析结果选择处理路径
            if input_info['inference_format'] in ['dual_modal_list', 'batch_inference'] and self.is_dual_modal:
                result_tensor = self._process_dual_modality(parsed_source)
            elif input_info['inference_format'] in ['single_modal_source', 'single_modal_list'] and self.is_single_modal:
                result_tensor = self._process_single_modality(parsed_source)
            else:
                # 兼容旧的处理方式
                if self.is_dual_modal:
                    result_tensor = self._process_dual_modality(parsed_source)
                else:
                    result_tensor = self._process_single_modality(parsed_source)
            
            # 最终格式验证和设备转换
            final_tensor = self._finalize_tensor(result_tensor)
            
            LOGGER.debug(f"多模态预处理完成: shape={final_tensor.shape}, device={final_tensor.device}")
            return final_tensor
            
        except Exception as e:
            # 统一异常处理
            error_msg = f"多模态预处理失败: {str(e)}"
            LOGGER.error(error_msg)
            self._log_debug_info(im, e)
            raise RuntimeError(error_msg) from e
    
    def _process_dual_modality(self, im):
        """
        Process dual-modal input: [rgb_path, x_path] or similar formats.
        
        Handles various dual-modal input formats and ensures proper 6-channel output
        with channel order [X, X, X, RGB, RGB, RGB] matching training stage.
        
        Args:
            im (List | torch.Tensor): Dual-modal input data
            
        Returns:
            torch.Tensor: 6-channel tensor [X, X, X, RGB, RGB, RGB]
        """
        # If already preprocessed tensor with 6 channels, return directly
        if isinstance(im, torch.Tensor) and im.shape[1] == 6:
            LOGGER.debug("输入已为6通道tensor，直接返回")
            return im
        
        # Parse dual-modal input and load images
        rgb_images, x_images = self._parse_dual_modal_input(im)
        
        # Preprocess each modality separately using parent's method
        rgb_tensor = super().preprocess(rgb_images)  # Shape: (B, 3, H, W)
        x_tensor = super().preprocess(x_images)      # Shape: (B, 3, H, W)
        
        # Ensure same spatial dimensions
        rgb_tensor, x_tensor = self._align_tensor_dimensions(rgb_tensor, x_tensor)
        
        # Combine modalities: [X, X, X, RGB, RGB, RGB] order (matching training)
        combined_tensor = torch.cat([x_tensor, rgb_tensor], dim=1)  # Shape: (B, 6, H, W)
        
        LOGGER.debug(f"双模态预处理完成: {combined_tensor.shape}")
        return combined_tensor
    
    def _parse_dual_modal_input(self, im):
        """
        Parse dual-modal input and separate RGB and X-modal data.
        
        Handles different input formats for dual-modal inference with enhanced integration
        of ultralytics loading mechanisms.
        
        Args:
            im (List | str | Path): Input data - can be:
                - [rgb_source, x_source]: Standard dual-modal format
                - [(rgb1, x1), (rgb2, x2), ...]: Batch of dual-modal pairs
                - Dataset object from load_inference_source
                
        Returns:
            tuple: (rgb_images, x_images) ready for preprocessing
        """
        if isinstance(im, (list, tuple)):
            # Check if it's batch of pairs format
            if len(im) > 2 and all(isinstance(item, (list, tuple)) and len(item) == 2 for item in im):
                # Batch format: [(rgb1, x1), (rgb2, x2), ...]
                LOGGER.debug(f"解析批量双模态输入: {len(im)}对图像")
                
                rgb_sources = []
                x_sources = []
                
                for i, (rgb_source, x_source) in enumerate(im):
                    try:
                        # Use enhanced loading with integration
                        rgb_data, rgb_meta = self._integrate_with_load_inference_source(rgb_source)
                        x_data, x_meta = self._integrate_with_load_inference_source(x_source)
                        
                        rgb_sources.append(rgb_data)
                        x_sources.append(x_data)
                        
                        LOGGER.debug(f"批量[{i}] RGB: {rgb_meta.get('dataset_type', 'direct')}, "
                                   f"X: {x_meta.get('dataset_type', 'direct')}")
                        
                    except Exception as e:
                        LOGGER.error(f"批量双模态输入[{i}]处理失败: {e}")
                        raise
                
                return rgb_sources, x_sources
                
            elif len(im) == 2:
                # Standard dual-modal format: [rgb_source, x_source]
                rgb_source, x_source = im
                LOGGER.debug(f"解析标准双模态输入: RGB={type(rgb_source)}, X={type(x_source)}")
                
                # Use enhanced loading with integration
                rgb_data, rgb_meta = self._integrate_with_load_inference_source(rgb_source)
                x_data, x_meta = self._integrate_with_load_inference_source(x_source)
                
                # Log loading information
                # 双模态加载成功
                
                # Handle dataset objects vs direct images
                if hasattr(rgb_data, '__iter__') and hasattr(rgb_data, 'source_type'):
                    # Dataset object - extract images
                    rgb_images = self._extract_images_from_dataset(rgb_data)
                else:
                    # Direct images or loaded images list
                    rgb_images = rgb_data if isinstance(rgb_data, list) else [rgb_data]
                
                if hasattr(x_data, '__iter__') and hasattr(x_data, 'source_type'):
                    # Dataset object - extract images
                    x_images = self._extract_images_from_dataset(x_data)
                else:
                    # Direct images or loaded images list
                    x_images = x_data if isinstance(x_data, list) else [x_data]
                
                return rgb_images, x_images
                
            else:
                # Invalid dual-modal input format
                raise ValueError(
                    f"双模态推理需要包含2个元素的列表输入 [rgb_source, x_source]，"
                    f"但接收到: {type(im)} with {len(im)} 元素"
                )
        else:
            # Single source input - invalid for dual-modal
            raise ValueError(
                f"双模态推理需要列表格式输入 [rgb_source, x_source]，"
                f"但接收到单个源: {type(im)}"
            )
    
    def _extract_images_from_dataset(self, dataset):
        """
        Extract images from a dataset object returned by load_inference_source.
        
        Args:
            dataset: Dataset object with __iter__ method
            
        Returns:
            List[np.ndarray]: Extracted images in numpy format
        """
        images = []
        
        try:
            for batch_idx, batch in enumerate(dataset):
                if isinstance(batch, (list, tuple)):
                    # Standard batch format: [paths, images, original_images, ...]
                    if len(batch) > 1:
                        batch_images = batch[1]  # Preprocessed images
                        
                        if isinstance(batch_images, torch.Tensor):
                            # Convert tensor to numpy
                            batch_np = batch_images.cpu().numpy()
                            
                            if batch_np.ndim == 4:  # Batch: (B, C, H, W)
                                for img in batch_np:
                                    # Convert from CHW to HWC format
                                    images.append(img.transpose(1, 2, 0))
                            elif batch_np.ndim == 3:  # Single: (C, H, W)
                                images.append(batch_np.transpose(1, 2, 0))
                        
                        elif isinstance(batch_images, np.ndarray):
                            # Handle numpy arrays
                            if batch_images.ndim == 4:
                                images.extend(list(batch_images))
                            elif batch_images.ndim == 3:
                                images.append(batch_images)
                
                # For inference, usually only need first batch
                if batch_idx == 0:
                    break
                    
        except Exception as e:
            LOGGER.error(f"从数据集提取图像失败: {e}")
            raise
        
        if not images:
            raise ValueError("无法从数据集中提取图像")
        
        LOGGER.debug(f"从数据集成功提取{len(images)}张图像")
        return images
    
    def _load_image_source(self, source):
        """
        Load image(s) from various source types with enhanced integration with load_inference_source.
        
        Supports multiple input formats and provides fallback to ultralytics standard loading mechanisms.
        
        Args:
            source (str | Path | PIL.Image | np.ndarray | torch.Tensor | List): Image source
            
        Returns:
            List[np.ndarray] | torch.Tensor: Loaded images ready for preprocessing
            
        Raises:
            FileNotFoundError: If image files cannot be found
            ValueError: If image format is invalid
            TypeError: If source type is not supported
        """
        import cv2
        import numpy as np
        from PIL import Image
        from pathlib import Path
        
        LOGGER.debug(f"加载图像源: 类型={type(source)}")
        
        # Case 1: String or Path (file path)
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            
            # Check if file exists
            if not source_path.exists():
                raise FileNotFoundError(f"图像文件不存在: {source_path}")
            
            # Use load_inference_source for standard loading
            try:
                dataset = load_inference_source(source_path)
                LOGGER.debug(f"使用load_inference_source加载: {source_path}")
                
                # Extract images from dataset
                images = []
                for batch in dataset:
                    if isinstance(batch, (list, tuple)):
                        # Batch format: [paths, images, original_images, ...]
                        if len(batch) > 1 and hasattr(batch[1], 'shape'):
                            # Use preprocessed images (already normalized)
                            batch_images = batch[1]
                            if isinstance(batch_images, torch.Tensor):
                                # Convert to numpy for consistent format
                                batch_images = batch_images.cpu().numpy()
                            
                            LOGGER.debug(f"load_inference_source返回的数据格式: {batch_images.shape}, dtype={batch_images.dtype}")
                            if batch_images.ndim == 4:  # Batch format
                                for i, img in enumerate(batch_images):
                                    LOGGER.debug(f"批处理图像[{i}]格式: {img.shape}")
                                    # Check if img is CHW or HWC format
                                    if img.shape[0] in [1, 3]:  # CHW format (C=1 or 3)
                                        LOGGER.debug(f"检测到CHW格式，执行transpose")
                                        images.append(img.transpose(1, 2, 0))  # CHW to HWC
                                    else:  # Already HWC format
                                        LOGGER.debug(f"检测到HWC格式，直接使用")
                                        images.append(img)
                            elif batch_images.ndim == 3:  # Single image
                                LOGGER.debug(f"单张图像格式: {batch_images.shape}")
                                # Check if img is CHW or HWC format
                                if batch_images.shape[0] in [1, 3]:  # CHW format (C=1 or 3)
                                    LOGGER.debug(f"检测到CHW格式，执行transpose")
                                    images.append(batch_images.transpose(1, 2, 0))  # CHW to HWC
                                else:  # Already HWC format
                                    LOGGER.debug(f"检测到HWC格式，直接使用")
                                    images.append(batch_images)
                    break  # Only process first batch for single image
                
                if images:
                    LOGGER.debug(f"通过load_inference_source成功加载{len(images)}张图像")
                    return images
                    
            except Exception as e:
                LOGGER.warning(f"load_inference_source加载失败，使用备用方法: {e}")
            
            # Fallback: Direct OpenCV loading
            img = cv2.imread(str(source_path))
            if img is None:
                raise ValueError(f"无法加载图像: {source_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            LOGGER.debug(f"使用OpenCV备用方法成功加载: {source_path}")
            return [img]
            
        # Case 2: PIL Image
        elif isinstance(source, Image.Image):
            LOGGER.debug("处理PIL图像输入")
            if source.mode != "RGB":
                source = source.convert("RGB")
            img = np.asarray(source)
            return [img]
            
        # Case 3: Numpy array
        elif isinstance(source, np.ndarray):
            LOGGER.debug(f"处理numpy数组输入: shape={source.shape}")
            
            if source.ndim == 3:
                # Single image: (H, W, C)
                return [source]
            elif source.ndim == 4:
                # Batch of images: (B, H, W, C) or (B, C, H, W)
                if source.shape[1] == 3 or source.shape[1] == 1:
                    # Format: (B, C, H, W) -> convert to (B, H, W, C)
                    images = [img.transpose(1, 2, 0) for img in source]
                else:
                    # Format: (B, H, W, C)
                    images = list(source)
                return images
            else:
                raise ValueError(f"不支持的numpy数组维度: {source.shape}")
                
        # Case 4: Torch Tensor
        elif isinstance(source, torch.Tensor):
            LOGGER.debug(f"处理torch.Tensor输入: shape={source.shape}")
            
            # Convert to numpy
            if source.device != torch.device('cpu'):
                source = source.cpu()
            source_np = source.numpy()
            
            # Recursive call with numpy array
            return self._load_image_source(source_np)
            
        # Case 5: List or Tuple (multiple images)
        elif isinstance(source, (list, tuple)):
            LOGGER.debug(f"处理列表输入: 长度={len(source)}")
            
            all_images = []
            for i, item in enumerate(source):
                try:
                    loaded = self._load_image_source(item)
                    all_images.extend(loaded)
                except Exception as e:
                    LOGGER.error(f"加载列表项[{i}]失败: {e}")
                    raise
            
            LOGGER.debug(f"列表加载完成: 总计{len(all_images)}张图像")
            return all_images
            
        # Case 6: Dataset or DataLoader objects
        elif hasattr(source, '__iter__') and hasattr(source, 'source_type'):
            LOGGER.debug("处理数据集对象输入")
            
            images = []
            for batch in source:
                if isinstance(batch, (list, tuple)) and len(batch) > 1:
                    batch_images = batch[1]  # Usually the processed images
                    if isinstance(batch_images, torch.Tensor):
                        # Convert and add to results
                        loaded = self._load_image_source(batch_images)
                        images.extend(loaded)
                break  # Only process first batch
            
            return images
            
        else:
            raise TypeError(f"不支持的图像源类型: {type(source)}")

    def _integrate_with_load_inference_source(self, source):
        """
        Enhanced integration with ultralytics load_inference_source mechanism.
        
        This method provides a bridge between YOLOMM's multimodal requirements
        and ultralytics' standard inference source loading.
        
        Args:
            source: Various input source types
            
        Returns:
            tuple: (loaded_data, source_metadata) where loaded_data is ready for processing
        """
        from ultralytics.data.build import check_source
        
        try:
            # Use ultralytics source checking
            checked_source, webcam, screenshot, from_img, in_memory, tensor = check_source(source)
            
            # Create metadata about the source
            source_metadata = {
                'original_source': source,
                'checked_source': checked_source,
                'is_webcam': webcam,
                'is_screenshot': screenshot,
                'from_img': from_img,
                'in_memory': in_memory,
                'is_tensor': tensor
            }
            
            LOGGER.debug(f"源检查结果: webcam={webcam}, screenshot={screenshot}, "
                        f"from_img={from_img}, in_memory={in_memory}, tensor={tensor}")
            
            # Handle different source types
            if tensor:
                # Tensor input - return as is
                return checked_source, source_metadata
                
            elif in_memory or from_img:
                # In-memory data (PIL, numpy, etc.)
                loaded_images = self._load_image_source(checked_source)
                return loaded_images, source_metadata
                
            else:
                # File paths, URLs, webcam, etc. - use load_inference_source
                dataset = load_inference_source(checked_source)
                source_metadata['dataset_type'] = type(dataset).__name__
                return dataset, source_metadata
                
        except Exception as e:
            LOGGER.warning(f"load_inference_source集成失败，使用标准加载: {e}")
            # Fallback to direct loading
            loaded_images = self._load_image_source(source)
            source_metadata = {
                'original_source': source,
                'fallback_used': True,
                'error': str(e)
            }
            return loaded_images, source_metadata
    
    def _align_tensor_dimensions(self, tensor1, tensor2):
        """
        Ensure two tensors have the same spatial dimensions.
        
        If dimensions differ, resize the larger one to match the smaller one.
        
        Args:
            tensor1 (torch.Tensor): First tensor (B, C, H, W)
            tensor2 (torch.Tensor): Second tensor (B, C, H, W)
            
        Returns:
            tuple: (aligned_tensor1, aligned_tensor2) with same spatial dimensions
        """
        import torch.nn.functional as F
        
        if tensor1.shape[2:] == tensor2.shape[2:]:
            # Already same dimensions
            return tensor1, tensor2
        
        # Get dimensions
        h1, w1 = tensor1.shape[2:]
        h2, w2 = tensor2.shape[2:]
        
        # Use the smaller dimensions as target
        target_h = min(h1, h2)
        target_w = min(w1, w2)
        target_size = (target_h, target_w)
        
        LOGGER.debug(f"对齐tensor维度到: {target_size}")
        
        # Resize if necessary
        if (h1, w1) != target_size:
            tensor1 = F.interpolate(tensor1, size=target_size, mode='bilinear', align_corners=False)
        
        if (h2, w2) != target_size:
            tensor2 = F.interpolate(tensor2, size=target_size, mode='bilinear', align_corners=False)
        
        return tensor1, tensor2
    
    def _process_single_modality(self, im):
        """
        Process single-modal input with intelligent filling for missing modality.
        
        Uses the ModalityFiller module to generate missing modality data through
        various strategies (copy, noise, edge detection, etc.).
        
        Args:
            im (str | Path | PIL.Image | np.ndarray | torch.Tensor): Single modal input
            
        Returns:
            torch.Tensor: 6-channel tensor [X, X, X, RGB, RGB, RGB] format
        """
        from .modal_filling import generate_modality_filling
        
        # 如果输入是路径字符串，需要先加载图像
        if isinstance(im, (str, Path)):
            im = self._load_image_source(im)
        
        # 确保传递给父类preprocess的是正确格式的列表
        if isinstance(im, np.ndarray) and im.ndim == 3:
            # 单个HWC图像需要包装成列表格式
            im = [im]
            # LOGGER.debug(f"单个图像已包装成列表格式用于父类preprocess处理")
        
        # Preprocess the input modality using parent's method
        source_tensor = super().preprocess(im)  # Shape: (B, 3, H, W)
        
        if self.modality == 'rgb':
            # RGB模态推理：RGB为真实数据，X模态需要填充
            rgb_tensor = source_tensor
            x_tensor = generate_modality_filling(
                source_tensor=rgb_tensor,
                source_modality='rgb', 
                target_modality='x'  # 通用X模态标识
            )
            LOGGER.debug(f"RGB单模态推理 - 生成X模态填充数据")
            
        elif self.modality in ['depth', 'thermal', 'ir', 'nir']:
            # X模态推理：X模态为真实数据，RGB需要填充
            x_tensor = source_tensor
            rgb_tensor = generate_modality_filling(
                source_tensor=x_tensor,
                source_modality=self.modality,
                target_modality='rgb'
            )
            LOGGER.debug(f"{self.modality}单模态推理 - 生成RGB填充数据")
            
        else:
            raise ValueError(f"不支持的单模态类型: {self.modality}")
        
        # 获取填充数据的统计信息用于调试
        from .modal_filling import default_modality_filler
        rgb_stats = default_modality_filler.get_statistics(rgb_tensor)
        x_stats = default_modality_filler.get_statistics(x_tensor)
        
        LOGGER.debug(f"RGB tensor统计: mean={rgb_stats['mean']:.3f}, std={rgb_stats['std']:.3f}")
        LOGGER.debug(f"X tensor统计: mean={x_stats['mean']:.3f}, std={x_stats['std']:.3f}")
        
        # 组合成6通道tensor: [X, X, X, RGB, RGB, RGB] 顺序（与训练保持一致）
        combined_tensor = torch.cat([x_tensor, rgb_tensor], dim=1)  # Shape: (B, 6, H, W)
        
        LOGGER.debug(f"单模态预处理完成: {combined_tensor.shape}")
        return combined_tensor
    
    def _validate_input_modality_consistency(self, im):
        """
        Validate input format consistency with modality settings.
        
        Args:
            im: Input data to validate
            
        Raises:
            ValueError: If input format is inconsistent with modality settings
        """
        if self.is_dual_modal:
            # 双模态输入验证
            if not isinstance(im, (list, tuple)) or len(im) != 2:
                if not (isinstance(im, torch.Tensor) and im.shape[1] == 6):
                    raise ValueError(
                        f"双模态推理需要包含2个元素的列表输入 [rgb_source, x_source] "
                        f"或6通道tensor，但接收到: {type(im)}"
                    )
        else:
            # 单模态输入验证
            if isinstance(im, (list, tuple)) and len(im) > 1:
                LOGGER.warning(
                    f"单模态推理模式({self.modality})接收到多个输入源，将仅使用第一个: {im[0]}"
                )
    
    def _finalize_tensor(self, tensor):
        """
        Finalize processed tensor with format validation and device management.
        
        Args:
            tensor (torch.Tensor): Processed tensor to finalize
            
        Returns:
            torch.Tensor: Finalized tensor on correct device
            
        Raises:
            ValueError: If tensor format is invalid
        """
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"期望torch.Tensor输出，但得到: {type(tensor)}")
        
        if tensor.dim() != 4:
            raise ValueError(f"期望4维tensor [B, C, H, W]，但得到维度: {tensor.dim()}")
        
        if tensor.shape[1] != 6:
            raise ValueError(f"期望6通道tensor，但得到: {tensor.shape[1]}通道")
        
        # 设备管理 - 确保tensor在正确的设备上
        if hasattr(self, 'device') and self.device != tensor.device:
            LOGGER.debug(f"转移tensor到设备: {self.device}")
            tensor = tensor.to(self.device)
        
        # 数据类型确保
        if tensor.dtype != torch.float32:
            LOGGER.debug(f"转换tensor数据类型: {tensor.dtype} -> torch.float32")
            tensor = tensor.float()
        
        return tensor
    
    def _log_debug_info(self, im, exception):
        """
        Log detailed debug information when preprocessing fails.
        
        Args:
            im: Original input data
            exception: The exception that occurred
        """
        LOGGER.debug("=== 多模态预处理调试信息 ===")
        LOGGER.debug(f"模态设置: modality={self.modality}, is_dual_modal={self.is_dual_modal}")
        LOGGER.debug(f"输入类型: {type(im)}")
        
        if isinstance(im, (list, tuple)):
            LOGGER.debug(f"列表输入长度: {len(im)}")
            for i, item in enumerate(im):
                LOGGER.debug(f"  项目[{i}]: {type(item)} - {item}")
        elif isinstance(im, torch.Tensor):
            LOGGER.debug(f"Tensor形状: {im.shape}")
            LOGGER.debug(f"Tensor设备: {im.device}")
            LOGGER.debug(f"Tensor数据类型: {im.dtype}")
        else:
            LOGGER.debug(f"输入内容: {im}")
        
        LOGGER.debug(f"异常类型: {type(exception).__name__}")
        LOGGER.debug(f"异常信息: {str(exception)}")
        LOGGER.debug("=== 调试信息结束 ===")
    
    def get_preprocessing_info(self):
        """
        Get information about the current preprocessing configuration.
        
        Returns:
            dict: Preprocessing configuration information
        """
        return {
            'modality': self.modality,
            'is_dual_modal': self.is_dual_modal,
            'is_single_modal': self.is_single_modal,
            'supported_modalities': list(self.SUPPORTED_MODALITIES),
            'expected_input_channels': 6,
            'device': getattr(self, 'device', 'not_set')
        }

    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """
        Streams real-time inference on camera feed and saves results to file.
        
        Overrides parent method to handle 6-channel warmup for multimodal models.
        
        Args:
            source (str, optional): The source of the image to make predictions on.
            model (nn.Module, optional): The model to use for predictions.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.
            
        Yields:
            (List[ultralytics.engine.results.Results]): The prediction results.
        """
        if self.args.verbose:
            LOGGER.info("")

        # Setup model
        if not self.model:
            self.setup_model(model)

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # Warmup model with 6 channels for multimodal models
            if not self.done_warmup:
                # 检测模型输入通道数
                model_channels = 6  # YOLOMM固定为6通道
                # YOLOMM模型warmup - 6通道输入
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, model_channels, *self.imgsz))
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            for self.batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch

                # Preprocess
                with profilers[0]:
                    im = self.preprocess(im0s)

                # Inference
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                        continue

                # Postprocess
                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)
                self.run_callbacks("on_predict_postprocess_end")

                # Visualize, save, write results
                n = len(im0s)  # 原始输入图像数量
                results_count = len(self.results)  # 实际生成的结果数量
                
                # 对于多模态推理，原始图像可能是2张，但结果只有1个
                # 需要根据实际results数量来处理
                if results_count != n:
                    LOGGER.debug(f"多模态推理: 输入{n}张图像，生成{results_count}个结果")
                
                for i in range(results_count):  # 使用实际结果数量
                    self.seen += 1
                    self.results[i].speed = {
                        "preprocess": profilers[0].dt * 1e3 / results_count,  # 使用结果数量计算平均时间
                        "inference": profilers[1].dt * 1e3 / results_count,
                        "postprocess": profilers[2].dt * 1e3 / results_count,
                    }
                    
                    # 为多模态推理调整路径处理
                    if results_count < n:
                        # 多模态情况：多个输入产生1个结果，使用第一个路径作为主路径
                        result_path = Path(paths[0])
                        result_string = s[0] if s else ""
                        
                        # 在结果字符串中添加多模态信息
                        if len(paths) > 1:
                            modality_info = f"({len(paths)}模态输入)"
                            result_string = f"{result_string} {modality_info}" if result_string else modality_info
                    else:
                        # 标准情况：1对1映射
                        result_path = Path(paths[i])
                        result_string = s[i] if i < len(s) else ""
                    
                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        result_string += self.write_results(i, result_path, im, result_string)
                    
                    # 保存更新后的字符串
                    if i < len(s):
                        s[i] = result_string
                    elif len(s) == 0:
                        s = [result_string]

                # Print batch results
                if self.args.verbose:
                    # 只打印有效的结果字符串
                    valid_strings = [s_item for s_item in s[:results_count] if s_item]
                    if valid_strings:
                        LOGGER.info("\n".join(valid_strings))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results

        # Release assets
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        # Print final results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), 6, *im.shape[2:])}"  # 显示6通道
                % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")
    
    def postprocess(self, preds, img, orig_imgs):
        """
        Override to store original images for multimodal visualization and handle dual-modal correctly.
        
        For dual-modal input, creates only one Results object using the RGB path to avoid duplicate outputs.
        """
        # Store original images for later use
        self._orig_imgs_cache = orig_imgs
        
        # Call parent's postprocess to get properly formatted results
        results = super().postprocess(preds, img, orig_imgs)
        
        # For dual-modal input, only keep the first result (RGB)
        if self.is_dual_modal and hasattr(self, '_dual_input_detected') and self._dual_input_detected:
            # Check if we have dual-modal paths and multiple results
            if hasattr(self, 'batch') and self.batch and len(self.batch[0]) == 2 and len(results) > 1:
                # Only keep the first result (RGB path)
                # This prevents creating duplicate outputs for IR path
                return [results[0]]
        
        # Return all results for single-modal or other cases
        return results
    
    def write_results(self, i: int, p: Path, im: torch.Tensor, s: list) -> str:
        """
        Override parent's write_results to handle multi-modal result saving.
        
        For dual-modal input: saves RGB result, X result, and side-by-side comparison
        For single-modal input: saves only the specified modality result
        
        Args:
            i (int): Index of the current result in the batch
            p (Path): Path to the original image file
            im (torch.Tensor): Preprocessed 6-channel tensor [X,X,X,RGB,RGB,RGB]
            s (list): List of result strings
            
        Returns:
            str: Result string with processing information
        """
        string = super().write_results(i, p, im, s)
        
        # Only process if saving is enabled
        if self.args.save:
            # Determine save mode based on input
            if self.is_dual_modal and hasattr(self, '_dual_input_detected') and self._dual_input_detected:
                # Dual-modal input: save RGB, X, and combined results
                self._save_multimodal_results(i, p, im)
            elif self.is_single_modal:
                # Single-modal input: modify filename to include modality
                self._update_single_modal_filename(p)
        
        return string
    
    def _save_multimodal_results(self, i: int, p: Path, im: torch.Tensor):
        """
        Save separate RGB and X modality results for dual-modal input.
        
        Creates three output files:
        1. filename_RGB.jpg - RGB modality inference result
        2. filename_X.jpg - X modality inference result  
        3. filename_Multimodal.jpg - Side-by-side comparison
        
        Args:
            i (int): Result index
            p (Path): Original image path
            im (torch.Tensor): 6-channel preprocessed tensor
        """
        try:
            # Get the result object
            result = self.results[i]
            
            # Get base filename without extension
            base_name = p.stem
            save_dir = self.save_dir
            
            # Get original images from cache if available
            if hasattr(self, '_orig_imgs_cache') and self._orig_imgs_cache is not None:
                # For dual-modal input, we should have RGB and X images
                if len(self._orig_imgs_cache) >= 2:
                    # First image is RGB, second is X modality
                    rgb_img = self._orig_imgs_cache[0]
                    x_img = self._orig_imgs_cache[1]
                    
                    # Ensure images are in RGB format
                    if isinstance(rgb_img, np.ndarray):
                        if rgb_img.ndim == 2:
                            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2RGB)
                        elif rgb_img.shape[2] == 4:
                            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGRA2RGB)
                        elif rgb_img.shape[2] == 3 and rgb_img.dtype == np.uint8:
                            # Assume BGR format from cv2.imread
                            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                    
                    if isinstance(x_img, np.ndarray):
                        if x_img.ndim == 2:
                            x_img = cv2.cvtColor(x_img, cv2.COLOR_GRAY2RGB)
                        elif x_img.shape[2] == 4:
                            x_img = cv2.cvtColor(x_img, cv2.COLOR_BGRA2RGB)
                        elif x_img.shape[2] == 3 and x_img.dtype == np.uint8:
                            # Assume BGR format from cv2.imread
                            x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
                else:
                    # Fallback to tensor conversion if not enough original images
                    LOGGER.warning("使用tensor恢复图像作为备用方案")
                    rgb_tensor, x_tensor = self._separate_modalities(im)
                    rgb_img = self._tensor_to_image(rgb_tensor)
                    x_img = self._tensor_to_image(x_tensor)
            else:
                # Fallback to tensor conversion
                LOGGER.warning("原始图像缓存不可用，使用tensor恢复")
                rgb_tensor, x_tensor = self._separate_modalities(im)
                rgb_img = self._tensor_to_image(rgb_tensor)
                x_img = self._tensor_to_image(x_tensor)
            
            # Plot results on each modality
            # RGB modality result (使用原始文件名，不加后缀)
            rgb_annotated = self._plot_on_image(result, rgb_img, "RGB")
            rgb_path = save_dir / f"{base_name}.jpg"  # 按照设计范式，RGB预测图使用原始文件名
            cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_annotated, cv2.COLOR_RGB2BGR))
            
            # X modality result
            x_annotated = self._plot_on_image(result, x_img, "X")
            x_path = save_dir / f"{base_name}_X.jpg"
            cv2.imwrite(str(x_path), cv2.cvtColor(x_annotated, cv2.COLOR_RGB2BGR))
            
            # Create side-by-side comparison  
            multimodal_img = self._create_multimodal_comparison(rgb_annotated, x_annotated)
            multimodal_path = save_dir / f"{base_name}_multimodal.jpg"  # 小写multimodal
            cv2.imwrite(str(multimodal_path), cv2.cvtColor(multimodal_img, cv2.COLOR_RGB2BGR))
            
            LOGGER.info(f"保存多模态结果: RGB={rgb_path}, X={x_path}, 对比图={multimodal_path}")
            
        except Exception as e:
            LOGGER.error(f"保存多模态结果失败: {e}")
    
    def _update_single_modal_filename(self, p: Path):
        """
        Update the saved filename for single-modal input to include modality.
        
        Args:
            p (Path): Original image path
        """
        try:
            # Wait a bit to ensure file is saved
            import time
            time.sleep(0.2)  # 稍微延长等待时间
            
            # Get the default save path
            default_path = self.save_dir / p.name
            default_jpg_path = default_path.with_suffix('.jpg')
            
            # Create new filename with modality
            base_name = p.stem
            modality_upper = self.modality.upper() if self.modality else "UNKNOWN"
            new_path = self.save_dir / f"{base_name}_{modality_upper}.jpg"
            
            # Also check for the file with original extension
            original_ext_path = self.save_dir / p.name
            
            # Try multiple possible default filenames
            possible_files = [default_jpg_path, original_ext_path]
            
            for default_file in possible_files:
                if default_file.exists() and not new_path.exists():
                    # Rename the file
                    default_file.rename(new_path)
                    LOGGER.info(f"单模态文件重命名成功: {default_file.name} -> {new_path.name}")
                    break
            else:
                # If no file found to rename, log debug info
                LOGGER.warning(f"单模态文件重命名失败: 未找到默认保存的文件")
                LOGGER.debug(f"尝试查找的文件: {[str(f) for f in possible_files]}")
                LOGGER.debug(f"目录中的文件: {list(self.save_dir.glob('*'))}")
                
        except Exception as e:
            LOGGER.error(f"更新单模态文件名失败: {e}")
    
    def _separate_modalities(self, tensor: torch.Tensor) -> tuple:
        """
        Separate 6-channel tensor into RGB and X modality tensors.
        
        Args:
            tensor (torch.Tensor): 6-channel tensor [X,X,X,RGB,RGB,RGB]
            
        Returns:
            tuple: (rgb_tensor, x_tensor) each with 3 channels
        """
        if tensor.dim() == 3:
            # Single image: (6, H, W)
            x_tensor = tensor[:3]    # First 3 channels (X modality)
            rgb_tensor = tensor[3:]  # Last 3 channels (RGB)
        else:
            # Batch: (B, 6, H, W)
            x_tensor = tensor[:, :3]    # X modality channels
            rgb_tensor = tensor[:, 3:]  # RGB channels
            
        return rgb_tensor, x_tensor
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert preprocessed tensor back to displayable image.
        
        Handles denormalization and format conversion.
        
        Args:
            tensor (torch.Tensor): Normalized tensor (C, H, W) or (B, C, H, W)
            
        Returns:
            np.ndarray: Image in HWC format with uint8 values
        """
        # Remove batch dimension if present
        if tensor.dim() == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        elif tensor.dim() == 4:
            # For batch, take first image
            tensor = tensor[0]
            
        # Move to CPU if on GPU
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
        
        # Convert to numpy and transpose to HWC
        img = tensor.numpy()
        if img.shape[0] == 3 or img.shape[0] == 1:
            img = img.transpose(1, 2, 0)  # CHW to HWC
        
        # Denormalize (assuming standard ImageNet normalization)
        # Note: This assumes the tensor was normalized with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        if img.shape[2] == 3:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
        
        # Clip values and convert to uint8
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        # Ensure 3 channels (convert grayscale to RGB if needed)
        if img.ndim == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2RGB)
            
        return img
    
    def _plot_on_image(self, result, img: np.ndarray, modality_name: str) -> np.ndarray:
        """
        Plot detection results on a specific modality image.
        
        Args:
            result: Detection result object
            img (np.ndarray): Image to plot on (HWC format)
            modality_name (str): Name of the modality for labeling
            
        Returns:
            np.ndarray: Annotated image
        """
        # Create a copy to avoid modifying original
        img_copy = img.copy()
        
        # Use the result's plot method with the specific image
        annotated = result.plot(
            img=img_copy,
            line_width=self.args.line_width,
            boxes=self.args.show_boxes,
            conf=self.args.show_conf,
            labels=self.args.show_labels
        )
        
        # Add modality label
        h, w = annotated.shape[:2]
        label_bg_color = (0, 0, 0)  # Black background
        label_text_color = (255, 255, 255)  # White text
        
        # Draw modality label in top-left corner
        cv2.rectangle(annotated, (10, 10), (150, 40), label_bg_color, -1)
        cv2.putText(annotated, f"{modality_name} Modality", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_text_color, 2)
        
        return annotated
    
    def _create_multimodal_comparison(self, rgb_img: np.ndarray, x_img: np.ndarray) -> np.ndarray:
        """
        Create side-by-side comparison of RGB and X modality results.
        
        Args:
            rgb_img (np.ndarray): Annotated RGB image
            x_img (np.ndarray): Annotated X modality image
            
        Returns:
            np.ndarray: Combined side-by-side image
        """
        # Ensure both images have the same height
        h1, w1 = rgb_img.shape[:2]
        h2, w2 = x_img.shape[:2]
        
        if h1 != h2:
            # Resize to match heights
            target_h = max(h1, h2)
            if h1 < target_h:
                scale = target_h / h1
                new_w1 = int(w1 * scale)
                rgb_img = cv2.resize(rgb_img, (new_w1, target_h))
            else:
                scale = target_h / h2
                new_w2 = int(w2 * scale)
                x_img = cv2.resize(x_img, (new_w2, target_h))
        
        # Create side-by-side image
        gap = 10  # Gap between images
        combined_width = rgb_img.shape[1] + x_img.shape[1] + gap
        combined_height = max(rgb_img.shape[0], x_img.shape[0])
        
        # Create black canvas
        combined = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        
        # Place images
        combined[:rgb_img.shape[0], :rgb_img.shape[1]] = rgb_img
        combined[:x_img.shape[0], rgb_img.shape[1] + gap:] = x_img
        
        # Add title
        title = "Multi-Modal Detection Results"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        title_x = (combined_width - title_size[0]) // 2
        cv2.putText(combined, title, (title_x, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        return combined
    
    def preprocess(self, im):
        """
        Override parent's preprocess to track input mode.
        """
        # Detect input mode before preprocessing
        if isinstance(im, (list, tuple)) and len(im) == 2:
            self._dual_input_detected = True
            self.input_mode = 'dual'
        else:
            self._dual_input_detected = False
            self.input_mode = f'single_{self.modality}' if self.modality else 'single'
        
        # Call parent's preprocess
        return super().preprocess(im) 