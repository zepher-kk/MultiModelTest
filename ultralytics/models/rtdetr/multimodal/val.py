# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
RT-DETR MultiModal validator module.

This module provides the RTDETRMMValidator class for validating multi-modal RT-DETR models
with support for RGB+X modality inputs.
"""

import os
import json
import torch
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.data.build import build_yolo_dataset
from ultralytics.utils import LOGGER, colorstr, TQDM
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.ops import Profile
from copy import copy


class RTDETRMMValidator(RTDETRValidator):
    """
    A validator class for RT-DETR MultiModal (RTDETRMM) object detection models.

    This class extends RTDETRValidator to support multi-modal inputs (RGB + X modality)
    during the validation process. It handles evaluation metrics and performance assessment
    for multi-modal RT-DETR models.

    Attributes:
        args: Validation arguments and settings.
        model: The RTDETRMM model being validated.
        dataloader: Multi-modal validation dataloader.
        metrics: Validation metrics for multi-modal detection.

    Methods:
        preprocess: Preprocess batch data for multi-modal validation.
        init_metrics: Initialize metrics for multi-modal evaluation.
        get_dataloader: Create multi-modal validation dataloader.

    Examples:
        >>> validator = RTDETRMMValidator(args={'data': 'multimodal-dataset.yaml'})
        >>> validator(model=rtdetrmm_model)
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Initialize RTDETRMMValidator for multi-modal validation.

        Args:
            dataloader: Multi-modal validation dataloader.
            save_dir (Path, optional): Directory to save validation results.
            pbar (tqdm, optional): Progress bar for validation.
            args (SimpleNamespace): Validation configuration arguments.
            _callbacks (list, optional): List of callback functions.
        """
        # RTDETRValidator inherits from DetectionValidator
        # DetectionValidator.__init__(dataloader=None, save_dir=None, args=None, _callbacks=None)
        # Note: pbar parameter is kept for interface compatibility but not passed to parent
        super().__init__(dataloader, save_dir, args, _callbacks)

        # Detect multimodal mode
        self.is_multimodal = self._detect_multimodal_mode()

        # Get modality parameter from args (consistent with trainer)
        if args:
            if isinstance(args, dict):
                self.modality = args.get('modality', None)
            else:
                self.modality = getattr(args, 'modality', None)
        else:
            self.modality = None

        # Initialize multimodal configuration (parsed later when data is available)
        self.multimodal_config = None

        # Log initialization status
        if self.is_multimodal:
            LOGGER.info(f"🚀 {colorstr('RTDETRMMValidator')}: 多模态验证模式已启用")
            if self.modality:
                LOGGER.info(f"🎯 单模态消融验证: {colorstr(self.modality)}")
            else:
                LOGGER.info("🔄 双模态验证模式")
        else:
            LOGGER.info(f"📋 {colorstr('RTDETRMMValidator')}: 标准模式（向后兼容）")

    def _detect_multimodal_mode(self) -> bool:
        """
        Detect if this is a multimodal validation session

        Returns:
            bool: True if multimodal mode detected
        """
        # Check for -mm suffix in model name from args
        if hasattr(self, 'args') and self.args:
            model_path = getattr(self.args, 'model', '')
            if isinstance(model_path, (str, Path)):
                model_str = str(model_path)
                return '-mm' in model_str

        return False

    @property
    def is_dual_modal(self) -> bool:
        """Check if in dual-modal mode"""
        return self.is_multimodal and self.modality is None

    @property
    def is_single_modal(self) -> bool:
        """Check if in single-modal mode"""
        return self.is_multimodal and self.modality is not None

    def __call__(self, trainer=None, model=None):
        """
        执行多模态RT-DETR验证过程，支持动态通道数warmup。
        
        重写基类方法以支持动态通道数warmup和多模态数据处理。
        参考YOLOMM的实现但改进为支持动态Xch配置。
        
        Args:
            trainer: Training instance (if called during training)
            model: Model to validate (if called independently)
            
        Returns:
            dict: Validation metrics
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        
        if self.training:
            self.device = trainer.device
            # 关键修复：从trainer获取data配置（包含更新后的channels）
            if self.data is None:
                self.data = trainer.data
            # force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            # 独立验证模式：使用传入的模型或加载模型
            from ultralytics.utils import callbacks, emojis
            from ultralytics.utils.checks import check_imgsz
            from ultralytics.nn.autobackend import AutoBackend
            from ultralytics.utils.torch_utils import select_device
            from ultralytics.data.utils import check_det_dataset
            
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = model.metadata.get("batch", 1)
                LOGGER.info(f"Setting batch={self.args.batch} for RT-DETR validation")

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                # 如果没有data配置，从args.data加载
                if not hasattr(self, 'data') or self.data is None:
                    self.data = check_det_dataset(self.args.data)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not pt:
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            
            # 🔧 关键修复：动态通道数warmup（参考YOLOMM但改进）
            if hasattr(self, 'data') and self.data and 'Xch' in self.data:
                # 动态计算通道数：RGB(3) + X(Xch)
                x_channels = self.data.get('Xch', 3)
                total_channels = 3 + x_channels
                LOGGER.info(f"执行{total_channels}通道多模态RT-DETR模型warmup (RGB:3 + X:{x_channels})")
                model.warmup(imgsz=(1 if pt else self.args.batch, total_channels, imgsz, imgsz))
            else:
                # 回退到默认6通道（向后兼容）
                LOGGER.info("执行6通道多模态RT-DETR模型warmup (默认)")
                model.warmup(imgsz=(1 if pt else self.args.batch, 6, imgsz, imgsz))

        # 继续执行标准验证流程
        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader) * 1E3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}
        else:
            LOGGER.info("Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image" %
                       tuple(self.speed.values()))
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build multi-modal dataset using YOLOMultiModalImageDataset.

        This method follows YOLOMM's successful pattern by overriding build_dataset
        to ensure proper initialization timing and data access for validation.

        Args:
            img_path (str): Path to images
            mode (str): Dataset mode ('val', 'test')
            batch (int, optional): Batch size for validation

        Returns:
            Dataset: YOLOMultiModalImageDataset for multi-modal, standard dataset otherwise
        """
        if not self.is_multimodal:
            # Fall back to standard RT-DETR dataset
            return super().build_dataset(img_path, mode, batch)

        # Get model stride parameter (consistent with DetectionValidator)
        stride = getattr(self, 'stride', 32)
        if hasattr(self, 'model') and self.model:
            stride = max(int(de_parallel(self.model).stride.max() if hasattr(self.model, 'stride') else 0), 32)

        # Lazy loading: parse multi-modal configuration on demand
        if self.multimodal_config is None:
            self.multimodal_config = self._parse_multimodal_config()
            LOGGER.info(f"多模态验证配置解析完成 - 模态: {self.multimodal_config['models']}")

        # Use parsed modality configuration
        modalities = self.multimodal_config['models']
        modalities_dict = self.multimodal_config['modalities']

        # 获取X模态信息（关键修复）
        x_modalities = [m for m in modalities if m != 'rgb']
        x_modality = x_modalities[0] if x_modalities else None
        x_modality_dir = modalities_dict.get(x_modality) if x_modality else None

        LOGGER.info(f"构建多模态验证数据集 - 模式: {mode}, 路径: {img_path}, 模态: {modalities}")

        # If single-modal validation is enabled, log modality info and validate compatibility
        if self.modality:
            self._validate_modality_compatibility()
            LOGGER.info(f"启用单模态验证: {self.modality}-only，将应用智能模态填充")

        # Call build_yolo_dataset with multi_modal_image=True to enable multi-modal dataset
        return build_yolo_dataset(
            self.args, img_path, batch, self.data,
            mode=mode,
            rect=True,  # RT-DETR validation uses rect inference
            stride=stride,
            multi_modal_image=True,  # Key parameter: enable YOLOMultiModalImageDataset
            x_modality=x_modality,  # ✅ 修复：传递模态类型
            x_modality_dir=x_modality_dir,  # ✅ 修复：传递模态目录
            modalities=modalities,  # Pass modality configuration（向后兼容）
            train_modality=self.modality,  # Pass modality control parameter for consistency
        )

    def preprocess(self, batch):
        """
        Preprocess batch data for multi-modal validation.

        Ensures 6-channel data is correctly processed and maintains consistency
        with the training phase preprocessing pipeline.

        Args:
            batch (dict): Batch data containing images and labels

        Returns:
            dict: Preprocessed batch data
        """
        # Call parent preprocessing method
        batch = super().preprocess(batch)

        # Multi-modal specific preprocessing
        if self.is_multimodal and "img" in batch:
            # Validate input channels
            if batch["img"].shape[1] == 6:
                # Apply single-modal ablation if specified
                if self.modality:
                    self._apply_modality_ablation(batch)
            elif batch["img"].shape[1] == 3:
                # Standard 3-channel input (fallback mode)
                LOGGER.debug("接收到3通道输入，使用标准预处理")
            else:
                LOGGER.warning(f"意外的通道数: {batch['img'].shape[1]}，期望3或6通道")

        return batch

    def _apply_modality_ablation(self, batch):
        """
        Apply single-modality ablation by zeroing out non-selected modality channels.

        Args:
            batch (dict): Batch data to modify
        """
        if not self.modality:
            return

        images = batch["img"]  # [B, 6, H, W]

        if self.modality == 'rgb':
            # Zero out X modality (channels 3-5)
            images[:, 3:6, :, :] = 0
            LOGGER.debug("单模态RGB验证: X模态通道已置零")
        else:
            # Zero out RGB modality (channels 0-2)
            images[:, 0:3, :, :] = 0
            LOGGER.debug(f"单模态{self.modality}验证: RGB通道已置零")

        batch["img"] = images

    def _parse_multimodal_config(self):
        """
        Parse and validate multi-modal configuration from data.yaml.

        Uses safe data access pattern from YOLOMM to avoid AttributeError during initialization.
        Maintains consistency with RTDETRMMTrainer configuration parsing.

        Returns:
            dict: Parsed multi-modal configuration
        """
        # Safe data access - use getattr to avoid AttributeError during initialization
        data = getattr(self, 'data', None)

        # Priority 1: User-specified modality parameter (single-modal validation)
        if self.modality:
            # Build single-modal configuration
            if self.modality == 'rgb':
                # RGB single-modal: use RGB + dynamically determined X modality for zero padding
                x_modality = self._determine_x_modality_from_data()
                config = {
                    'models': ['rgb', x_modality],
                    'modalities': {
                        'rgb': 'images',
                        x_modality: f'images_{x_modality}'
                    }
                }
                LOGGER.info(f"RGB单模态验证，动态确定X模态: {x_modality}")
            else:
                # 处理 'X' 特殊标记（大小写不敏感）
                if self.modality.upper() == 'X':
                    # 'X' 是特殊标记，需要解析为实际的X模态
                    actual_x_modality = self._determine_x_modality_from_data()
                    # 从data.yaml获取实际的路径映射
                    x_modality_path = self._get_x_modality_path(actual_x_modality)
                    
                    config = {
                        'models': ['rgb', actual_x_modality],
                        'modalities': {
                            'rgb': 'images',
                            actual_x_modality: x_modality_path
                        }
                    }
                    LOGGER.info(f"X模态单模态验证: {actual_x_modality}-only (从'X'解析)")
                else:
                    # 用户指定了具体的模态名称（如 'depth', 'thermal', 'ir' 等）
                    x_modality_path = self._get_x_modality_path(self.modality)
                    
                    config = {
                        'models': ['rgb', self.modality],
                        'modalities': {
                            'rgb': 'images',
                            self.modality: x_modality_path
                        }
                    }
                    LOGGER.info(f"X模态单模态验证: {self.modality}-only")

            return config

        # Priority 2: Dual-modal validation (use original configuration parsing logic)
        return self._get_default_multimodal_config()

    def _get_default_multimodal_config(self):
        """
        Get default multi-modal configuration from data.yaml.

        Returns:
            dict: Default multi-modal configuration.
        """
        # Check if data is available
        data = getattr(self, 'data', None)

        # Priority 1: modality_used field
        if data and 'modality_used' in data:
            modality_used = data['modality_used']
            if isinstance(modality_used, list) and len(modality_used) >= 2:
                config = {
                    'models': modality_used,
                    'modalities': {}
                }

                # Get path mappings from modality field
                if 'modality' in data and isinstance(data['modality'], dict):
                    modality_paths = data['modality']
                    for mod in modality_used:
                        config['modalities'][mod] = modality_paths.get(
                            mod, 'images' if mod == 'rgb' else f'images_{mod}'
                        )
                else:
                    # Generate default paths
                    for mod in modality_used:
                        config['modalities'][mod] = 'images' if mod == 'rgb' else f'images_{mod}'

                LOGGER.info(f"Loaded multi-modal validation config: {modality_used}")
                return config

        # Priority 2: models field (backward compatibility)
        if data and 'models' in data:
            models = data['models']
            if isinstance(models, list) and len(models) >= 2:
                config = {'models': models, 'modalities': {}}
                for modality in models:
                    config['modalities'][modality] = 'images' if modality == 'rgb' else f'images_{modality}'
                return config

        # Default fallback
        x_modality = self._determine_x_modality_from_data()
        config = {
            'models': ['rgb', x_modality],
            'modalities': {
                'rgb': 'images',
                x_modality: f'images_{x_modality}'
            }
        }
        LOGGER.info(f"Using default multi-modal validation config: rgb+{x_modality}")
        return config

    def _get_x_modality_path(self, modality_name):
        """
        获取指定模态的实际路径。
        
        优先从data.yaml的modality字段读取，
        如果不存在则使用默认格式 'images_{modality_name}'。
        
        Args:
            modality_name (str): 模态名称（如 'ir', 'depth', 'thermal'）
            
        Returns:
            str: 模态对应的目录路径
        """
        # 优先从data.yaml的modality字段读取
        data = getattr(self, 'data', None)
        if data and 'modality' in data:
            modality_paths = data['modality']
            if isinstance(modality_paths, dict) and modality_name in modality_paths:
                return modality_paths[modality_name]
        
        # 向后兼容：检查modalities字段
        if data and 'modalities' in data:
            modalities = data['modalities']
            if isinstance(modalities, dict) and modality_name in modalities:
                return modalities[modality_name]
        
        # 如果没有配置，使用默认格式
        return f'images_{modality_name}'

    def _determine_x_modality_from_data(self):
        """
        Intelligently determine X modality type from data configuration.

        Returns:
            str: X modality identifier (e.g., 'depth', 'thermal', 'ir').
        """
        # Safe data access - use getattr to avoid AttributeError during initialization
        data = getattr(self, 'data', None)

        # Check data.yaml for modality information
        if data:
            # Check modality_used
            if 'modality_used' in data:
                modality_used = data['modality_used']
                if isinstance(modality_used, list):
                    x_modalities = [m for m in modality_used if m != 'rgb']
                    if x_modalities:
                        return x_modalities[0]

            # Check models field
            if 'models' in data:
                models = data['models']
                if isinstance(models, list):
                    x_modalities = [m for m in models if m != 'rgb']
                    if x_modalities:
                        return x_modalities[0]

            # Check modality paths
            if 'modality' in data and isinstance(data['modality'], dict):
                modality_paths = data['modality']
                x_modalities = [m for m in modality_paths.keys() if m != 'rgb']
                if x_modalities:
                    x_modality = x_modalities[0]
                    LOGGER.info(f"从modality路径推断X模态: {x_modality}")
                    return x_modality

        # Fallback: use depth as default
        LOGGER.warning("无法自动确定X模态类型，使用默认值: depth")
        return 'depth'

    def _validate_modality_compatibility(self):
        """
        Validate compatibility between user-specified modality parameter and data configuration.

        Raises:
            ValueError: When modality parameter is incompatible with available data
        """
        if not self.modality:
            return

        # Get available modalities
        available_modalities = []
        if hasattr(self, 'multimodal_config') and self.multimodal_config:
            available_modalities = self.multimodal_config.get('models', [])
        elif hasattr(self, 'data') and self.data and 'models' in self.data:
            available_modalities = self.data['models']

        # Validate modality compatibility
        if available_modalities:
            # 处理 'X' 特殊标记的验证
            if self.modality.upper() == 'X':
                # 'X' 是特殊标记，检查是否有非RGB的X模态
                x_modalities = [m for m in available_modalities if m != 'rgb']
                if x_modalities:
                    LOGGER.info(f"✅ 验证模态兼容性通过: '{self.modality}' 映射到 {x_modalities[0]}")
                else:
                    raise ValueError(
                        f"指定的验证modality '{self.modality}' 无法映射到有效的X模态。"
                        f"可用模态列表: {available_modalities}，但没有找到非RGB的X模态。"
                    )
            else:
                # 标准模态验证
                if self.modality not in available_modalities:
                    raise ValueError(
                        f"指定的验证modality '{self.modality}' 不在可用模态列表中: {available_modalities}。"
                        f"请检查数据配置或modality参数。"
                    )
                LOGGER.info(f"✅ 验证模态兼容性通过: {self.modality} 在可用模态 {available_modalities} 中")
        else:
            # If unable to get available modalities, just give a warning
            LOGGER.warning(f"⚠️  无法验证验证modality '{self.modality}' 的兼容性，未找到可用模态配置")

