# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
from copy import copy

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.utils import LOGGER, DEFAULT_CFG
from ultralytics.data.dataset import YOLOMultiModalImageDataset
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.patches import torch_load


class MultiModalDetectionTrainer(DetectionTrainer):
    """
    多模态检测训练器，基于Input字段路由系统的RGB+X模态训练流程。
    
    核心特色:
    - 支持配置驱动的多模态路由 (通过第5字段: 'RGB', 'X', 'Dual')
    - 早期融合: 6通道RGB+X输入统一处理
    - 中期融合: 独立RGB和X路径特征提取后融合
    - 晚期融合: 高层语义特征融合
    
    这个类继承DetectionTrainer，集成MultiModalRouter实现灵活的多模态数据流控制。
    支持RGB+深度、RGB+热红外等任意X模态组合的完整训练流程。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        初始化多模态检测训练器。
        
        Args:
            cfg (str | DictConfig, optional): 配置文件路径或配置字典
            overrides (dict, optional): 配置覆盖参数
            _callbacks (list, optional): 回调函数列表
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "detect"  # 确保任务类型正确
        super().__init__(cfg, overrides, _callbacks)
        
        # Get modality parameter from standard cfg system (与推理器保持一致)
        # Modality validation is handled by cfg system, no local validation needed
        self.modality = getattr(self.args, 'modality', None)
        
        # Initialize modality-specific attributes
        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None

        # Initialize logging control flags
        self._multimodal_config_logged = False  # 控制多模态配置日志只记录一次

        # Log initialization with modality information
        if self.modality:
            LOGGER.info(f"初始化MultiModalDetectionTrainer - 单模态训练模式: {self.modality}-only")
        else:
            LOGGER.info("初始化MultiModalDetectionTrainer - 双模态训练模式")

    def _parse_multimodal_config(self):
        """
        解析和验证数据配置文件中的多模态设置。
        
        解析data.yaml中的modalities和models字段，确保配置正确性，
        提供默认配置和友好的错误信息。
        
        优先支持用户指定的单模态训练参数。
        
        Returns:
            dict: 解析后的多模态配置
            
        Raises:
            ValueError: 当多模态配置不正确时
        """
        # 优先检查用户指定的modality参数（单模态训练）
        if self.modality:
            # 构建单模态配置
            if self.modality == 'rgb':
                # RGB单模态：使用RGB + 动态确定的X模态进行零填充
                x_modality = self._determine_x_modality_from_data()
                config = {
                    'models': ['rgb', x_modality],
                    'modalities': {
                        'rgb': 'images',
                        x_modality: f'images_{x_modality}'
                    }
                }
                LOGGER.info(f"RGB单模态训练，动态确定X模态: {x_modality}")
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
                    LOGGER.info(f"X模态单模态训练: {actual_x_modality}-only (从'X'解析)")
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
                    LOGGER.info(f"X模态单模态训练: {self.modality}-only")
            
            return config
        
        # 双模态训练：使用原有配置解析逻辑（优先从数据配置读取）
        config = self._get_default_multimodal_config()
        
        if not self.data:
            LOGGER.warning("训练器未提供数据配置，使用默认多模态配置: rgb+depth")
            return config
        
        # 解析modality_used字段（使用的模态组合）- 优先级最高
        if 'modality_used' in self.data:
            models = self.data['modality_used']

            # 验证modality_used格式
            if not isinstance(models, list):
                raise ValueError(f"data.yaml中的'modality_used'必须是列表格式，当前为: {type(models)}")

            if len(models) != 2:
                raise ValueError(f"多模态检测要求恰好2个模态，当前提供: {len(models)} - {models}")

            if 'rgb' not in models:
                raise ValueError(f"多模态组合必须包含'rgb'模态，当前: {models}")

            config['models'] = models
            LOGGER.info(f"从data.yaml的modality_used读取模态组合: {models}")
        elif 'models' in self.data:
            # 向后兼容：支持旧的models字段
            models = self.data['models']

            # 验证models格式
            if not isinstance(models, list):
                raise ValueError(f"data.yaml中的'models'必须是列表格式，当前为: {type(models)}")

            if len(models) != 2:
                raise ValueError(f"多模态检测要求恰好2个模态，当前提供: {len(models)} - {models}")

            if 'rgb' not in models:
                raise ValueError(f"多模态组合必须包含'rgb'模态，当前: {models}")

            config['models'] = models
            LOGGER.info(f"使用配置中的模态组合: {models}")
        else:
            LOGGER.debug(f"未找到'modality_used'或'models'配置，使用默认组合: {config['models']}")
        
        # 解析modality字段（模态路径映射）- 优先级最高
        if 'modality' in self.data:
            modalities = self.data['modality']

            # 验证modality格式
            if not isinstance(modalities, dict):
                raise ValueError(f"data.yaml中的'modality'必须是字典格式，当前为: {type(modalities)}")

            # 验证所有必需模态都有路径配置
            for modality in config['models']:
                if modality not in modalities:
                    if modality == 'rgb':
                        modalities[modality] = 'images'  # RGB默认路径
                        LOGGER.debug(f"'{modality}'模态路径未配置，使用默认: images")
                    else:
                        modalities[modality] = f'images_{modality}'  # X模态默认路径
                        LOGGER.debug(f"'{modality}'模态路径未配置，使用默认: images_{modality}")

            config['modalities'] = modalities
            LOGGER.info(f"从data.yaml的modality读取路径映射: {modalities}")
        elif 'modalities' in self.data:
            # 向后兼容：支持旧的modalities字段
            modalities = self.data['modalities']

            # 验证modalities格式
            if not isinstance(modalities, dict):
                raise ValueError(f"data.yaml中的'modalities'必须是字典格式，当前为: {type(modalities)}")

            # 验证所有必需模态都有路径配置
            for modality in config['models']:
                if modality not in modalities:
                    if modality == 'rgb':
                        modalities[modality] = 'images'  # RGB默认路径
                        LOGGER.debug(f"'{modality}'模态路径未配置，使用默认: images")
                    else:
                        modalities[modality] = f'images_{modality}'  # X模态默认路径
                        LOGGER.debug(f"'{modality}'模态路径未配置，使用默认: images_{modality}")

            config['modalities'] = modalities
            LOGGER.info(f"使用配置中的模态路径映射: {modalities}")
        else:
            # 为当前模态组合生成默认路径映射
            x_modality = [m for m in config['models'] if m != 'rgb'][0]
            config['modalities']['rgb'] = 'images'
            config['modalities'][x_modality] = f'images_{x_modality}'
            LOGGER.debug(f"未找到'modality'或'modalities'配置，生成默认路径映射: {config['modalities']}")
        
        # ✅ 移除硬编码限制，改为配置驱动
        # 用户通过配置明确指定了模态类型，系统应该信任并支持
        x_modality = [m for m in config['models'] if m != 'rgb'][0]
        LOGGER.info(f"✅ 使用用户配置的X模态: {x_modality} (配置驱动，支持任意模态类型)")
        
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
        if self.data and 'modality' in self.data:
            modality_paths = self.data['modality']
            if isinstance(modality_paths, dict) and modality_name in modality_paths:
                return modality_paths[modality_name]
        
        # 向后兼容：检查modalities字段
        if self.data and 'modalities' in self.data:
            modalities = self.data['modalities']
            if isinstance(modalities, dict) and modality_name in modalities:
                return modalities[modality_name]
        
        # 如果没有配置，使用默认格式
        return f'images_{modality_name}'
    
    def _determine_x_modality_from_data(self):
        """
        智能确定X模态类型，避免硬编码depth。

        优先级:
        1. 从data.yaml的modality_used字段读取（最高优先级）
        2. 从data.yaml的models字段读取
        3. 从modality字段推断
        4. 从数据目录结构推断
        5. 最后使用depth作为fallback

        Returns:
            str: X模态类型标识符
        """
        # 方法1: 从data.yaml的modality_used字段读取（最高优先级）
        if self.data and 'modality_used' in self.data:
            modality_used = self.data['modality_used']
            if isinstance(modality_used, list) and len(modality_used) >= 2:
                x_modalities = [m for m in modality_used if m != 'rgb']
                if x_modalities:
                    x_modality = x_modalities[0]
                    LOGGER.info(f"从data.yaml的modality_used读取X模态: {x_modality}")
                    return x_modality

        # 方法2: 从data.yaml的models字段读取（向后兼容）
        if self.data and 'models' in self.data:
            models = self.data['models']
            if isinstance(models, list) and len(models) >= 2:
                x_modalities = [m for m in models if m != 'rgb']
                if x_modalities:
                    x_modality = x_modalities[0]
                    LOGGER.info(f"从数据配置读取X模态: {x_modality}")
                    return x_modality
        
        # 方法3: 从modality字段推断（检查配置的模态类型）
        if self.data and 'modality' in self.data:
            modality = self.data['modality']
            if isinstance(modality, dict):
                x_modalities = [k for k in modality.keys() if k != 'rgb']
                if x_modalities:
                    x_modality = x_modalities[0]
                    LOGGER.info(f"从data.yaml的modality配置推断X模态: {x_modality}")
                    return x_modality

        # 方法4: 检查modalities配置（向后兼容）
        if self.data and 'modalities' in self.data:
            modalities = self.data['modalities']
            if isinstance(modalities, dict):
                x_modalities = [k for k in modalities.keys() if k != 'rgb']
                if x_modalities:
                    x_modality = x_modalities[0]
                    LOGGER.info(f"从modalities配置推断X模态: {x_modality}")
                    return x_modality

        # 方法5: 从数据目录结构推断（最低优先级）
        if self.data and 'path' in self.data:
            try:
                import os
                data_path = self.data['path']
                if os.path.exists(data_path):
                    # 查找images_xxx目录
                    for item in os.listdir(data_path):
                        if item.startswith('images_') and item != 'images':
                            x_modality = item.replace('images_', '')
                            LOGGER.info(f"从目录结构推断X模态: {x_modality}")
                            return x_modality
            except Exception as e:
                LOGGER.debug(f"目录结构推断失败: {e}")
        
        # Fallback: 使用depth作为默认值
        LOGGER.warning("无法自动确定X模态类型，使用默认值: depth")
        return 'depth'
    
    def _get_default_multimodal_config(self):
        """
        获取默认的多模态配置，优先从数据配置文件读取。

        Returns:
            dict: 默认多模态配置
        """
        # 方法1: 从data.yaml的modality_used和modality字段读取（最高优先级）
        if self.data and 'modality_used' in self.data:
            modality_used = self.data['modality_used']
            if isinstance(modality_used, list) and len(modality_used) >= 2:
                LOGGER.info(f"从data.yaml读取模态组合: {modality_used}")
                config = {
                    'models': modality_used,
                    'modalities': {}
                }

                # 从modality字段读取路径映射
                if 'modality' in self.data and isinstance(self.data['modality'], dict):
                    modality_paths = self.data['modality']
                    for mod in modality_used:
                        if mod in modality_paths:
                            config['modalities'][mod] = modality_paths[mod]
                        else:
                            # 如果modality字段中没有，使用默认路径
                            config['modalities'][mod] = 'images' if mod == 'rgb' else f'images_{mod}'
                    LOGGER.info(f"从data.yaml读取路径映射: {config['modalities']}")
                else:
                    # 如果没有modality字段，生成默认路径
                    for mod in modality_used:
                        config['modalities'][mod] = 'images' if mod == 'rgb' else f'images_{mod}'
                    LOGGER.info(f"生成默认路径映射: {config['modalities']}")

                return config

        # 方法2: 从data.yaml的models字段读取（向后兼容）
        if self.data and 'models' in self.data:
            models = self.data['models']
            if isinstance(models, list) and len(models) >= 2:
                LOGGER.info(f"从数据配置读取模态组合: {models}")
                config = {
                    'models': models,
                    'modalities': {
                        'rgb': 'images'  # RGB路径固定
                    }
                }
                # 为非RGB模态生成默认路径
                for modality in models:
                    if modality != 'rgb':
                        config['modalities'][modality] = f'images_{modality}'
                return config
        
        # 智能推断默认配置
        x_modality = self._determine_x_modality_from_data()
        config = {
            'models': ['rgb', x_modality],  # 动态确定的模态组合
            'modalities': {  # 动态生成的模态路径映射
                'rgb': 'images',
                x_modality: f'images_{x_modality}'
            }
        }
        LOGGER.info(f"生成默认多模态配置: rgb+{x_modality}")
        return config
    
    def _validate_modality_compatibility(self):
        """
        验证用户指定的modality参数与数据配置的兼容性。
        
        Raises:
            ValueError: 当modality参数与可用数据不兼容时
        """
        if not self.modality:
            return
        
        # 获取可用的模态
        available_modalities = []
        if hasattr(self, 'multimodal_config') and self.multimodal_config:
            available_modalities = self.multimodal_config.get('models', [])
        elif self.data and 'models' in self.data:
            available_modalities = self.data['models']
        
        # 验证modality是否在可用模态中
        if available_modalities:
            # 处理 'X' 特殊标记的验证
            if self.modality.upper() == 'X':
                # 'X' 是特殊标记，检查是否有非RGB的X模态
                x_modalities = [m for m in available_modalities if m != 'rgb']
                if x_modalities:
                    LOGGER.info(f"✅ 模态兼容性验证通过: '{self.modality}' 映射到 {x_modalities[0]}")
                else:
                    raise ValueError(
                        f"指定的modality '{self.modality}' 无法映射到有效的X模态。"
                        f"可用模态列表: {available_modalities}，但没有找到非RGB的X模态。"
                    )
            else:
                # 标准模态验证
                if self.modality not in available_modalities:
                    raise ValueError(
                        f"指定的modality '{self.modality}' 不在可用模态列表中: {available_modalities}。"
                        f"请检查数据配置或modality参数。"
                    )
                LOGGER.info(f"✅ 模态兼容性验证通过: {self.modality} 在可用模态 {available_modalities} 中")
        else:
            # 如果无法获取可用模态，仅给出警告
            LOGGER.warning(f"⚠️  无法验证modality '{self.modality}' 的兼容性，未找到可用模态配置")

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        构建多模态数据集，支持RGB+X模态的图像数据。

        Args:
            img_path (str): RGB图像路径
            mode (str): 数据集模式 ('train', 'val', 'test')
            batch (int, optional): 批次大小

        Returns:
            Dataset: 多模态数据集实例
        """
        # 解析多模态配置
        self.multimodal_config = self._parse_multimodal_config()

        # 验证modality兼容性
        self._validate_modality_compatibility()

        # 获取X模态信息
        x_modality = [m for m in self.multimodal_config['models'] if m != 'rgb'][0]
        x_modality_dir = self.multimodal_config['modalities'][x_modality]

        # 构建多模态数据集
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)

        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=gs,
            multi_modal_image=True,  # 启用图像多模态
            x_modality=x_modality,
            x_modality_dir=x_modality_dir,
            enable_self_modal_generation=getattr(self.args, 'enable_self_modal_generation', False)
        )

    def get_validator(self):
        """
        获取多模态检测验证器。

        Returns:
            MultiModalDetectionValidator: 多模态验证器实例
        """
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"

        # 延迟导入避免循环依赖
        from ultralytics.models.yolo.multimodal.val import MultiModalDetectionValidator

        return MultiModalDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        """
        绘制多模态训练样本，包括RGB和X模态图像。

        Args:
            batch (dict): 训练批次数据
            ni (int): 当前迭代次数
        """
        from ultralytics.utils.plotting import plot_images

        # 获取6通道图像数据
        images = batch["img"]  # [B, 6, H, W]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]

        # 分离RGB和X模态 (前3通道RGB，后3通道X)
        rgb_images = images[:, :3, :, :]  # [B, 3, H, W]
        x_images = images[:, 3:6, :, :]   # [B, 3, H, W]

        # 绘制RGB图像样本
        plot_images(
            rgb_images,
            batch["batch_idx"],
            cls,
            bboxes,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}_rgb.jpg",
            on_plot=self.on_plot,
        )

        # 绘制X模态图像样本
        plot_images(
            x_images,
            batch["batch_idx"],
            cls,
            bboxes,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}_x_modality.jpg",
            on_plot=self.on_plot,
        )

        # 可选：绘制融合可视化
        if hasattr(self.args, 'plot_fusion') and self.args.plot_fusion:
            # 简单的RGB+X融合可视化（平均融合）
            fusion_images = (rgb_images + x_images) / 2
            plot_images(
                fusion_images,
                batch["batch_idx"],
                cls,
                bboxes,
                paths=batch["im_file"],
                fname=self.save_dir / f"train_batch{ni}_fusion.jpg",
                on_plot=self.on_plot,
            )

    def plot_metrics(self):
        """
        绘制多模态训练指标图表。

        继承父类的指标绘制功能，添加多模态特定的指标可视化。
        """
        # 调用父类的指标绘制
        super().plot_metrics()

        # 可以在这里添加多模态特定的指标绘制
        # 例如：模态特定的损失、融合效果等
        LOGGER.info("多模态训练指标绘制完成")
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        获取多模态检测模型，确保使用正确的通道数。
        
        重写父类方法以支持多模态输入的动态通道数配置。
        
        Args:
            cfg (str, optional): 模型配置文件路径
            weights (str, optional): 预训练权重路径
            verbose (bool): 是否打印详细信息
            
        Returns:
            DetectionModel: 配置了正确通道数的检测模型
        """
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import RANK
        
        # 计算正确的输入通道数
        if self.is_dual_modal:
            # 双模态训练：从data配置中读取Xch
            x_channels = self.data.get('Xch', 3)
            channels = 3 + x_channels  # RGB(3) + X(Xch)
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"多模态模型初始化: RGB(3ch) + X({x_channels}ch) = {channels}ch总输入")
        else:
            # 单模态训练：始终使用3通道
            channels = 3
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"单模态模型初始化: {self.modality or 'RGB'}(3ch)")
        
        # 创建模型
        model = DetectionModel(cfg, nc=self.data["nc"], ch=channels, verbose=verbose and RANK == -1)
        
        # 更新multimodal_router的dataset_config（如果存在）
        if hasattr(model, 'multimodal_router') and model.multimodal_router:
            model.multimodal_router.update_dataset_config(self.data)
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"已更新MultiModalRouter的数据集配置，Xch={self.data.get('Xch', 3)}")
        
        if weights:
            model.load(weights)
        
        return model

    def save_model(self):
        """
        保存多模态模型，包含模态配置信息。

        重写父类方法，确保多模态配置信息被正确保存。
        """
        # 调用父类的模型保存
        super().save_model()

        # 保存多模态配置到模型检查点
        if hasattr(self, 'multimodal_config'):
            ckpt = torch_load(self.last, map_location='cpu')
            ckpt['multimodal_config'] = self.multimodal_config
            ckpt['modality'] = self.modality  # 保存单模态训练信息
            torch.save(ckpt, self.last)

            # 如果存在best模型，也更新它
            if self.best.exists():
                ckpt_best = torch_load(self.best, map_location='cpu')
                ckpt_best['multimodal_config'] = self.multimodal_config
                ckpt_best['modality'] = self.modality
                torch.save(ckpt_best, self.best)

    def final_eval(self):
        """
        执行最终评估，包含多模态特定的评估指标。

        注意：父类final_eval()没有返回值，所以这里主要是执行评估并记录多模态信息。
        """
        # 执行标准评估（父类方法没有返回值）
        super().final_eval()

        # 记录多模态特定信息
        if hasattr(self, 'multimodal_config') and self.multimodal_config:
            # 记录模态信息
            x_modality = [m for m in self.multimodal_config['models'] if m != 'rgb'][0]
            if self.modality:
                LOGGER.info(f"最终评估完成 - 单模态训练: {self.modality}-only")
            else:
                LOGGER.info(f"最终评估完成 - 双模态训练: RGB+{x_modality}")
        else:
            LOGGER.info("最终评估完成 - 多模态训练")
