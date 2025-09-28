# Ultralytics Multi-modal Visualization Utilities
"""
多模态可视化工具函数和类。

包含图像加载、预处理、保存等基础功能，以及用于特征提取的HookManager类。
"""

import os
from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def load_image(
    image: Union[str, Path, np.ndarray, torch.Tensor, Image.Image],
    mode: str = 'RGB'
) -> np.ndarray:
    """
    加载图像并转换为numpy数组。
    
    Args:
        image: 图像输入，支持文件路径、numpy数组、torch张量或PIL图像
        mode: 图像模式，'RGB'或'BGR'，默认'RGB'
        
    Returns:
        np.ndarray: 加载的图像数组，shape为(H, W, C)
        
    Raises:
        ValueError: 不支持的输入类型或图像格式
    """
    if isinstance(image, (str, Path)):
        # 从文件路径加载
        image_path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
        # 使用cv2加载图像
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"无法加载图像: {image_path}")
            
        # cv2默认是BGR，如果需要RGB则转换
        if mode == 'RGB' and len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
    elif isinstance(image, np.ndarray):
        # 已经是numpy数组
        img = image.copy()
        
    elif isinstance(image, torch.Tensor):
        # 从torch张量转换
        img = image.cpu().numpy()
        
        # 处理通道顺序：CHW -> HWC
        if img.ndim == 3 and img.shape[0] in [1, 3, 6]:
            img = np.transpose(img, (1, 2, 0))
        elif img.ndim == 4:
            # 批次维度，取第一个样本
            img = img[0]
            if img.shape[0] in [1, 3, 6]:
                img = np.transpose(img, (1, 2, 0))
                
    elif isinstance(image, Image.Image):
        # 从PIL图像转换
        if mode == 'RGB':
            img = np.array(image.convert('RGB'))
        else:
            img = np.array(image)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
    else:
        raise ValueError(f"不支持的图像类型: {type(image)}")
        
    return img.astype(np.float32)


def preprocess_image(
    image: np.ndarray,
    size: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
    to_tensor: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    """
    预处理图像，包括调整大小、归一化和转换为张量。
    
    Args:
        image: 输入图像，numpy数组格式 (H, W, C)
        size: 目标尺寸 (height, width)，如果为None则不调整大小
        normalize: 是否归一化到[0, 1]
        to_tensor: 是否转换为torch张量
        
    Returns:
        处理后的图像，numpy数组或torch张量
    """
    img = image.copy()
    
    # 调整大小
    if size is not None:
        img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    
    # 归一化
    if normalize:
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.max() > 1.0:
            img = img / img.max()
    
    # 转换为张量
    if to_tensor:
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        
    return img


def normalize_image(
    image: Union[np.ndarray, torch.Tensor],
    mean: Optional[Union[float, List[float]]] = None,
    std: Optional[Union[float, List[float]]] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    使用给定的均值和标准差归一化图像。
    
    Args:
        image: 输入图像
        mean: 均值，可以是单个值或每个通道的值列表
        std: 标准差，可以是单个值或每个通道的值列表
        
    Returns:
        归一化后的图像
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]  # ImageNet默认值
    if std is None:
        std = [0.229, 0.224, 0.225]   # ImageNet默认值
        
    if isinstance(image, np.ndarray):
        # numpy处理
        img = image.copy()
        if isinstance(mean, (list, tuple)):
            mean = np.array(mean).reshape(1, 1, -1)
        if isinstance(std, (list, tuple)):
            std = np.array(std).reshape(1, 1, -1)
        return (img - mean) / std
        
    else:
        # torch处理
        img = image.clone()
        if isinstance(mean, (list, tuple)):
            mean = torch.tensor(mean).view(-1, 1, 1)
        if isinstance(std, (list, tuple)):
            std = torch.tensor(std).view(-1, 1, 1)
            
        # 确保设备一致
        if img.device != mean.device:
            mean = mean.to(img.device)
            std = std.to(img.device)
            
        return (img - mean) / std


def save_image(
    image: Union[np.ndarray, torch.Tensor],
    path: Union[str, Path],
    denormalize: bool = False,
    mean: Optional[Union[float, List[float]]] = None,
    std: Optional[Union[float, List[float]]] = None
) -> None:
    """
    保存图像到文件。
    
    Args:
        image: 要保存的图像
        path: 保存路径
        denormalize: 是否反归一化
        mean: 用于反归一化的均值
        std: 用于反归一化的标准差
    """
    # 确保输出目录存在
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # 处理torch张量
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        
    # 处理维度
    if image.ndim == 3:
        if image.shape[0] in [1, 3, 6]:
            # CHW -> HWC
            image = np.transpose(image, (1, 2, 0))
    elif image.ndim == 2:
        # 灰度图像
        image = image[:, :, np.newaxis]
        
    # 反归一化
    if denormalize:
        if mean is not None and std is not None:
            if isinstance(mean, (list, tuple)):
                mean = np.array(mean).reshape(1, 1, -1)
            if isinstance(std, (list, tuple)):
                std = np.array(std).reshape(1, 1, -1)
            image = image * std + mean
    
    # 转换为uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).clip(0, 255)
        else:
            image = image.clip(0, 255)
        image = image.astype(np.uint8)
    
    # 保存图像
    if image.shape[2] == 1:
        # 灰度图像
        cv2.imwrite(str(path), image[:, :, 0])
    elif image.shape[2] == 3:
        # RGB图像，转换为BGR保存
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        # 多通道图像，直接保存
        cv2.imwrite(str(path), image)


class HookManager:
    """
    管理模型层的前向和反向钩子，用于特征提取和梯度分析。
    
    该类提供了注册、管理和清理PyTorch模型钩子的功能，
    支持在前向传播和反向传播过程中捕获中间特征和梯度。
    """
    
    def __init__(self, model: nn.Module):
        """
        初始化HookManager。
        
        Args:
            model: PyTorch模型实例
        """
        self.model = model  # 存储模型引用，避免params显示N/A问题
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.features: Dict[int, torch.Tensor] = {}
        self.gradients: Dict[int, torch.Tensor] = {}
        
    def register_forward_hook(
        self,
        layer_idx: int,
        hook_fn: Optional[Any] = None
    ) -> torch.utils.hooks.RemovableHandle:
        """
        在指定层注册前向钩子。
        
        Args:
            layer_idx: 层索引
            hook_fn: 自定义钩子函数，如果为None则使用默认函数
            
        Returns:
            RemovableHandle: 可移除的钩子句柄
            
        Raises:
            IndexError: 层索引超出范围
        """
        # 验证层索引
        if not hasattr(self.model, 'model'):
            raise AttributeError("模型没有'model'属性")
            
        if layer_idx < 0 or layer_idx >= len(self.model.model):
            raise IndexError(f"层索引 {layer_idx} 超出范围 [0, {len(self.model.model)-1}]")
        
        # 默认钩子函数：保存特征
        def default_forward_hook(module, input, output):
            self.features[layer_idx] = output.detach()
            
        # 使用提供的钩子函数或默认函数
        if hook_fn is None:
            hook_fn = default_forward_hook
            
        # 注册钩子
        handle = self.model.model[layer_idx].register_forward_hook(hook_fn)
        self.hooks.append(handle)
        
        return handle
        
    def register_backward_hook(
        self,
        layer_idx: int,
        hook_fn: Optional[Any] = None
    ) -> torch.utils.hooks.RemovableHandle:
        """
        在指定层注册反向钩子。
        
        Args:
            layer_idx: 层索引
            hook_fn: 自定义钩子函数，如果为None则使用默认函数
            
        Returns:
            RemovableHandle: 可移除的钩子句柄
            
        Raises:
            IndexError: 层索引超出范围
        """
        # 验证层索引
        if not hasattr(self.model, 'model'):
            raise AttributeError("模型没有'model'属性")
            
        if layer_idx < 0 or layer_idx >= len(self.model.model):
            raise IndexError(f"层索引 {layer_idx} 超出范围 [0, {len(self.model.model)-1}]")
        
        # 默认钩子函数：保存梯度
        def default_backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                self.gradients[layer_idx] = grad_output[0].detach()
            else:
                self.gradients[layer_idx] = grad_output.detach()
                
        # 使用提供的钩子函数或默认函数
        if hook_fn is None:
            hook_fn = default_backward_hook
            
        # 注册钩子
        handle = self.model.model[layer_idx].register_backward_hook(hook_fn)
        self.hooks.append(handle)
        
        return handle
        
    def clear_hooks(self) -> None:
        """
        清理所有注册的钩子。
        
        这个方法会移除所有已注册的前向和反向钩子，
        并清空保存的特征和梯度。
        """
        # 移除所有钩子
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
        # 清空保存的数据
        self.features.clear()
        self.gradients.clear()
        
    def get_features(self, layer_idx: Optional[int] = None) -> Union[Dict[int, torch.Tensor], torch.Tensor]:
        """
        获取保存的特征。
        
        Args:
            layer_idx: 指定层索引，如果为None则返回所有特征
            
        Returns:
            指定层的特征或所有特征字典
        """
        if layer_idx is not None:
            return self.features.get(layer_idx)
        return self.features.copy()
        
    def get_gradients(self, layer_idx: Optional[int] = None) -> Union[Dict[int, torch.Tensor], torch.Tensor]:
        """
        获取保存的梯度。
        
        Args:
            layer_idx: 指定层索引，如果为None则返回所有梯度
            
        Returns:
            指定层的梯度或所有梯度字典
        """
        if layer_idx is not None:
            return self.gradients.get(layer_idx)
        return self.gradients.copy()
        
    def __del__(self):
        """
        析构函数，确保钩子被正确清理。
        
        在对象被销毁时自动调用，防止内存泄漏。
        """
        self.clear_hooks()