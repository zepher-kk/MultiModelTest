"""
BiFocus (Bi-directional Decoupled Focus) 双向解耦聚焦模块

专为YOLOMM骨干网络设计，通过双向像素分组卷积扩大感受野
可集成到C2f模块中，提升特征提取能力

Author: YOLOMM Team  
Date: 2024
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck


class BiFocus(nn.Module):
    """
    BiFocus 双向解耦聚焦模块
    
    通过水平和垂直方向的像素分组卷积，扩大网络感受野
    将像素分为两组进行卷积，每组同时关注相邻和远程像素
    
    Args:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        
    Input:
        x: [B, C1, H, W] 输入特征图
        
    Output:
        [B, C2, H, W] 增强后的特征图
        
    Example:
        >>> bifocus = BiFocus(256, 256)
        >>> x = torch.randn(2, 256, 40, 40)
        >>> output = bifocus(x)
        >>> print(output.shape)  # torch.Size([2, 256, 40, 40])
    """
    
    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.focus_h = FocusH(c1, c1, 3, 1)
        self.focus_v = FocusV(c1, c1, 3, 1)
        self.depth_wise = DepthWiseConv(3 * c1, c2, 3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, C, H, W]
            
        Returns:
            增强后的特征图 [B, C2, H, W]
        """
        # 原始特征 + 水平聚焦 + 垂直聚焦
        h_focused = self.focus_h(x)
        v_focused = self.focus_v(x)
        
        # 在通道维度拼接
        combined = torch.cat([x, h_focused, v_focused], dim=1)
        
        # 深度可分离卷积融合
        output = self.depth_wise(combined)
        
        return output


class FocusH(nn.Module):
    """
    FocusH 水平方向聚焦模块
    
    将像素按水平方向分组，每组进行独立卷积后重新组合
    """
    
    def __init__(self, c1: int, c2: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.c2 = c2
        self.conv1 = Conv(c1, c2, kernel, stride)
        self.conv2 = Conv(c1, c2, kernel, stride)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        水平方向像素分组聚焦
        
        Args:
            x: 输入特征图 [B, C, H, W]
            
        Returns:
            水平聚焦后的特征图 [B, C2, H, W]
        """
        b, c, h, w = x.shape
        
        # 初始化输出和中间张量
        result = torch.zeros(size=[b, self.c2, h, w], device=x.device, dtype=x.dtype)
        x1 = torch.zeros(size=[b, c, h, w // 2], device=x.device, dtype=x.dtype)
        x2 = torch.zeros(size=[b, c, h, w // 2], device=x.device, dtype=x.dtype)
        
        # 像素分组：奇偶位置分离
        x1[..., ::2, :] = x[..., ::2, ::2]  # 偶行偶列 -> 偶行
        x1[..., 1::2, :] = x[..., 1::2, 1::2]  # 奇行奇列 -> 奇行
        x2[..., ::2, :] = x[..., ::2, 1::2]  # 偶行奇列 -> 偶行  
        x2[..., 1::2, :] = x[..., 1::2, ::2]  # 奇行偶列 -> 奇行
        
        # 分别卷积
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        
        # 重新组合
        result[..., ::2, ::2] = x1[..., ::2, :]
        result[..., 1::2, 1::2] = x1[..., 1::2, :]
        result[..., ::2, 1::2] = x2[..., ::2, :]
        result[..., 1::2, ::2] = x2[..., 1::2, :]
        
        return result


class FocusV(nn.Module):
    """
    FocusV 垂直方向聚焦模块
    
    将像素按垂直方向分组，每组进行独立卷积后重新组合
    """
    
    def __init__(self, c1: int, c2: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.c2 = c2
        self.conv1 = Conv(c1, c2, kernel, stride)
        self.conv2 = Conv(c1, c2, kernel, stride)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        垂直方向像素分组聚焦
        
        Args:
            x: 输入特征图 [B, C, H, W]
            
        Returns:
            垂直聚焦后的特征图 [B, C2, H, W]
        """
        b, c, h, w = x.shape
        
        # 初始化输出和中间张量
        result = torch.zeros(size=[b, self.c2, h, w], device=x.device, dtype=x.dtype)
        x1 = torch.zeros(size=[b, c, h // 2, w], device=x.device, dtype=x.dtype)
        x2 = torch.zeros(size=[b, c, h // 2, w], device=x.device, dtype=x.dtype)
        
        # 像素分组：奇偶位置分离
        x1[..., ::2] = x[..., ::2, ::2]  # 偶行偶列 -> 偶列
        x1[..., 1::2] = x[..., 1::2, 1::2]  # 奇行奇列 -> 奇列
        x2[..., ::2] = x[..., 1::2, ::2]  # 奇行偶列 -> 偶列
        x2[..., 1::2] = x[..., ::2, 1::2]  # 偶行奇列 -> 奇列
        
        # 分别卷积
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        
        # 重新组合
        result[..., ::2, ::2] = x1[..., ::2]
        result[..., 1::2, 1::2] = x1[..., 1::2]
        result[..., 1::2, ::2] = x2[..., ::2]
        result[..., ::2, 1::2] = x2[..., 1::2]
        
        return result


class DepthWiseConv(nn.Module):
    """
    DepthWiseConv 深度可分离卷积
    
    先进行深度卷积，再进行逐点卷积，减少参数量和计算量
    """
    
    def __init__(self, in_channel: int, out_channel: int, kernel: int):
        super().__init__()
        self.depth_conv = Conv(in_channel, in_channel, kernel, 1, kernel//2, in_channel)
        self.point_conv = Conv(in_channel, out_channel, 1, 1, 0, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        深度可分离卷积前向传播
        
        Args:
            x: 输入特征图
            
        Returns:
            卷积后的特征图
        """
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out


class C2f_BiFocus(nn.Module):
    """
    C2f_BiFocus 集成BiFocus的C2f模块
    
    在标准C2f模块基础上集成BiFocus，提升特征提取能力
    适用于YOLOMM骨干网络的早期层
    
    Args:
        c1 (int): 输入通道数
        c2 (int): 输出通道数  
        n (int): Bottleneck重复次数
        shortcut (bool): 是否使用残差连接
        g (int): 分组卷积组数
        e (float): 通道扩展比例
        
    Example:
        >>> c2f_bifocus = C2f_BiFocus(256, 256, n=3)
        >>> x = torch.randn(2, 256, 80, 80)
        >>> output = c2f_bifocus(x)
        >>> print(output.shape)  # torch.Size([2, 256, 80, 80])
    """
    
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, 
                 g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) 
            for _ in range(n)
        )
        
        # 集成BiFocus模块
        self.bifocus = BiFocus(c2, c2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, C1, H, W]
            
        Returns:
            BiFocus增强后的特征图 [B, C2, H, W]
        """
        # 标准C2f流程
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        
        # BiFocus增强
        return self.bifocus(y)
        
    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用split()而非chunk()的前向传播版本
        
        Args:
            x: 输入特征图
            
        Returns:
            处理后的特征图
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1)) 