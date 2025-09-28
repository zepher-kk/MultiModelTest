"""
EdgePrompt (边缘引导提示) 模块

专为YOLOMM渐进式边缘引导设计，实现边缘特征作为提示信号引导RGB特征增强
核心理念：RGB为主导，边缘为提示，实现渐进式特征增强

Author: YOLOMM Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


class EdgePrompt(nn.Module):
    """
    EdgePrompt (边缘引导提示) 模块
    
    专为YOLOMM渐进式边缘引导设计，将边缘特征作为提示信号引导RGB特征增强
    
    核心设计理念：
    1. RGB为主导：保持RGB特征的完整性和主导地位
    2. 边缘为提示：边缘特征提供空间和方向引导信息
    3. 智慧轻量化：通过设计智慧减少计算复杂度
    4. 渐进式增强：在RGB特征提取过程中渐进式注入边缘提示
    
    Args:
        rgb_channels (int): RGB特征通道数
        edge_channels (int, optional): 边缘特征通道数，默认为rgb_channels//2
        reduction (int): 通道降维比例，默认为4
        alpha (float): 可学习注入强度初始值，默认为0.5
        
    Input:
        x: 双模态特征列表 [rgb_features, edge_features]
           rgb_features: [B, C_rgb, H, W] RGB特征
           edge_features: [B, C_edge, H, W] 边缘特征
           
    Output:
        增强后的RGB特征 [B, C_rgb, H, W]，通道数保持不变
        
    Example:
        >>> edge_prompt = EdgePrompt(rgb_channels=256, edge_channels=128)
        >>> rgb_feat = torch.randn(2, 256, 40, 40)
        >>> edge_feat = torch.randn(2, 128, 40, 40)
        >>> enhanced_rgb = edge_prompt([rgb_feat, edge_feat])
        >>> print(enhanced_rgb.shape)  # torch.Size([2, 256, 40, 40])
    """
    
    def __init__(self, rgb_channels, edge_channels=None, reduction=4, alpha=0.5):
        super(EdgePrompt, self).__init__()
        
        # 参数设置
        self.rgb_channels = rgb_channels
        self.edge_channels = edge_channels or rgb_channels // 2
        self.reduction = reduction
        
        # 1. 边缘特征预处理 - 轻量化设计
        self.edge_preprocess = nn.Sequential(
            nn.Conv2d(self.edge_channels, self.edge_channels, 3, 1, 1, groups=self.edge_channels),  # 深度可分离
            nn.BatchNorm2d(self.edge_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.edge_channels, self.edge_channels, 1),  # 点卷积
            nn.BatchNorm2d(self.edge_channels)
        )
        
        # 2. 边缘特征扩展到RGB通道数 - 智慧设计
        self.edge_expand = nn.Sequential(
            nn.Conv2d(self.edge_channels, rgb_channels, 1),
            nn.BatchNorm2d(rgb_channels)
        )
        
        # 3. 统一的方向感知核心 - 融合EMA思想
        self.direction_core = DirectionAwareCore(rgb_channels, reduction)
        
        # 4. 门控保护机制 - 融合CA思想
        self.gate_protection = GateProtection(rgb_channels)
        
        # 5. 可学习注入强度参数
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        
        # 6. 输出调整
        self.output_adjust = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, 1),
            nn.BatchNorm2d(rgb_channels)
        )
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 双模态特征列表 [rgb_features, edge_features]
               
        Returns:
            增强后的RGB特征 [B, C_rgb, H, W]
        """
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(f"EdgePrompt期望输入为[RGB, Edge]双模态特征，实际输入: {type(x)}")
        
        rgb_feat, edge_feat = x[0], x[1]
        
        # 验证输入维度
        if rgb_feat.dim() != 4 or edge_feat.dim() != 4:
            raise ValueError(f"输入特征必须为4维张量，实际: RGB={rgb_feat.dim()}D, Edge={edge_feat.dim()}D")
        
        # 自适应尺寸匹配 - 如果尺寸不匹配，将边缘特征调整到RGB特征尺寸
        if rgb_feat.shape[2:] != edge_feat.shape[2:]:
            original_size = edge_feat.shape[2:]  # 保存原始尺寸
            target_size = rgb_feat.shape[2:]  # (H, W)
            edge_feat = F.interpolate(
                edge_feat, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
            # print(f"🔧 EdgePrompt: 自动调整边缘特征尺寸 {original_size} -> {target_size}")
        
        # 自适应通道匹配 - 如果边缘特征通道数与预期不匹配，动态调整
        actual_edge_channels = edge_feat.shape[1]
        if actual_edge_channels != self.edge_channels:
            print(f"🔧 EdgePrompt: 检测到边缘通道数不匹配，期望{self.edge_channels}，实际{actual_edge_channels}")
            # 动态调整边缘特征到预期通道数
            if actual_edge_channels < self.edge_channels:
                # 通道数不足，使用1x1卷积扩展
                padding_channels = self.edge_channels - actual_edge_channels
                padding = torch.zeros(edge_feat.shape[0], padding_channels, *edge_feat.shape[2:], 
                                    device=edge_feat.device, dtype=edge_feat.dtype)
                edge_feat = torch.cat([edge_feat, padding], dim=1)
                print(f"🔧 EdgePrompt: 通道填充 {actual_edge_channels} -> {self.edge_channels}")
            else:
                # 通道数过多，使用1x1卷积降维
                edge_feat = F.conv2d(edge_feat, 
                                   torch.ones(self.edge_channels, actual_edge_channels, 1, 1, 
                                            device=edge_feat.device, dtype=edge_feat.dtype) / actual_edge_channels)
                print(f"🔧 EdgePrompt: 通道降维 {actual_edge_channels} -> {self.edge_channels}")
        
        # 1. 边缘特征预处理
        processed_edge = self.edge_preprocess(edge_feat)
        
        # 2. 边缘特征扩展到RGB通道数
        expanded_edge = self.edge_expand(processed_edge)
        
        # 3. 方向感知处理
        direction_guidance = self.direction_core(expanded_edge)
        
        # 4. 门控保护RGB特征
        protected_rgb = self.gate_protection(rgb_feat, direction_guidance)
        
        # 5. 可学习强度注入
        enhanced_rgb = rgb_feat + self.alpha * protected_rgb
        
        # 6. 输出调整
        output = self.output_adjust(enhanced_rgb)
        
        return output


class DirectionAwareCore(nn.Module):
    """
    方向感知核心模块
    
    融合EMA多尺度思想，用统一的核心提取边缘的方向信息
    不分别处理X/Y方向，而是用一个核心同时感知所有方向
    """
    
    def __init__(self, channels, reduction=4):
        super(DirectionAwareCore, self).__init__()
        
        self.channels = channels
        self.reduction = reduction
        
        # 统一方向感知 - 替代传统的X/Y分离处理
        self.unified_direction = nn.Sequential(
            # 多尺度感受野 - 融合EMA思想
            nn.Conv2d(channels, channels // reduction, 1),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            
            # 方向敏感卷积组合
            nn.Conv2d(channels // reduction, channels // reduction, (1, 3), padding=(0, 1)),  # 水平
            nn.Conv2d(channels // reduction, channels // reduction, (3, 1), padding=(1, 0)),  # 垂直
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            
            # 恢复通道数
            nn.Conv2d(channels // reduction, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()  # 生成0-1权重
        )
        
    def forward(self, edge_feat):
        """
        提取边缘的方向感知信息
        
        Args:
            edge_feat: 边缘特征 [B, C, H, W]
            
        Returns:
            方向引导权重 [B, C, H, W]
        """
        direction_weights = self.unified_direction(edge_feat)
        return direction_weights


class GateProtection(nn.Module):
    """
    门控保护机制
    
    融合CA坐标注意力思想，保护RGB特征的重要信息
    通过门控机制控制边缘信息的注入程度
    """
    
    def __init__(self, channels):
        super(GateProtection, self).__init__()
        
        self.channels = channels
        
        # 全局上下文提取 - 融合CA思想
        self.global_pool_h = nn.AdaptiveAvgPool2d((None, 1))  # H方向池化
        self.global_pool_w = nn.AdaptiveAvgPool2d((1, None))  # W方向池化
        
        # 门控信号生成
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 3, channels // 4, 1),  # RGB + H池化 + W池化
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()  # 门控信号
        )
        
    def forward(self, rgb_feat, direction_guidance):
        """
        门控保护RGB特征
        
        Args:
            rgb_feat: RGB特征 [B, C, H, W]
            direction_guidance: 方向引导权重 [B, C, H, W]
            
        Returns:
            保护后的增强特征 [B, C, H, W]
        """
        B, C, H, W = rgb_feat.shape
        
        # 提取全局上下文 - 融合CA坐标注意力思想
        pool_h = self.global_pool_h(rgb_feat)  # [B, C, H, 1]
        pool_w = self.global_pool_w(rgb_feat)  # [B, C, 1, W]
        
        # 扩展到原始尺寸
        pool_h_expanded = pool_h.expand(-1, -1, -1, W)  # [B, C, H, W]
        pool_w_expanded = pool_w.expand(-1, -1, H, -1)  # [B, C, H, W]
        
        # 组合特征
        combined_feat = torch.cat([rgb_feat, pool_h_expanded, pool_w_expanded], dim=1)  # [B, 3C, H, W]
        
        # 生成门控信号
        gate = self.gate_conv(combined_feat)  # [B, C, H, W]
        
        # 门控保护：只允许有益的边缘信息通过
        protected_enhancement = gate * direction_guidance * rgb_feat
        
        return protected_enhancement


 