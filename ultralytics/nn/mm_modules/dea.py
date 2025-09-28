"""
DEA (Dual Enhancement Attention) 双增强注意力融合模块

专为YOLOMM中期融合设计，替代传统的 Concat + C3k2 + C2PSA 组合
实现RGB和X模态的跨模态交叉增强

Author: YOLOMM Team
Date: 2024
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class DEA(nn.Module):
    """
    DEA (Dual Enhancement Attention) 双增强注意力融合模块
    
    专为YOLOMM中期融合设计，替代 Concat + C3k2 + C2PSA 组合
    实现RGB和X模态的跨模态交叉增强
    
    Args:
        channel (int): 输入特征通道数 (RGB和X模态通道数需相同)
        reduction (int): 通道注意力降维比例 (默认16)
        kernel_size (int): 卷积金字塔的参考尺寸 (自动适配特征图大小)
        
    Input:
        x: 双模态特征列表 [rgb_features, x_features]
           rgb_features: [B, C, H, W] RGB特征
           x_features: [B, C, H, W] X模态特征
           
    Output:
        融合后的特征 [B, C, H, W]，通道数保持不变
        
    Example:
        >>> dea = DEA(channel=512, reduction=16, kernel_size=40)
        >>> rgb_feat = torch.randn(2, 512, 40, 40)
        >>> x_feat = torch.randn(2, 512, 40, 40)
        >>> output = dea([rgb_feat, x_feat])
        >>> print(output.shape)  # torch.Size([2, 512, 40, 40])
    """
    
    def __init__(self, channel=512, reduction=16, kernel_size=None):
        super().__init__()
        
        # 自动计算卷积金字塔参数
        if kernel_size is None:
            # 根据常见特征图尺寸自适应
            kernel_size = 40  # P4层典型尺寸
        
        self.deca = DECA(channel, kernel_size, reduction=reduction)
        self.depa = DEPA(channel)
        self.output_conv = Conv(channel, channel, 1)  # 保持输出通道数
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 双模态特征列表 [rgb_features, x_features]
               
        Returns:
            融合后的特征 [B, C, H, W]，通道数保持不变
        """
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(f"DEA期望输入为[RGB, X]双模态特征，实际输入: {type(x)}")
        
        rgb_feat, x_feat = x[0], x[1]
        
        # 验证输入形状
        if rgb_feat.shape != x_feat.shape:
            raise ValueError(f"RGB和X模态特征形状不匹配: {rgb_feat.shape} vs {x_feat.shape}")
        
        # DECA: 双语义增强通道注意力
        enhanced_rgb, enhanced_x = self.deca([rgb_feat, x_feat])
        
        # DEPA: 双空间增强像素注意力  
        final_rgb, final_x = self.depa([enhanced_rgb, enhanced_x])
        
        # 融合输出 (替代简单Add/Concat)
        fused = self.act(final_rgb + final_x)
        output = self.output_conv(fused)
        
        return output


class DECA(nn.Module):
    """
    DECA (Dual Enhancement Channel Attention) 双语义增强通道注意力
    
    实现跨模态通道权重交换：RGB特征用X模态权重增强，X模态特征用RGB权重增强
    """
    
    def __init__(self, channel=512, kernel_size=40, p_kernel=None, reduction=16):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 通道注意力网络
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        
        # 特征压缩层
        self.compress = Conv(channel * 2, channel, 3)
        
        # 自适应卷积金字塔
        if p_kernel is None:
            p_kernel = self._auto_adjust_kernels(kernel_size)
        
        kernel1, kernel2 = p_kernel
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel1, kernel1, 0, groups=channel), 
            nn.SiLU()
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel2, kernel2, 0, groups=channel), 
            nn.SiLU()
        )
        self.conv_c3 = nn.Sequential(
            nn.Conv2d(channel, channel, 
                      max(1, int(self.kernel_size/kernel1/kernel2)), 
                      max(1, int(self.kernel_size/kernel1/kernel2)), 
                      0, groups=channel),
            nn.SiLU()
        )
        
        self.act = nn.Sigmoid()
    
    def _auto_adjust_kernels(self, kernel_size):
        """根据特征图尺寸自动调整卷积核参数"""
        if kernel_size >= 40:
            return [5, 4]  # P3/P4层
        elif kernel_size >= 20:
            return [4, 2]  # P4/P5层
        else:
            return [2, 2]  # P5层
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [rgb_features, x_features]
            
        Returns:
            (enhanced_rgb, enhanced_x): 跨模态增强后的特征
        """
        rgb_feat, x_feat = x[0], x[1]
        b, c, h, w = rgb_feat.size()
        
        # 生成跨模态通道权重
        w_rgb = self.avg_pool(rgb_feat).view(b, c)
        w_x = self.avg_pool(x_feat).view(b, c)
        w_rgb = self.fc(w_rgb).view(b, c, 1, 1)
        w_x = self.fc(w_x).view(b, c, 1, 1)
        
        # 全局上下文建模
        fused_global = self.compress(torch.cat([rgb_feat, x_feat], 1))
        
        if min(h, w) >= self.kernel_size:
            global_context = self.conv_c3(self.conv_c2(self.conv_c1(fused_global)))
        else:
            # 小尺寸特征图使用全局池化
            global_context = torch.mean(fused_global, dim=[2, 3], keepdim=True)
        
        # 跨模态增强 (核心创新)
        enhanced_rgb = rgb_feat * (self.act(w_x * global_context)).expand_as(rgb_feat)
        enhanced_x = x_feat * (self.act(w_rgb * global_context)).expand_as(x_feat)
        
        return enhanced_rgb, enhanced_x


class DEPA(nn.Module):
    """
    DEPA (Dual Enhancement Pixel Attention) 双空间增强像素注意力
    
    实现跨模态空间权重交换：RGB特征用X模态空间权重增强，X模态特征用RGB空间权重增强
    """
    
    def __init__(self, channel=512, m_kernel=None):
        super().__init__()
        
        # 空间注意力卷积
        self.conv1 = Conv(2, 1, 5)  # RGB空间权重
        self.conv2 = Conv(2, 1, 5)  # X模态空间权重
        self.compress1 = Conv(channel, 1, 3)  # RGB压缩
        self.compress2 = Conv(channel, 1, 3)  # X模态压缩
        
        # 多尺度空间特征提取
        if m_kernel is None:
            m_kernel = [3, 7]  # 多尺度感受野
        
        self.cv_rgb1 = Conv(channel, 1, m_kernel[0])
        self.cv_rgb2 = Conv(channel, 1, m_kernel[1])
        self.cv_x1 = Conv(channel, 1, m_kernel[0])
        self.cv_x2 = Conv(channel, 1, m_kernel[1])
        
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [rgb_features, x_features]
            
        Returns:
            (enhanced_rgb, enhanced_x): 跨模态空间增强后的特征
        """
        rgb_feat, x_feat = x[0], x[1]
        
        # 多尺度空间特征
        spatial_rgb = self.conv1(torch.cat([
            self.cv_rgb1(rgb_feat), 
            self.cv_rgb2(rgb_feat)
        ], 1))
        
        spatial_x = self.conv2(torch.cat([
            self.cv_x1(x_feat), 
            self.cv_x2(x_feat)
        ], 1))
        
        # 全局空间上下文
        global_spatial = self.act(
            self.compress1(rgb_feat) + self.compress2(x_feat)
        )
        
        # 生成最终空间权重
        weight_rgb = self.act(global_spatial + spatial_rgb)
        weight_x = self.act(global_spatial + spatial_x)
        
        # 跨模态空间增强
        enhanced_rgb = rgb_feat * weight_x.expand_as(rgb_feat)
        enhanced_x = x_feat * weight_rgb.expand_as(x_feat)
        
        return enhanced_rgb, enhanced_x 