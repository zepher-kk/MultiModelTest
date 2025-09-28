"""
多模态模块解析器 (Multimodal Module Parser)

专为YOLOMM插入式设计，将多模态相关的解析逻辑从tasks.py中分离
实现模块化、可插拔的多模态处理

Author: YOLOMM Team
Date: 2024
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple, Optional
from ultralytics.utils.ops import make_divisible


class MultiModalParser:
    """
    多模态模块解析器
    
    负责处理所有多模态相关的模块解析逻辑，
    为tasks.py提供统一的接口，实现插入式设计
    """
    
    def __init__(self):
        """初始化多模态解析器"""
        self._register_modules()
    
    def _register_modules(self):
        """注册所有多模态模块"""
        try:
            # 动态导入多模态模块，避免循环导入
            from ultralytics.nn.mm_modules.dea import DEA, DECA, DEPA
            from ultralytics.nn.mm_modules.edge_prompt import EdgePrompt
            from ultralytics.nn.mm_modules.bifocus import BiFocus, C2f_BiFocus
            
            # 注册多模态模块映射
            self.multimodal_modules = {
                'DEA': DEA,
                'DECA': DECA,
                'DEPA': DEPA,
                'EdgePrompt': EdgePrompt,
                'BiFocus': BiFocus,
                'C2f_BiFocus': C2f_BiFocus,
            }
            
            # 注册处理函数映射
            self.module_handlers = {
                DEA: self._handle_dea_family,
                DECA: self._handle_dea_family,
                DEPA: self._handle_dea_family,
                EdgePrompt: self._handle_edge_prompt_family,
                BiFocus: self._handle_bifocus_family,
                C2f_BiFocus: self._handle_c2f_bifocus_family,
            }
            
            self.available = True
            
        except ImportError as e:
            # 多模态模块不可用时的优雅降级
            self.multimodal_modules = {}
            self.module_handlers = {}
            self.available = False
            print(f"警告: 多模态模块不可用 - {e}")
    
    def is_multimodal_module(self, module_name: str) -> bool:
        """
        检查模块是否为多模态模块
        
        Args:
            module_name: 模块名称
            
        Returns:
            bool: 是否为多模态模块
        """
        return module_name in self.multimodal_modules
    
    def is_available(self) -> bool:
        """检查多模态功能是否可用"""
        return self.available
    
    def process_multimodal_module(
        self, 
        module_class: type, 
        f: List[int], 
        args: List[Any], 
        ch: List[int],
        width: float,
        max_channels: float,
        d: Dict[str, Any]
    ) -> Tuple[int, List[Any]]:
        """
        处理多模态模块的参数解析
        
        Args:
            module_class: 模块类
            f: 输入层索引列表
            args: 原始参数列表
            ch: 通道数列表
            width: 宽度倍数
            max_channels: 最大通道数
            d: 配置字典
            
        Returns:
            Tuple[int, List[Any]]: (输出通道数, 处理后的参数列表)
        """
        if not self.available:
            raise RuntimeError("多模态功能不可用，请检查模块导入")
        
        handler = self.module_handlers.get(module_class)
        if handler:
            return handler(f, args, ch, width, max_channels, d)
        else:
            raise ValueError(f"未知的多模态模块: {module_class}")
    
    def _handle_dea_family(
        self, 
        f: List[int], 
        args: List[Any], 
        ch: List[int],
        width: float,
        max_channels: float,
        d: Dict[str, Any]
    ) -> Tuple[int, List[Any]]:
        """
        处理DEA族模块 (DEA, DECA, DEPA)
        
        DEA族模块的特点：
        - 需要恰好2个输入源（RGB和X模态）
        - 要求两个输入通道数相同
        - 输出通道数与输入保持一致
        - 支持channel, reduction, kernel_size参数
        """
        # 验证输入源数量
        if len(f) != 2:
            raise ValueError(
                f"DEA族模块需要恰好2个输入源（RGB和X模态），实际输入: {len(f)}\n"
                f"请在YAML配置中使用: [[-1, -2], 1, 'DEA', [reduction, kernel_size]]"
            )
        
        # 获取输入通道数
        c1_rgb, c1_x = ch[f[0]], ch[f[1]]
        
        # 验证通道数一致性
        if c1_rgb != c1_x:
            raise ValueError(
                f"DEA族模块要求RGB和X模态通道数相同\n"
                f"实际: 第{f[0]}层={c1_rgb}通道, 第{f[1]}层={c1_x}通道\n"
                f"请检查网络结构或在融合前添加通道调整层"
            )
        
        # 设置输出通道数（与输入保持一致）
        c2 = c1_rgb
        
        # 处理参数
        processed_args = self._process_dea_args(args, c1_rgb, width, max_channels)
        
        return c2, processed_args
    
    def _handle_bifocus_family(
        self, 
        f: List[int], 
        args: List[Any], 
        ch: List[int],
        width: float,
        max_channels: float,
        d: Dict[str, Any]
    ) -> Tuple[int, List[Any]]:
        """
        处理BiFocus模块
        
        BiFocus模块特点：
        - 单输入源（标准YOLO层）
        - 输入输出通道数可以不同
        - 支持c1, c2参数
        """
        # 获取输入通道数
        c1 = ch[f]
        
        # 处理输出通道数
        if len(args) >= 1:
            c2 = args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
        else:
            c2 = c1  # 默认输出通道数与输入相同
        
        # 处理参数：[c2] -> [c1, c2]
        processed_args = [c1, c2] + args[1:]
        
        return c2, processed_args
    
    def _handle_c2f_bifocus_family(
        self, 
        f: List[int], 
        args: List[Any], 
        ch: List[int],
        width: float,
        max_channels: float,
        d: Dict[str, Any]
    ) -> Tuple[int, List[Any]]:
        """
        处理C2f_BiFocus模块
        
        C2f_BiFocus模块特点：
        - 单输入源（标准YOLO层）
        - 类似C2f模块的参数处理
        - 支持c1, c2, n, shortcut, g, e参数
        """
        # 获取输入通道数
        c1 = ch[f]
        
        # 处理输出通道数
        if len(args) >= 1:
            c2 = args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
        else:
            c2 = c1  # 默认输出通道数与输入相同
        
        # 处理参数：[c2, n, shortcut, g, e] -> [c1, c2, n, shortcut, g, e]
        processed_args = [c1, c2] + args[1:]
        
        return c2, processed_args
    
    def _handle_edge_prompt_family(
        self, 
        f: List[int], 
        args: List[Any], 
        ch: List[int],
        width: float,
        max_channels: float,
        d: Dict[str, Any]
    ) -> Tuple[int, List[Any]]:
        """
        处理EdgePrompt族模块 (EdgePrompt)
        
        EdgePrompt族模块的特点：
        - 需要恰好2个输入源（RGB和边缘特征）
        - RGB为主导，边缘为提示
        - 输出通道数与RGB输入保持一致
        - 支持rgb_channels, edge_channels, reduction, alpha参数
        """
        # 验证输入源数量
        if len(f) != 2:
            raise ValueError(
                f"EdgePrompt族模块需要恰好2个输入源（RGB和边缘特征），实际输入: {len(f)}\n"
                f"请在YAML配置中使用: [[-1, -2], 1, 'EdgePrompt', [edge_channels, reduction, alpha]]"
            )
        
        # 获取输入通道数 - RGB为主导
        c1_rgb, c1_edge = ch[f[0]], ch[f[1]]
        
        # 输出通道数与RGB保持一致（RGB为主导）
        c2 = c1_rgb
        
        # 处理参数
        processed_args = self._process_edge_prompt_args(args, c1_rgb, c1_edge, width, max_channels)
        
        return c2, processed_args
    
    def _process_edge_prompt_args(
        self, 
        args: List[Any], 
        rgb_channels: int, 
        edge_channels: int,
        width: float, 
        max_channels: float
    ) -> List[Any]:
        """
        处理EdgePrompt族模块的参数
        
        参数格式:
        - [] 或 [rgb_channels]: 使用默认参数
        - [edge_channels]: 指定边缘通道数
        - [edge_channels, reduction]: 指定边缘通道数和降维比例
        - [edge_channels, reduction, alpha]: 完整参数
        """
        if len(args) == 0:
            # 使用默认参数: rgb_channels=auto, edge_channels=auto, reduction=4, alpha=0.5
            return [rgb_channels]
        
        elif len(args) == 1:
            # 情况1: 指定edge_channels
            # 情况2: 明确指定rgb_channels（用于验证）
            if isinstance(args[0], int):
                if args[0] == rgb_channels:
                    # 明确指定rgb_channels，使用默认edge_channels
                    return [rgb_channels]
                else:
                    # 指定edge_channels
                    edge_ch = make_divisible(min(args[0], max_channels) * width, 8)
                    return [rgb_channels, edge_ch]
            else:
                return [rgb_channels, args[0]]
        
        elif len(args) == 2:
            # 指定了edge_channels和reduction
            edge_ch = make_divisible(min(args[0], max_channels) * width, 8) if isinstance(args[0], int) else args[0]
            return [rgb_channels, edge_ch, args[1]]
        
        elif len(args) == 3:
            # 完整参数: edge_channels, reduction, alpha
            edge_ch = make_divisible(min(args[0], max_channels) * width, 8) if isinstance(args[0], int) else args[0]
            return [rgb_channels, edge_ch, args[1], args[2]]
        
        else:
            # 超出预期的参数数量，保持原样
            return [rgb_channels] + args
    
    def _process_dea_args(
        self, 
        args: List[Any], 
        channels: int, 
        width: float, 
        max_channels: float
    ) -> List[Any]:
        """
        处理DEA族模块的参数
        
        参数格式:
        - [] 或 [channels]: 使用默认参数
        - [reduction]: 指定reduction比例
        - [reduction, kernel_size]: 指定reduction和kernel_size
        - [channels, reduction, kernel_size]: 完整参数
        """
        if len(args) == 0:
            # 使用默认参数: channel=auto, reduction=16, kernel_size=None
            return [channels]
        
        elif len(args) == 1:
            # 情况1: 只指定了reduction
            # 情况2: 明确指定了channels（用于验证）
            if isinstance(args[0], int):
                if args[0] == channels:
                    # 明确指定channels，使用默认reduction
                    return [channels]
                elif 1 <= args[0] <= 64:
                    # 指定reduction
                    return [channels, args[0]]
                else:
                    # 可能是错误的channels值
                    return [channels, args[0]]
            else:
                return [channels, args[0]]
        
        elif len(args) == 2:
            # 指定了reduction和kernel_size
            return [channels, args[0], args[1]]
        
        else:
            # 完整参数，确保channels正确
            processed_args = [channels] + args[1:]
            return processed_args


# 全局多模态解析器实例
_mm_parser = None


def get_multimodal_parser() -> MultiModalParser:
    """
    获取全局多模态解析器实例（单例模式）
    
    Returns:
        MultiModalParser: 多模态解析器实例
    """
    global _mm_parser
    if _mm_parser is None:
        _mm_parser = MultiModalParser()
    return _mm_parser


def is_multimodal_module(module_name: str) -> bool:
    """
    检查模块是否为多模态模块（便捷接口）
    
    Args:
        module_name: 模块名称
        
    Returns:
        bool: 是否为多模态模块
    """
    parser = get_multimodal_parser()
    return parser.is_multimodal_module(module_name)


def process_multimodal_layer(
    module_class: type,
    f: List[int],
    args: List[Any],
    ch: List[int],
    width: float,
    max_channels: float,
    d: Dict[str, Any]
) -> Tuple[int, List[Any]]:
    """
    处理多模态层（便捷接口）
    
    这是tasks.py中调用的主要接口函数
    
    Args:
        module_class: 模块类
        f: 输入层索引列表  
        args: 原始参数列表
        ch: 通道数列表
        width: 宽度倍数
        max_channels: 最大通道数
        d: 配置字典
        
    Returns:
        Tuple[int, List[Any]]: (输出通道数, 处理后的参数列表)
    """
    parser = get_multimodal_parser()
    return parser.process_multimodal_module(
        module_class, f, args, ch, width, max_channels, d
    )


def get_multimodal_modules() -> Dict[str, type]:
    """
    获取所有已注册的多模态模块（便捷接口）
    
    Returns:
        Dict[str, type]: 模块名称到模块类的映射
    """
    parser = get_multimodal_parser()
    return parser.multimodal_modules.copy() if parser.available else {}


# 为了向后兼容，也可以直接导出具体模块
__all__ = [
    'MultiModalParser',
    'get_multimodal_parser', 
    'is_multimodal_module',
    'process_multimodal_layer',
    'get_multimodal_modules'
] 