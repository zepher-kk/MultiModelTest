# Ultralytics Multimodal Router - Universal RGB+X Data Routing
# Supports YOLO and RTDETR with zero-copy tensor routing
# Version: v1.0

import torch
import torch.nn as nn
from ultralytics.utils import LOGGER


class MultiModalRouter:
    """
    Universal RGB+X Multimodal Data Router
    
    Supports:
    - RGB: 3-channel visible light images
    - X: 3-channel unified other modality (depth/thermal/lidar/etc.)
    - Dual: 6-channel RGB+X concatenated input
    
    Features:
    - Zero-copy tensor view routing
    - Configuration-driven data flow
    - Support for both YOLO and RTDETR architectures
    """
    
    def __init__(self, config_dict=None, verbose=True):
        """Initialize multimodal router with configuration"""
        # Get X modality channel count from dataset config
        x_channels = 3  # default value
        self.x_modality_type = 'unknown'
        if config_dict and 'dataset_config' in config_dict:
            dataset_config = config_dict['dataset_config']
            x_channels = dataset_config.get('Xch', 3)
            self.x_modality_type = dataset_config.get('x_modality', 'unknown')
        
        self.INPUT_SOURCES = {
            'RGB': 3,   # RGB modality input channels (fixed)
            'X': x_channels,     # X modality input channels (configurable via Xch)
            'Dual': 3 + x_channels   # RGB+X dual modality concatenated input channels
        }
        
        # Check if this is a multimodal configuration
        self.has_multimodal_config = self._detect_multimodal_config(config_dict)
        
        self.verbose = verbose
        self.original_spatial_size = None  # Will be set dynamically from input tensor
        self.original_inputs = {}  # Cache original multimodal inputs for spatial reset
        
        if self.verbose:
            LOGGER.info(f"🚀 MultiModal: RGB+X二元模态路由系统已启用")
            LOGGER.info(f"📋 MultiModal: RGB模态=可见光图像(3ch), X模态={self.x_modality_type}({x_channels}ch)")
            LOGGER.info(f"🎯 MultiModal: 支持路由='RGB'(3ch), 'X'({x_channels}ch), 'Dual'({3 + x_channels}ch)")
            LOGGER.info(f"🔍 MultiModal: 多模态配置检测={self.has_multimodal_config}")
    
    def parse_layer_config(self, layer_config, layer_index, ch, verbose=True):
        """
        Parse layer configuration with optional 5th field for multimodal routing
        
        Args:
            layer_config: Layer configuration [from, repeats, module, args, input_source?]
            layer_index: Current layer index
            ch: Channel information
            verbose: Whether to print verbose information
            
        Returns:
            tuple: (input_channels, mm_input_source, mm_attributes)
        """
        # Parse standard 4 fields and optional 5th field (MM input source identifier)
        if len(layer_config) >= 5:
            f, n, m, args, mm_input_source = layer_config[:5]
        else:
            f, n, m, args = layer_config[:4]
            mm_input_source = None
            
        mm_attributes = {}
        
        # Check 5th field: MM input source routing processing
        if mm_input_source and mm_input_source in self.INPUT_SOURCES:
            # RGB+X routing identifier detected, redirect input channel count
            c1 = self.INPUT_SOURCES[mm_input_source]
            
            # Set MM attributes for the module
            mm_attributes = {
                '_mm_input_source': mm_input_source,
                '_mm_layer_index': layer_index,
                '_mm_version': 'v1.0',
                '_mm_x_modality': self.x_modality_type
            }
            
            # Special handling: if X modality and from=-1, mark as new input start
            if mm_input_source == 'X' and f == -1:
                mm_attributes['_mm_new_input_start'] = True
                # Add spatial reset marking for X modality new input start
                mm_attributes['_mm_spatial_reset'] = True
                # Note: Original size will be dynamically determined from actual input tensor
                if verbose:
                    LOGGER.info(f"📍 MultiModal Layer {layer_index}: X模态新输入起点 (from=-1被重定向)")
                    LOGGER.info(f"🔄 MultiModal Layer {layer_index}: 空间重置标记已设置 (尺寸将从输入动态获取)")
            
            if verbose:
                if mm_input_source == 'RGB':
                    LOGGER.info(f"🎯 MultiModal Layer {layer_index}: '{m.__name__ if hasattr(m, '__name__') else m}' ← RGB模态输入 ({c1}通道)")
                elif mm_input_source == 'X':
                    LOGGER.info(f"🎯 MultiModal Layer {layer_index}: '{m.__name__ if hasattr(m, '__name__') else m}' ← X模态({self.x_modality_type})输入 ({c1}通道)")
                else:  # Dual
                    LOGGER.info(f"🎯 MultiModal Layer {layer_index}: '{m.__name__ if hasattr(m, '__name__') else m}' ← RGB+X双模态输入 ({c1}通道)")
        else:
            # Standard format, existing logic remains completely unchanged
            # Handle both single index and list of indices
            if isinstance(f, list):
                if len(f) == 1:
                    f_idx = f[0]
                    c1 = ch[f_idx] if f_idx != -1 else ch[-1]
                else:
                    # Multiple inputs case, calculate total channels
                    c1 = sum(ch[i] if i != -1 else ch[-1] for i in f)
            else:
                c1 = ch[f] if f != -1 else ch[-1]
            
        return c1, mm_input_source, mm_attributes
    
    def setup_multimodal_routing(self, x, profile=False):
        """
        Setup multimodal input sources and routing system initialization
        
        Args:
            x: Input tensor
            profile: Whether to print profiling information
            
        Returns:
            tuple: (routing_enabled, input_sources_dict)
        """
        routing_enabled = False
        input_sources = None
        
        # Detect multimodal modes
        expected_dual_channels = self.INPUT_SOURCES['Dual']  # 3 + Xch
        x_channels = self.INPUT_SOURCES['X']  # Xch from config
        is_dual_channel_input = x.shape[1] == expected_dual_channels
        is_multimodal_config = self.has_multimodal_config
        
        if is_dual_channel_input:
            # Dual-channel input: Always use proper RGB+X splitting (works for both early and mid fusion)
            routing_enabled = True
            # Cache original input spatial size for spatial reset
            self.original_spatial_size = (x.shape[2], x.shape[3])  # (H, W)
            
            # Use tensor view for zero-copy data routing
            input_sources = {
                'RGB': x[:, :3, :, :],      # RGB channels [B,3,H,W] - tensor view, zero-copy
                'X': x[:, 3:3+x_channels, :, :],   # X modality channels [B,Xch,H,W] - tensor view, zero-copy  
                'Dual': x                   # Full dual-channel [B,3+Xch,H,W] - direct reference
            }
            
            # Cache original inputs for spatial reset
            self.cache_original_inputs(input_sources)
            if profile:
                fusion_type = "中期融合" if is_multimodal_config else "早期融合"
                LOGGER.info(f"🎯 MultiModal: RGB+X路由已启用 ({fusion_type}) - 输入形状: {x.shape}")
                LOGGER.info(f"📊 MultiModal: RGB:{input_sources['RGB'].shape}, "
                           f"X:{input_sources['X'].shape}, Dual:{input_sources['Dual'].shape}")
                LOGGER.info(f"📐 MultiModal: 原始空间尺寸缓存: {self.original_spatial_size}")
                           
        elif is_multimodal_config and x.shape[1] == 3:
            # Mid/late fusion: 3-channel input with MM config, simulate dual-modal for routing
            routing_enabled = True
            # Cache original input spatial size for spatial reset
            self.original_spatial_size = (x.shape[2], x.shape[3])  # (H, W)
            
            # For mid/late fusion during model initialization, simulate RGB+X using same tensor
            # This allows the routing system to work during model construction
            input_sources = {
                'RGB': x,                   # RGB channels [B,3,H,W] - full input
                'X': x.clone(),             # X modality channels [B,3,H,W] - cloned for simulation
                'Dual': x                   # Use 3-channel as "dual" during init
            }
            
            # Cache original inputs for spatial reset
            self.cache_original_inputs(input_sources)
            if profile:
                LOGGER.info(f"🎯 MultiModal: RGB+X路由已启用 (中期/晚期融合) - 输入形状: {x.shape}")
                LOGGER.info(f"📊 MultiModal: RGB:{input_sources['RGB'].shape}, "
                           f"X:{input_sources['X'].shape}, Dual:{input_sources['Dual'].shape}")
                LOGGER.info(f"📐 MultiModal: 原始空间尺寸缓存: {self.original_spatial_size}")
                LOGGER.info(f"⚠️  MultiModal: 模型初始化时使用模拟双模态数据")
        
        return routing_enabled, input_sources
    
    def route_layer_input(self, x, module, input_sources, profile=False):
        """
        Route input data for a specific layer based on its MM attributes
        
        Args:
            x: Current input tensor
            module: Current module
            input_sources: Multimodal input sources dictionary
            profile: Whether to print profiling information
            
        Returns:
            torch.Tensor or None: Routed input tensor, None if no routing needed
        """
        if not hasattr(module, '_mm_input_source'):
            return None
            
        # Validate input sources availability
        if not input_sources:
            if profile:
                LOGGER.warning(f"⚠️  MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} - 输入源不可用")
            return None
            
        mm_input_source = module._mm_input_source

        # ===== 修复关键逻辑：参考原YOLOMM的正确实现 =====
        # 先处理特殊情况（X模态新输入起点），再进行常规路由
        if hasattr(module, '_mm_new_input_start') and module._mm_new_input_start:
            # X modality new input start, directly use X modality data
            if 'X' not in input_sources:
                if profile:
                    LOGGER.warning(f"⚠️  MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} - "
                                  f"X模态新输入起点需要X输入源")
                return None
                
            routed_x = input_sources['X']

            # Validate X modality data has correct shape (should match Xch channels)
            expected_x_channels = self.INPUT_SOURCES['X']
            if routed_x.shape[1] != expected_x_channels:
                if profile:
                    LOGGER.error(f"❌ MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} - "
                                f"X模态新输入起点期望{expected_x_channels}通道，但接收到{routed_x.shape[1]}通道")
                    LOGGER.error(f"📊 MultiModal: 当前输入源状态:")
                    for k, v in input_sources.items():
                        LOGGER.error(f"   {k}: {v.shape}")
                return None

            if profile:
                x_modality = getattr(module, '_mm_x_modality', 'unknown')
                LOGGER.info(f"🚀 MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} - "
                           f"X模态({x_modality})新输入起点")
                LOGGER.info(f"📍 MultiModal: 输入切换 {x.shape} → {routed_x.shape}")
        else:
            # Normal modality routing - validate and use requested modality
            if mm_input_source not in input_sources:
                if profile:
                    LOGGER.warning(f"⚠️  MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} - "
                                  f"请求的模态 '{mm_input_source}' 不存在于输入源中")
                return None
                
            routed_x = input_sources[mm_input_source]

            if profile:
                LOGGER.info(f"🚀 MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} "
                           f"路由到 '{mm_input_source}' - 输入形状: {x.shape} → {routed_x.shape}")
        
        # Final validation: ensure routed tensor is valid
        if routed_x is None:
            if profile:
                LOGGER.warning(f"⚠️  MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} - "
                              f"路由结果为None")
            return None
            
        return routed_x
    
    def set_module_attributes(self, module, mm_attributes):
        """Set multimodal attributes on a module"""
        for attr_name, attr_value in mm_attributes.items():
            setattr(module, attr_name, attr_value)
            
    def get_original_spatial_size(self):
        """Get the original input spatial size for spatial reset"""
        return self.original_spatial_size
    
    def cache_original_inputs(self, input_sources):
        """
        Cache original multimodal inputs for spatial reset operations
        
        Args:
            input_sources (dict): Multimodal input sources to cache
        """
        # Cache original inputs using references (zero-copy), especially X modality for spatial reset
        self.original_inputs = {
            'RGB': input_sources['RGB'] if 'RGB' in input_sources else None,
            'X': input_sources['X'] if 'X' in input_sources else None,  # Cache X modality reference
            'Dual': input_sources['Dual'] if 'Dual' in input_sources else None
        }
        
    def get_original_x_input(self, target_size=None):
        """
        Get original X modality input with specified target size
        
        Args:
            target_size (tuple, optional): Target spatial size (H, W). If None, returns original size.
            
        Returns:
            torch.Tensor or None: Original X modality tensor
        """
        if 'X' not in self.original_inputs or self.original_inputs['X'] is None:
            return None
            
        x_input = self.original_inputs['X']
        
        # If target_size is specified and different from current size, could add resize logic here
        # For now, we assume the original input already has the correct target size
        if target_size and target_size != x_input.shape[2:4]:
            # Target size validation for future extension
            pass
            
        return x_input
        
    def reset_spatial_input(self, x, module, mm_input_sources, profile=False):
        """
        Reset X modality input to original spatial size for spatial reset layers.
        
        Args:
            x (torch.Tensor): Current input tensor
            module (nn.Module): Module with spatial reset requirement
            mm_input_sources (dict): Multimodal input sources
            profile (bool): Whether to print profiling information
            
        Returns:
            torch.Tensor: Reset input tensor with original spatial size
        """
        if not hasattr(module, '_mm_new_input_start') or not module._mm_new_input_start:
            return x
            
        # Validate that we have the required input sources
        if not mm_input_sources or 'X' not in mm_input_sources:
            if profile:
                LOGGER.warning(f"🔄 MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} 空间重置失败 - 缺少X模态输入源")
            return x
            
        # Get original spatial size validation
        if self.original_spatial_size is None:
            if profile:
                LOGGER.warning(f"🔄 MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} 空间重置失败 - 无法获取原始尺寸")
            return x
            
        # Use X modality data with original spatial size
        reset_x = mm_input_sources['X']  # This already has original spatial size
        
        if profile:
            LOGGER.info(f"🔄 MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} 空间重置完成")
            LOGGER.info(f"📐 MultiModal: 尺寸重置 {x.shape} → {reset_x.shape}")
            
        return reset_x

    def update_dataset_config(self, dataset_config):
        """
        Update dataset configuration, particularly Xch value.
        
        Args:
            dataset_config (dict): Dataset configuration containing Xch
        """
        if dataset_config and 'Xch' in dataset_config:
            x_channels = dataset_config['Xch']
            self.INPUT_SOURCES['X'] = x_channels
            self.INPUT_SOURCES['Dual'] = 3 + x_channels
            
            if self.verbose:
                LOGGER.info(f"🔄 MultiModal: 更新X模态通道数为 {x_channels}")
                LOGGER.info(f"📋 MultiModal: 更新后路由配置: RGB(3ch), X({x_channels}ch), Dual({3 + x_channels}ch)")
    
    def _detect_multimodal_config(self, config_dict):
        """
        Detect if the configuration contains multimodal layers

        Args:
            config_dict (dict, optional): Model configuration dictionary

        Returns:
            bool: True if multimodal configuration detected, False otherwise
        """
        if not config_dict:
            return False

        # Check backbone and head layers for 5th field (MM input source)
        all_layers = config_dict.get('backbone', []) + config_dict.get('head', [])

        for layer_config in all_layers:
            if len(layer_config) >= 5:
                mm_input_source = layer_config[4]
                if mm_input_source in self.INPUT_SOURCES:
                    return True

        return False
