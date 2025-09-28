"""
Multi-Modal Modules for YOLOMM

这个目录中主要是包含多模态项目特有的一些模块，主要是负责跨模态融合，定制模态处理等方面。.

Author: YOLOMM Team
Date: 2024
"""

import json
import logging
import importlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Type
from dataclasses import dataclass

from .dea import DEA, DECA, DEPA
from .edge_prompt import EdgePrompt, DirectionAwareCore, GateProtection
from .bifocus import BiFocus, FocusH, FocusV, DepthWiseConv, C2f_BiFocus

__all__ = ['DEA', 'DECA', 'DEPA', 'EdgePrompt', 'DirectionAwareCore', 'GateProtection', 
           'BiFocus', 'FocusH', 'FocusV', 'DepthWiseConv', 'C2f_BiFocus',
           'MMModuleRegistry', 'register_mm_modules']


@dataclass
class ModuleInfo:
    """模块信息数据类"""
    name: str
    class_name: str
    import_path: str
    module_type: str
    description: str
    parameters: Dict[str, Any]
    input_format: str
    output_format: str
    compatibility: Dict[str, Any]
    performance: Dict[str, Any]


class MMModuleRegistry:
    """
    多模态模块配置化注册管理器
    
    负责从配置文件加载模块信息，验证模块可用性，并提供注册接口
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化注册管理器
        
        Args:
            config_path: 配置文件路径，默认使用mm_templates/mm_modules_config.json
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or self._get_default_config_path()
        self.config: Dict[str, Any] = {}
        self.registered_modules: Dict[str, Type] = {}
        self.module_info: Dict[str, ModuleInfo] = {}
        self.validation_enabled = True
        
        # 加载配置
        self._load_config()
        
    def _get_default_config_path(self) -> str:
        """获取默认配置文件路径"""
        current_dir = Path(__file__).parent
        config_path = current_dir.parent / "mm_templates" / "mm_modules_config.json"
        return str(config_path)
    
    def _load_config(self) -> bool:
        """
        加载配置文件
        
        Returns:
            bool: 加载是否成功
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            # 解析模块信息
            self._parse_module_info()
            
            # 设置验证级别
            strategy = self.config.get('registration_strategy', {})
            self.validation_enabled = strategy.get('validation_level') == 'strict'
            
            self.logger.info(f"✅ 成功加载配置文件: {self.config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 加载配置文件失败: {e}")
            return False
    
    def _parse_module_info(self):
        """解析配置文件中的模块信息"""
        modules_config = self.config.get('modules', {})
        
        for module_name, module_config in modules_config.items():
            try:
                module_info = ModuleInfo(
                    name=module_name,
                    class_name=module_config['class_name'],
                    import_path=module_config['import_path'],
                    module_type=module_config.get('module_type', 'unknown'),
                    description=module_config.get('description', ''),
                    parameters=module_config.get('parameters', {}),
                    input_format=module_config.get('input_format', ''),
                    output_format=module_config.get('output_format', ''),
                    compatibility=module_config.get('compatibility', {}),
                    performance=module_config.get('performance', {})
                )
                self.module_info[module_name] = module_info
                
            except KeyError as e:
                self.logger.error(f"❌ 模块 {module_name} 配置不完整，缺少字段: {e}")
    
    def validate_module_parameters(self, module_name: str, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证模块参数
        
        Args:
            module_name: 模块名称
            params: 参数字典
            
        Returns:
            (is_valid, error_messages): 验证结果和错误信息
        """
        if not self.validation_enabled:
            return True, []
        
        if module_name not in self.module_info:
            return False, [f"未知模块: {module_name}"]
        
        module_info = self.module_info[module_name]
        param_configs = module_info.parameters
        errors = []
        
        # 检查必需参数
        for param_name, param_config in param_configs.items():
            if param_config.get('required', False) and param_name not in params:
                errors.append(f"缺少必需参数: {param_name}")
        
        # 检查参数类型和范围
        for param_name, param_value in params.items():
            if param_name in param_configs:
                param_config = param_configs[param_name]
                
                # 类型检查
                expected_type = param_config.get('type')
                if expected_type and not self._check_parameter_type(param_value, expected_type):
                    errors.append(f"参数 {param_name} 类型错误，期望: {expected_type}")
                
                # 范围检查
                validation = param_config.get('validation', {})
                if validation and not self._check_parameter_range(param_value, validation):
                    errors.append(f"参数 {param_name} 超出有效范围")
        
        return len(errors) == 0, errors
    
    def _check_parameter_type(self, value: Any, expected_type: str) -> bool:
        """检查参数类型"""
        type_mapping = {
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'list': list,
            'dict': dict
        }
        
        if expected_type in type_mapping:
            return isinstance(value, type_mapping[expected_type])
        return True  # 未知类型跳过检查
    
    def _check_parameter_range(self, value: Any, validation: Dict[str, Any]) -> bool:
        """检查参数范围"""
        if isinstance(value, (int, float)):
            if 'min' in validation and value < validation['min']:
                return False
            if 'max' in validation and value > validation['max']:
                return False
        
        if isinstance(value, list):
            if 'min_length' in validation and len(value) < validation['min_length']:
                return False
            if 'max_length' in validation and len(value) > validation['max_length']:
                return False
        
        return True
    
    def import_module(self, module_name: str) -> Optional[Type]:
        """
        动态导入模块
        
        Args:
            module_name: 模块名称
            
        Returns:
            模块类或None
        """
        if module_name in self.registered_modules:
            return self.registered_modules[module_name]
        
        if module_name not in self.module_info:
            self.logger.error(f"❌ 未知模块: {module_name}")
            return None
        
        module_info = self.module_info[module_name]
        
        try:
            # 动态导入
            module = importlib.import_module(module_info.import_path)
            module_class = getattr(module, module_info.class_name)
            
            # 缓存模块
            self.registered_modules[module_name] = module_class
            
            self.logger.info(f"✅ 成功导入模块: {module_name}")
            return module_class
            
        except Exception as e:
            error_handling = self.config.get('error_handling', {})
            import_error_strategy = error_handling.get('import_errors', 'log_and_skip')
            
            if import_error_strategy == 'log_and_skip':
                self.logger.warning(f"⚠️ 导入模块 {module_name} 失败: {e}")
                return None
            else:
                raise ImportError(f"导入模块 {module_name} 失败: {e}")
    
    def get_available_modules(self) -> List[str]:
        """获取可用模块列表"""
        return list(self.module_info.keys())
    
    def get_module_info(self, module_name: str) -> Optional[ModuleInfo]:
        """获取模块信息"""
        return self.module_info.get(module_name)
    
    def register_all_modules(self) -> Dict[str, Type]:
        """
        注册所有配置的模块
        
        Returns:
            成功注册的模块字典
        """
        registered = {}
        
        for module_name in self.get_available_modules():
            module_class = self.import_module(module_name)
            if module_class:
                registered[module_name] = module_class
        
        self.logger.info(f"📦 成功注册 {len(registered)} 个模块")
        return registered
    
    def generate_globals_dict(self) -> Dict[str, Type]:
        """
        生成用于parse_model的globals字典
        
        Returns:
            包含所有注册模块的字典
        """
        return self.register_all_modules()


# 全局注册管理器实例
_global_registry = None


def get_registry() -> MMModuleRegistry:
    """获取全局注册管理器实例"""
    global _global_registry
    if _global_registry is None:
        _global_registry = MMModuleRegistry()
    return _global_registry


def register_mm_modules() -> Dict[str, Type]:
    """
    注册所有多模态模块的便捷函数
    
    Returns:
        注册的模块字典，可直接用于globals().update()
    """
    registry = get_registry()
    return registry.generate_globals_dict()


def validate_module_config(module_name: str, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    验证模块配置的便捷函数
    
    Args:
        module_name: 模块名称
        params: 参数字典
        
    Returns:
        (is_valid, error_messages): 验证结果和错误信息
    """
    registry = get_registry()
    return registry.validate_module_parameters(module_name, params)


def get_module_documentation(module_name: str) -> Optional[str]:
    """
    获取模块文档的便捷函数
    
    Args:
        module_name: 模块名称
        
    Returns:
        模块描述信息
    """
    registry = get_registry()
    module_info = registry.get_module_info(module_name)
    return module_info.description if module_info else None 