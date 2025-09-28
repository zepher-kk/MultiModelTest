# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
多模态YOLO模型架构组件

本模块包含完整的多模态目标检测系统，支持RGB+X模态的训练、验证、推理和数据生成。

主要组件:
- MultiModalDetectionTrainer: 多模态训练器
- MultiModalDetectionValidator: 多模态验证器  
- MultiModalDetectionPredictor: 多模态预测器
- ModalityFiller: 模态填充器
- SelfModalGenerator: 自体模态生成器

使用示例:
    >>> from ultralytics.models.yolo.multimodal import MultiModalDetectionTrainer
    >>> trainer = MultiModalDetectionTrainer(model='yolo11n-mm.yaml', data='multimodal_data.yaml')
    >>> trainer.train()
    
    >>> from ultralytics.models.yolo.multimodal import MultiModalDetectionPredictor
    >>> predictor = MultiModalDetectionPredictor()
    >>> results = predictor.predict(['rgb_image.jpg', 'x_image.jpg'])
    
    >>> from ultralytics.models.yolo.multimodal import generate_modality_filling
    >>> filled_tensor = generate_modality_filling(rgb_tensor, 'rgb', 'depth')
"""

# 导入多模态训练器
from .train import MultiModalDetectionTrainer

# 导入多模态验证器
from .val import MultiModalDetectionValidator

# 导入多模态COCO验证器
from .cocoval import MultiModalCOCOValidator

# 导入多模态预测器
from .predict import MultiModalDetectionPredictor

# 导入模态填充功能
from .modal_filling import (
    ModalityFiller,
    generate_modality_filling,
    generate_self_modality,
    default_modality_filler,
    default_self_modal_generator,
)

# 导入自体模态生成器
from .self_modal_generator import (
    SelfModalGenerator,
    generate_self_modality as generate_self_modality_direct,
    default_self_modal_generator as default_self_modal_generator_direct,
)

# 公开的API
__all__ = [
    # 核心多模态组件
    'MultiModalDetectionTrainer',
    'MultiModalDetectionValidator',
    'MultiModalCOCOValidator',
    'MultiModalDetectionPredictor',
    
    # 模态填充组件
    'ModalityFiller',
    'generate_modality_filling',
    'default_modality_filler',
    
    # 自体模态生成组件
    'SelfModalGenerator',
    'generate_self_modality',
    'default_self_modal_generator',
]

# 版本信息
__version__ = '1.0.0'
__author__ = 'Ultralytics MultiModal Team'
__description__ = 'Ultralytics YOLO MultiModal Detection Components'

# 模块级别的配置
SUPPORTED_MODALITIES = ['rgb', 'depth', 'thermal', 'ir', 'nir', 'lidar', 'sar']
SUPPORTED_FUSION_TYPES = ['early', 'mid', 'late']
DEFAULT_CHANNELS = 6  # RGB(3) + X(3)

def get_multimodal_info():
    """
    获取多模态模块信息
    
    Returns:
        dict: 包含模块版本、支持的模态类型等信息
    """
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'supported_modalities': SUPPORTED_MODALITIES,
        'supported_fusion_types': SUPPORTED_FUSION_TYPES,
        'default_channels': DEFAULT_CHANNELS,
        'components': {
            'trainer': 'MultiModalDetectionTrainer',
            'validator': 'MultiModalDetectionValidator',
            'coco_validator': 'MultiModalCOCOValidator',
            'predictor': 'MultiModalDetectionPredictor',
            'modality_filler': 'ModalityFiller',
            'self_modal_generator': 'SelfModalGenerator',
        }
    }

def validate_multimodal_setup():
    """
    验证多模态组件设置
    
    Returns:
        bool: 如果所有组件都正确导入则返回True
    """
    try:
        # 验证核心组件
        assert MultiModalDetectionTrainer is not None
        assert MultiModalDetectionValidator is not None
        assert MultiModalCOCOValidator is not None
        assert MultiModalDetectionPredictor is not None
        
        # 验证模态填充组件
        assert ModalityFiller is not None
        assert generate_modality_filling is not None
        assert default_modality_filler is not None
        
        # 验证自体模态生成组件
        assert SelfModalGenerator is not None
        assert generate_self_modality is not None
        assert default_self_modal_generator is not None
        
        return True
    except (AssertionError, ImportError) as e:
        print(f"多模态组件验证失败: {e}")
        return False

# 模块初始化时进行验证
if __name__ != '__main__':
    # 只在被导入时进行验证，避免在直接运行时出错
    _validation_result = validate_multimodal_setup()
    if not _validation_result:
        import warnings
        warnings.warn("多模态组件验证失败，某些功能可能不可用", UserWarning)
