# Ultralytics Multimodal Config Parser
# Universal YAML configuration parsing for RGB+X architectures  
# Version: v1.0

from ultralytics.utils import LOGGER


class MultiModalConfigParser:
    """
    Universal Multimodal Configuration Parser
    
    Handles YAML configuration parsing for both YOLO and RTDETR
    with RGB+X multimodal extensions
    """
    
    def __init__(self):
        self.supported_input_sources = ['RGB', 'X', 'Dual']
    
    def validate_config_format(self, config):
        """Validate multimodal configuration format correctness"""
        
        rgb_layers = []
        x_layers = []
        dual_layers = []
        
        for section in ['backbone', 'head']:
            for i, layer_config in enumerate(config.get(section, [])):
                if len(layer_config) == 5:  # Has 5th field
                    input_source = layer_config[4]
                    if input_source == 'RGB':
                        rgb_layers.append(i)
                    elif input_source == 'X':
                        x_layers.append(i)
                    elif input_source == 'Dual':
                        dual_layers.append(i)
        
        LOGGER.info(f"✅ MultiModal: 配置验证完成")
        LOGGER.info(f"📊 MultiModal: RGB路由层={len(rgb_layers)}, X路由层={len(x_layers)}, Dual路由层={len(dual_layers)}")
        
        return {
            'rgb_layers': rgb_layers,
            'x_layers': x_layers, 
            'dual_layers': dual_layers,
            'total_routing_layers': len(rgb_layers) + len(x_layers) + len(dual_layers)
        }
    
    def extract_multimodal_info(self, config):
        """Extract multimodal information from configuration"""
        
        # Get X modality type from dataset config
        x_modality_type = config.get('dataset_config', {}).get('x_modality', 'unknown')
        
        # Count multimodal layers
        mm_layer_count = 0
        for section in ['backbone', 'head']:
            for layer_config in config.get(section, []):
                if len(layer_config) >= 5 and layer_config[4] in self.supported_input_sources:
                    mm_layer_count += 1
        
        return {
            'x_modality_type': x_modality_type,
            'mm_layer_count': mm_layer_count,
            'supports_multimodal': mm_layer_count > 0
        } 
