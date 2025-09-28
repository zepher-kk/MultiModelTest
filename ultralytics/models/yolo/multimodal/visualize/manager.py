"""
Visualization Manager for YOLOMM multimodal object detection.

This module provides:
- Data models for visualization results
- VisualizationManager as the main entry point for all visualization tasks
- Support for heatmap and feature map visualization
- Automatic output directory management
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import numpy as np
import cv2
import torch

from .utils import HookManager


class VisualizationResult:
    """Base class for all visualization results."""
    
    def __init__(self, 
                 vis_type: str,
                 data: Union[np.ndarray, Dict[str, np.ndarray], List[np.ndarray]],
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize visualization result.
        
        Args:
            vis_type: Type of visualization ('heatmap', 'feature_map', etc.)
            data: Visualization data (numpy arrays)
            metadata: Additional metadata (layer info, method type, etc.)
        """
        self.type = vis_type
        self.data = data
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        result = {
            'type': self.type,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }
        
        # Handle different data types
        if isinstance(self.data, dict):
            result['data_keys'] = list(self.data.keys())
        elif isinstance(self.data, list):
            result['data_count'] = len(self.data)
        else:
            result['data_shape'] = self.data.shape if hasattr(self.data, 'shape') else None
            
        return result
    
    def save(self, output_dir: Union[str, Path], prefix: str = "") -> List[str]:
        """
        Save visualization result to files.
        
        Args:
            output_dir: Directory to save results
            prefix: Prefix for filename
            
        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []
        
        timestamp = self.timestamp.strftime("%Y%m%d_%H%M%S")
        base_name = f"{prefix}_{self.type}_{timestamp}" if prefix else f"{self.type}_{timestamp}"
        
        # Save based on data type
        if isinstance(self.data, dict):
            # Save each item in dictionary
            for key, value in self.data.items():
                if isinstance(value, np.ndarray):
                    filename = f"{base_name}_{key}.png"
                    filepath = output_dir / filename
                    cv2.imwrite(str(filepath), value)
                    saved_files.append(str(filepath))
                    
        elif isinstance(self.data, list):
            # Save each item in list
            for idx, value in enumerate(self.data):
                if isinstance(value, np.ndarray):
                    filename = f"{base_name}_{idx:03d}.png"
                    filepath = output_dir / filename
                    cv2.imwrite(str(filepath), value)
                    saved_files.append(str(filepath))
                    
        elif isinstance(self.data, np.ndarray):
            # Save single array
            filename = f"{base_name}.png"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), self.data)
            saved_files.append(str(filepath))
            
        return saved_files


class HeatmapResult(VisualizationResult):
    """Result class for heatmap visualizations."""
    
    def __init__(self,
                 original_image: Union[np.ndarray, Dict[str, np.ndarray]],
                 heatmap: Union[np.ndarray, Dict[str, np.ndarray]],
                 overlay: Union[np.ndarray, Dict[str, np.ndarray]],
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize heatmap result.
        
        Args:
            original_image: Original input image(s)
            heatmap: Generated heatmap(s)
            overlay: Overlay of heatmap on original image(s)
            metadata: Additional metadata
        """
        # Store individual components
        self.original_image = original_image
        self.heatmap = heatmap
        self.overlay = overlay
        
        # Prepare data for base class
        if isinstance(original_image, dict):
            # Multi-modal case
            data = {}
            for modal in original_image.keys():
                if modal in heatmap and modal in overlay:
                    data[f"{modal}_original"] = original_image[modal]
                    data[f"{modal}_heatmap"] = heatmap[modal]
                    data[f"{modal}_overlay"] = overlay[modal]
        else:
            # Single modal case
            data = {
                'original': original_image,
                'heatmap': heatmap,
                'overlay': overlay
            }
            
        super().__init__(vis_type='heatmap', data=overlay, metadata=metadata)
        
    @property
    def rgb_heatmap(self) -> Optional[np.ndarray]:
        """Get RGB heatmap if available."""
        if isinstance(self.heatmap, dict):
            return self.heatmap.get('rgb')
        return self.heatmap if not isinstance(self.original_image, dict) else None
    
    @property
    def x_heatmap(self) -> Optional[np.ndarray]:
        """Get X modality heatmap if available."""
        if isinstance(self.heatmap, dict):
            return self.heatmap.get('x')
        return None


class FeatureMapResult(VisualizationResult):
    """Result class for feature map visualizations."""
    
    def __init__(self,
                 layer_idx: Union[int, List[int]],
                 feature_maps: List[np.ndarray],
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize feature map result.
        
        Args:
            layer_idx: Layer index or indices
            feature_maps: List of feature map visualizations
            metadata: Additional metadata
        """
        self.layer_idx = layer_idx
        self.feature_maps = feature_maps
        
        # Update metadata with layer info
        if metadata is None:
            metadata = {}
        metadata['layer_idx'] = layer_idx
        metadata['num_maps'] = len(feature_maps)
        
        super().__init__(vis_type='feature_map', data=feature_maps, metadata=metadata)
        
    def get_feature_map(self, idx: int) -> Optional[np.ndarray]:
        """Get specific feature map by index."""
        if 0 <= idx < len(self.feature_maps):
            return self.feature_maps[idx]
        return None


class VisualizationManager:
    """Main manager class for all visualization operations."""
    
    # Supported visualization methods
    SUPPORTED_METHODS = ['heatmap', 'feature_map']
    
    def __init__(self, model, project: str = "runs/visualize", name: str = "exp"):
        """
        Initialize VisualizationManager.
        
        Args:
            model: YOLOMM model instance (MUST store reference to avoid params N/A)
            project: Base project directory
            name: Experiment name
        """
        # CRITICAL: Store model reference to ensure params are accessible
        self.model = model
        
        # Setup output directory
        self.output_dir = self._setup_output_dir(project, name)
        
        # Initialize visualizer dictionary (lazy loading)
        self._visualizers = {}
        
        # Initialize cache dictionary for visualization results
        self.cache = {}
        
        # Hook manager for feature extraction
        self.hook_manager = HookManager(model)
    
    def __call__(self, 
                 source: Union[np.ndarray, Dict[str, np.ndarray]],
                 method: str = "heatmap",
                 **kwargs) -> VisualizationResult:
        """
        Make VisualizationManager callable for convenience.
        
        Args:
            source: Standardized input - either NumPy array or dict of NumPy arrays
            method: Visualization method
            **kwargs: Additional arguments passed to visualize
            
        Returns:
            Visualization result
            
        Examples:
            >>> manager = VisualizationManager(model)
            >>> result = manager(numpy_image, method='heatmap', alg='gradcam')
            >>> result = manager({'rgb': rgb_array, 'x': thermal_array}, method='feature_map', layers=[0, 5])
        """
        return self.visualize(source, method=method, **kwargs)
        
    def _setup_output_dir(self, project: str, name: str) -> Path:
        """
        Setup output directory with automatic numbering.
        
        Args:
            project: Base project directory
            name: Experiment name
            
        Returns:
            Path to output directory
        """
        project_path = Path(project)
        
        # Find next available experiment number
        i = 1
        while True:
            exp_name = name if i == 1 else f"{name}{i}"
            output_dir = project_path / exp_name
            if not output_dir.exists():
                break
            i += 1
            
        # Create directory structure
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Visualization output directory: {output_dir}")
        return output_dir
    
    def _load_images(self, source: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Ensure images are in the expected NumPy format.
        
        Args:
            source: Standardized input - either NumPy array or dict of NumPy arrays
            
        Returns:
            Input source (already in the correct format)
        """
        # Since input is already standardized, just return it
        # This method is kept for backward compatibility and potential future preprocessing
        return source
    
    def _validate_method(self, method: str) -> None:
        """
        Validate visualization method.
        
        Args:
            method: Method name to validate
            
        Raises:
            ValueError: If method is not supported
        """
        if method not in self.SUPPORTED_METHODS:
            # Provide more helpful error message with list of supported methods
            method_descriptions = {
                'heatmap': 'Heatmap visualization (supports multiple algorithms: gradcam, gradcam++, ablationcam)',
                'feature_map': 'Feature map extraction and visualization'
            }
            
            supported_list = "\n".join([
                f"  - {m}: {method_descriptions.get(m, 'Method description not available')}"
                for m in self.SUPPORTED_METHODS
            ])
            
            raise ValueError(
                f"Unsupported visualization method: '{method}'.\n"
                f"\nSupported methods are:\n{supported_list}\n\n"
                f"Please choose one of the above methods."
            )
    
    def _get_visualizer(self, method: str):
        """
        Get or create visualizer for specified method (lazy loading).
        
        Args:
            method: Visualization method name
            
        Returns:
            Visualizer instance
        """
        # Note: This method implements lazy loading and caching of visualizers.
        # Once a visualizer is created, it's cached in self._visualizers for reuse.
        if method not in self._visualizers:
            try:
                if method == 'heatmap':
                    from .heatmap import HeatmapVisualizer
                    self._visualizers[method] = HeatmapVisualizer(self.model)
                elif method == 'feature_map':
                    from .feature import FeatureMapVisualizer
                    self._visualizers[method] = FeatureMapVisualizer(self.model)
                else:
                    raise ValueError(f"Visualizer not implemented for method: {method}")
            except ImportError as e:
                # Enhanced error message for better user experience
                available_methods = ['heatmap', 'feature_map']  # Methods we know are implemented
                raise ImportError(
                    f"Failed to import visualizer for method '{method}'.\n"
                    f"Please check if the module exists and all dependencies are installed.\n"
                    f"Available methods: {', '.join(available_methods)}\n"
                    f"Original error: {e}"
                )
                
        return self._visualizers[method]
    
    def _validate_input(self, source: Any) -> None:
        """
        Validate standardized input source.
        
        Args:
            source: Input to validate - should be NumPy array or dict of NumPy arrays
            
        Raises:
            ValueError: If input is invalid
            TypeError: If input type is not supported
        """
        if source is None:
            raise ValueError("Input source cannot be None")
            
        if isinstance(source, dict):
            if len(source) == 0:
                raise ValueError("Input dictionary cannot be empty")
            # Check for valid modality keys
            valid_keys = {'rgb', 'x', 'thermal', 'depth'}
            if not any(k in valid_keys for k in source.keys()):
                raise ValueError(
                    f"Dictionary must contain at least one valid modality key: {valid_keys}. "
                    f"Found keys: {list(source.keys())}"
                )
            # Validate each array in the dictionary
            for key, array in source.items():
                if not isinstance(array, np.ndarray):
                    raise TypeError(f"Dictionary value for '{key}' must be a NumPy array, got {type(array)}")
                self._validate_numpy_array(array, f"Dict['{key}']")
                
        elif isinstance(source, np.ndarray):
            self._validate_numpy_array(source, "Input")
        else:
            raise TypeError(
                f"Unsupported input type: {type(source)}. "
                "Expected: numpy array or dict of numpy arrays"
            )
    
    def _validate_numpy_array(self, array: np.ndarray, name: str) -> None:
        """
        Validate a NumPy array for visualization.
        
        Args:
            array: Array to validate
            name: Name for error messages
            
        Raises:
            ValueError: If array is invalid
            TypeError: If array has wrong dtype
        """
        if array.size == 0:
            raise ValueError(f"{name} numpy array cannot be empty")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{name} numpy array must have numeric dtype, got {array.dtype}")
        if array.ndim < 2 or array.ndim > 4:
            raise ValueError(
                f"{name} numpy array must be 2D (HW), 3D (HWC/CHW), or 4D (NCHW), got shape {array.shape}. "
                f"Dimensions: {array.ndim}D"
            )
        # Check for plausible image dimensions
        if array.ndim >= 3:
            # Check channel count
            channel_dim = array.shape[2] if array.ndim == 3 else array.shape[1] if array.ndim == 4 else None
            if channel_dim is not None and channel_dim not in [1, 3, 4, 6]:
                raise ValueError(
                    f"{name} numpy array has unusual channel count: {channel_dim}. "
                    f"Expected 1 (grayscale), 3 (RGB), 4 (RGBA), or 6 (multi-modal)"
                )
        # Check for valid values
        if np.any(np.isnan(array)):
            raise ValueError(f"{name} numpy array contains NaN values")
        if np.any(np.isinf(array)):
            raise ValueError(f"{name} numpy array contains infinite values")
    
    def _generate_cache_key(self, source: Any, method: str, layers: Optional[List[str]], alg: Optional[str] = None, **kwargs) -> str:
        """
        Generate a unique cache key based on input parameters.
        
        Args:
            source: Input source
            method: Visualization method
            layers: Target layers
            alg: Algorithm for heatmap visualization
            **kwargs: Additional parameters
            
        Returns:
            MD5 hash string as cache key
        """
        # Create a string representation of all parameters
        key_parts = [method]
        
        # Add algorithm if specified
        if alg is not None:
            key_parts.append(f"alg:{alg}")
        
        # Handle different source types
        if isinstance(source, np.ndarray):
            # For arrays, use shape and a sample of values
            key_parts.append(f"array_{source.shape}_{source.dtype}")
            # Sample some values for uniqueness
            flat = source.flatten()
            indices = np.linspace(0, len(flat)-1, min(100, len(flat)), dtype=int)
            key_parts.append(str(flat[indices].tolist()))
        elif isinstance(source, dict):
            # For dicts, process each item
            for k, v in sorted(source.items()):
                key_parts.append(f"{k}:{self._generate_cache_key(v, '', None)}")
            
        # Add layers
        if layers is not None:
            key_parts.append(str(layers))
            
        # Add sorted kwargs
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")
            
        # Generate MD5 hash
        key_string = "_".join(str(part) for part in key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def visualize(self, 
                  source: Union[np.ndarray, Dict[str, np.ndarray]],
                  method: str = "heatmap",
                  layers: Optional[List[str]] = None,  # 现在接收转换后的层名称列表
                  save: bool = True,
                  alg: str = 'gradcam',
                  **kwargs) -> List[VisualizationResult]:
        """
        Main visualization entry point supporting multi-layer visualization.
        
        Args:
            source: Standardized input - either NumPy array or dict of NumPy arrays
            method: Visualization method ('heatmap', 'feature_map')
            layers: List of target layer names for visualization (e.g., ['model.6', 'model.8'])
            save: Whether to save results to disk
            alg: Algorithm for heatmap visualization ('gradcam', 'gradcam++', 'ablationcam')
            **kwargs: Additional arguments for specific methods
            
        Returns:
            List[VisualizationResult]: Always returns a list, even for single layer
            
        Raises:
            ValueError: For invalid inputs or methods
            RuntimeError: For visualization failures
            
        Examples:
            >>> # Single layer visualization
            >>> results = manager.visualize(numpy_image, method='heatmap', layers=['model.6'])
            
            >>> # Multi-layer visualization
            >>> results = manager.visualize({'rgb': rgb_array, 'x': thermal_array}, layers=['model.4', 'model.6'])
        """
        try:
            # Validate inputs
            self._validate_input(source)
            self._validate_method(method)
            
            # Ensure layers is provided
            if not layers:
                raise ValueError("layers parameter must be provided")
            
            # Process each layer and collect results
            all_results = []
            total_layers = len(layers)
            
            for idx, layer in enumerate(layers):
                # Show progress for multiple layers
                if total_layers > 1:
                    print(f"Processing layer {layer} ({idx+1}/{total_layers})...")
                
                # Generate cache key for this specific layer
                cache_key = self._generate_cache_key(source, method, [layer], alg, **kwargs)
                
                # Check cache
                if cache_key in self.cache:
                    print(f"Using cached result for {method} visualization of layer {layer}")
                    cached_result = self.cache[cache_key]
                    
                    # Save if requested (even for cached results)
                    if save and cached_result:
                        # Extract layer index from layer name for file naming
                        layer_idx = layer.split('.')[-1]  # 'model.6' -> '6'
                        saved_files = self._save_with_layer_info(cached_result, layer_idx)
                        print(f"Saved cached visualization results for layer {layer}: {len(saved_files)} files")
                    
                    all_results.append(cached_result)
                    continue
                
                # Get appropriate visualizer
                try:
                    visualizer = self._get_visualizer(method)
                except Exception as e:
                    # Fallback to placeholder if visualizer not implemented yet
                    print(f"Warning: {e}")
                    print(f"Using placeholder visualization for method: {method}")
                    result = self._placeholder_visualization(source, method, [layer], **kwargs)
                    all_results.append(result)
                    continue
                
                # Perform visualization for this layer
                try:
                    # Pass alg parameter for heatmap visualizers
                    if method == 'heatmap':
                        result = visualizer.visualize(source, layers=[layer], alg=alg, **kwargs)
                    else:
                        result = visualizer.visualize(source, layers=[layer], **kwargs)
                    
                    # Handle the result - it might be a list or single result
                    if isinstance(result, list) and len(result) == 1:
                        result = result[0]
                    
                    # Store layer info in metadata
                    if result and hasattr(result, 'metadata'):
                        result.metadata['layer'] = layer
                        layer_idx = layer.split('.')[-1]
                        result.metadata['layer_idx'] = int(layer_idx)
                    
                    # Store in cache
                    self.cache[cache_key] = result
                    
                    # Save results if requested
                    if save and result:
                        layer_idx = layer.split('.')[-1]
                        saved_files = self._save_with_layer_info(result, layer_idx)
                        print(f"Saved visualization results for layer {layer}: {len(saved_files)} files")
                    
                    all_results.append(result)
                
                except Exception as e:
                    raise RuntimeError(f"Visualization failed for layer {layer}: {e}")
            
            return all_results
        except Exception as e:
            # Log error and clean up resources
            print(f"Error in visualization: {e}")
            
            # Ensure resources are cleaned up
            try:
                if hasattr(self, 'hook_manager'):
                    self.hook_manager.remove_all_hooks()
            except:
                pass
                
            raise
    
    
    def _placeholder_visualization(self,
                                 images: Union[np.ndarray, Dict[str, np.ndarray]],
                                 method: str,
                                 layers: Optional[Union[int, List[int]]],
                                 **kwargs) -> VisualizationResult:
        """
        Create placeholder visualization when actual visualizer is not available.
        
        Args:
            images: Loaded images
            method: Requested method
            layers: Target layers
            **kwargs: Additional arguments
            
        Returns:
            Placeholder visualization result
        """
        print(f"Creating placeholder visualization for method: {method}")
        print(f"Target layers: {layers}")
        if kwargs:
            print(f"Additional args: {kwargs}")
        
        if isinstance(images, dict):
            # Multi-modal placeholder
            placeholder_heatmap = {}
            placeholder_overlay = {}
            
            for modal, img in images.items():
                # Create simple gradient as placeholder
                h, w = img.shape[:2]
                gradient = np.linspace(0, 255, h*w, dtype=np.uint8).reshape(h, w)
                
                # Create colored heatmap
                heatmap = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
                
                # Create overlay
                if len(img.shape) == 2:
                    img_color = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 1:
                    img_color = cv2.cvtColor(img[:, :, 0].astype(np.uint8), cv2.COLOR_GRAY2BGR)
                else:
                    img_color = img[:, :, :3].astype(np.uint8) if img.shape[2] > 3 else img.astype(np.uint8)
                    if img_color.shape[2] < 3:
                        img_color = cv2.cvtColor(img_color, cv2.COLOR_GRAY2BGR)
                
                # Ensure same dtype and shape for overlay
                img_color = img_color.astype(np.uint8)
                heatmap = heatmap.astype(np.uint8)
                    
                overlay = cv2.addWeighted(img_color, 0.7, heatmap, 0.3, 0)
                
                placeholder_heatmap[modal] = heatmap
                placeholder_overlay[modal] = overlay
                
            return HeatmapResult(
                original_image=images,
                heatmap=placeholder_heatmap,
                overlay=placeholder_overlay,
                metadata={
                    'method': method,
                    'layers': layers,
                    'note': 'This is a placeholder visualization'
                }
            )
        else:
            # Single modal placeholder
            h, w = images.shape[:2]
            gradient = np.linspace(0, 255, h*w, dtype=np.uint8).reshape(h, w)
            
            # Create colored heatmap
            heatmap = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
            
            # Create overlay
            if len(images.shape) == 2:
                img_color = cv2.cvtColor(images, cv2.COLOR_GRAY2BGR)
            else:
                img_color = images[:, :, :3] if images.shape[2] > 3 else images
            
            # Ensure same dtype for overlay
            if img_color.dtype != heatmap.dtype:
                img_color = img_color.astype(np.uint8)
                heatmap = heatmap.astype(np.uint8)
                
            overlay = cv2.addWeighted(img_color, 0.7, heatmap, 0.3, 0)
            
            return HeatmapResult(
                original_image=images,
                heatmap=heatmap,
                overlay=overlay,
                metadata={
                    'method': method,
                    'layers': layers,
                    'note': 'This is a placeholder visualization'
                }
            )
    
    def _save_with_layer_info(self, result: VisualizationResult, layer_idx: str) -> List[str]:
        """
        Save visualization result with layer information in filename.
        
        Args:
            result: Visualization result to save
            layer_idx: Layer index to include in filename
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Prepare the base name with layer info
        base_name = f"{result.type}_layer{layer_idx}"
        
        # Save based on data type
        if isinstance(result.data, dict):
            # Multi-modal case - save each modality
            for key, value in result.data.items():
                if isinstance(value, np.ndarray):
                    filename = f"{base_name}_{key}.png"
                    filepath = self.output_dir / filename
                    cv2.imwrite(str(filepath), value)
                    saved_files.append(str(filepath))
                    
        elif isinstance(result.data, list):
            # List of arrays
            for idx, value in enumerate(result.data):
                if isinstance(value, np.ndarray):
                    filename = f"{base_name}_{idx:03d}.png"
                    filepath = self.output_dir / filename
                    cv2.imwrite(str(filepath), value)
                    saved_files.append(str(filepath))
                    
        elif isinstance(result.data, np.ndarray):
            # Single array
            filename = f"{base_name}.png"
            filepath = self.output_dir / filename
            cv2.imwrite(str(filepath), result.data)
            saved_files.append(str(filepath))
            
        return saved_files
    
    def clear_cache(self):
        """Clear all cached visualizers and results to free memory."""
        self._visualizers.clear()
        self.cache.clear()
        print("Cleared all cached visualizers and results")
    
    def get_supported_methods(self) -> List[str]:
        """
        Get list of supported visualization methods.
        
        Returns:
            List of method names
        """
        return self.SUPPORTED_METHODS.copy()
    
    def save_config(self, config_file: Optional[str] = None):
        """
        Save current configuration to file.
        
        Args:
            config_file: Path to config file (default: output_dir/config.json)
        """
        if config_file is None:
            config_file = self.output_dir / "config.json"
        else:
            config_file = Path(config_file)
            
        config = {
            'model_name': self.model.__class__.__name__,
            'output_dir': str(self.output_dir),
            'supported_methods': self.SUPPORTED_METHODS,
            'cached_visualizers': list(self._visualizers.keys()),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"Saved configuration to: {config_file}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        try:
            # Remove all hooks
            if hasattr(self, 'hook_manager'):
                self.hook_manager.remove_all_hooks()
            
            # Clear visualizer cache
            self.clear_cache()
            
            # Save final configuration
            self.save_config()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            
        return False  # Don't suppress exceptions
    
    def __del__(self):
        """Cleanup resources."""
        try:
            # Ensure hooks are removed
            if hasattr(self, 'hook_manager'):
                # Let hook manager handle its own cleanup
                pass
            
            # Clear visualizers
            if hasattr(self, '_visualizers'):
                self._visualizers.clear()
                
        except Exception:
            # Ignore errors during deletion
            pass