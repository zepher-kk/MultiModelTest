"""
Feature map visualization module for YOLOMM multi-modal detection.

This module provides the FeatureMapVisualizer class for extracting and visualizing
intermediate feature maps from model layers during forward pass.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import cv2

from .manager import FeatureMapResult
from .utils import HookManager, load_image


class FeatureMapVisualizer:
    """
    Visualizer for extracting and rendering feature maps from model layers.
    
    This class uses forward hooks to capture intermediate feature maps during
    model inference and provides methods to visualize selected channels with
    various selection strategies.
    
    Attributes:
        model: The model to extract feature maps from
        hook_manager: Manager for registering and handling forward hooks
    """
    
    def __init__(self, model: torch.nn.Module):
        """
        Initialize the feature map visualizer.
        
        Args:
            model: PyTorch model to extract feature maps from
        """
        self.model = model
        self.hook_manager = HookManager(model)
    
    def visualize(
        self,
        images: Union[str, np.ndarray, torch.Tensor, Dict[str, Any]],
        layers: Optional[List[Union[str, int]]] = None,
        top_k: int = 8,
        selection_method: str = 'sum',
        **kwargs
    ) -> FeatureMapResult:
        """
        Extract and visualize feature maps from specified layers.
        
        Args:
            images: Input image(s) - can be path, numpy array, tensor, or dict for multi-modal
            layers: List of layer names/indices to extract features from. If None, uses default layers
            top_k: Number of top channels to visualize per layer
            selection_method: Method for selecting channels ('sum' or 'var')
            **kwargs: Additional arguments for rendering
            
        Returns:
            FeatureMapResult containing the rendered feature map grid and metadata
        """
        # Handle multi-modal inputs
        if isinstance(images, dict):
            results = {}
            for modality, img in images.items():
                results[modality] = self._visualize_single_modality(
                    img, layers, top_k, selection_method, **kwargs
                )
            # Combine results from multiple modalities
            return self._combine_multimodal_results(results)
        else:
            # Single modality input
            return self._visualize_single_modality(
                images, layers, top_k, selection_method, **kwargs
            )
    
    def _visualize_single_modality(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        layers: Optional[List[Union[str, int]]],
        top_k: int,
        selection_method: str,
        **kwargs
    ) -> FeatureMapResult:
        """
        Visualize feature maps for a single modality input.
        
        Args:
            image: Input image
            layers: Layers to extract from
            top_k: Number of channels to select
            selection_method: Channel selection method
            **kwargs: Additional rendering arguments
            
        Returns:
            FeatureMapResult for the single modality
        """
        # Load and preprocess image
        img_tensor = self._prepare_input(image)
        
        # Determine layers to hook if not specified
        if layers is None:
            layers = self._get_default_layers()
        
        # Extract feature maps using hooks
        feature_maps = self._extract_features(img_tensor, layers)
        
        # Select top channels from each layer
        selected_features = self._select_channels(
            feature_maps, top_k, selection_method
        )
        
        # Render feature maps as grid
        grid_image = self._render_grid(selected_features, **kwargs)
        
        # Create result object with metadata
        metadata = {
            'layers': layers,
            'top_k': top_k,
            'selection_method': selection_method,
            'num_features': len(selected_features),
            'feature_stats': self._compute_feature_stats(selected_features)
        }
        
        return FeatureMapResult(
            feature_maps=grid_image,
            metadata=metadata
        )
    
    def _prepare_input(self, image: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Prepare input image for model inference.
        
        Args:
            image: Input image in various formats
            
        Returns:
            Preprocessed tensor ready for model
        """
        # Load image if path
        if isinstance(image, str):
            image = load_image(image)
        
        # Convert to tensor if numpy
        if isinstance(image, np.ndarray):
            # Assume HWC format, convert to CHW
            if image.ndim == 3:
                image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image).float()
        
        # Ensure correct shape (add batch dimension if needed)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        image = image.to(device)
        
        return image
    
    def _get_default_layers(self) -> List[str]:
        """
        Get default layers to extract features from.
        
        Returns:
            List of default layer names
        """
        # Default to backbone output layers
        default_layers = []
        
        # Try to find backbone layers
        if hasattr(self.model, 'model'):
            model = self.model.model
            # Look for typical backbone stages
            for i, layer in enumerate(model):
                layer_name = layer.__class__.__name__
                if any(name in layer_name for name in ['Conv', 'C2f', 'C3', 'SPPF']):
                    if i in [4, 6, 8, 10]:  # Typical backbone output indices
                        default_layers.append(f'model.{i}')
        
        # Fallback to some reasonable defaults
        if not default_layers:
            default_layers = ['model.4', 'model.6', 'model.8', 'model.10']
        
        return default_layers
    
    def _extract_features(
        self,
        input_tensor: torch.Tensor,
        layers: List[Union[str, int]]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract feature maps from specified layers using hooks with GPU OOM fallback.
        
        Args:
            input_tensor: Input tensor for model
            layers: Layer names/indices to extract from
            
        Returns:
            Dictionary mapping layer names to feature tensors
        """
        # Store original device
        original_device = next(self.model.parameters()).device
        
        # Register hooks on specified layers
        for layer in layers:
            if isinstance(layer, int):
                # Convert index to module
                modules = list(self.model.modules())
                if 0 <= layer < len(modules):
                    self.hook_manager.register_forward_hook(
                        modules[layer], name=f'layer_{layer}'
                    )
                else:
                    # Provide helpful error for out-of-range index
                    raise ValueError(
                        f"Layer index {layer} is out of range.\n"
                        f"Model has {len(modules)} modules (indices 0 to {len(modules)-1})."
                    )
            else:
                # Layer specified by name/path
                try:
                    module = self._get_module_by_name(layer)
                    if module is not None:
                        self.hook_manager.register_forward_hook(module, name=layer)
                    else:
                        raise ValueError(f"Could not find module: {layer}")
                except ValueError as ve:
                    # Re-raise with additional context about available layers
                    available_layers = self._get_default_layers()
                    raise ValueError(
                        f"Failed to register hook for layer '{layer}':\n{str(ve)}\n\n"
                        f"Suggested layers to try:\n" + 
                        "\n".join([f"  - {l}" for l in available_layers[:5]])
                    )
        
        try:
            # Try feature extraction on current device (likely GPU)
            input_tensor = input_tensor.to(original_device)
            
            # Run forward pass to collect features
            with torch.no_grad():
                _ = self.model(input_tensor)
            
            # Get collected features
            features = self.hook_manager.get_features()
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                # GPU OOM detected, fallback to CPU
                print(f"WARNING: GPU out of memory during feature extraction. Falling back to CPU...")
                torch.cuda.empty_cache()  # Clear GPU memory
                
                # Clean up existing hooks
                self.hook_manager.remove_all_hooks()
                
                # Move model and data to CPU
                self.model.to('cpu')
                input_tensor = input_tensor.to('cpu')
                
                # Re-register hooks on CPU
                for layer in layers:
                    if isinstance(layer, int):
                        modules = list(self.model.modules())
                        if 0 <= layer < len(modules):
                            self.hook_manager.register_forward_hook(
                                modules[layer], name=f'layer_{layer}'
                            )
                    else:
                        try:
                            module = self._get_module_by_name(layer)
                            if module is not None:
                                self.hook_manager.register_forward_hook(module, name=layer)
                        except ValueError:
                            # Skip this layer if it cannot be found
                            print(f"Warning: Skipping layer '{layer}' during CPU fallback")
                            continue
                
                # Retry on CPU
                with torch.no_grad():
                    _ = self.model(input_tensor)
                
                # Get collected features
                features = self.hook_manager.get_features()
                
                # Move model back to original device
                self.model.to(original_device)
            else:
                # Re-raise if not OOM error
                raise
        finally:
            # Clean up hooks
            self.hook_manager.remove_all_hooks()
        
        return features
    
    def _get_module_by_name(self, name: str) -> Optional[torch.nn.Module]:
        """
        Get module by its name/path in the model.
        
        Args:
            name: Module name like 'model.4' or 'backbone.conv1'
            
        Returns:
            Module if found, None otherwise
        """
        parts = name.split('.')
        module = self.model
        
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            elif part.isdigit() and hasattr(module, '__getitem__'):
                module = module[int(part)]
            else:
                return None
        
        return module
    
    def _select_channels(
        self,
        feature_maps: Dict[str, torch.Tensor],
        top_k: int,
        method: str
    ) -> List[Tuple[str, int, torch.Tensor]]:
        """
        Select top channels from feature maps based on selection method with batch support.
        
        Args:
            feature_maps: Dictionary of layer features
            top_k: Number of channels to select per layer
            method: Selection method ('sum' or 'var')
            
        Returns:
            List of (layer_name, channel_idx, feature_map) tuples
            For batched inputs, returns nested list structure
        """
        selected = []
        
        for layer_name, features in feature_maps.items():
            # Features shape: [B, C, H, W]
            if features.ndim == 4:
                batch_size = features.shape[0]
                num_channels = features.shape[1]
                
                # Process each item in the batch
                for batch_idx in range(batch_size):
                    batch_features = features[batch_idx]
                    
                    # Compute selection metric for each channel
                    if method == 'sum':
                        # Sum of absolute activations
                        channel_scores = batch_features.abs().sum(dim=(1, 2))
                    elif method == 'var':
                        # Variance of activations
                        channel_scores = batch_features.var(dim=(1, 2))
                    else:
                        raise ValueError(f"Unknown selection method: {method}")
                    
                    # Select top k channels
                    k = min(top_k, num_channels)
                    top_indices = torch.topk(channel_scores, k).indices
                    
                    # Add selected channels to list
                    # Include batch index in tuple for batch-aware processing
                    for idx in top_indices:
                        selected.append((
                            layer_name,
                            idx.item(),
                            batch_features[idx].cpu().numpy(),
                            batch_idx  # Add batch index
                        ))
            elif features.ndim == 3:
                # Single image without batch dimension [C, H, W]
                num_channels = features.shape[0]
                
                # Compute selection metric for each channel
                if method == 'sum':
                    channel_scores = features.abs().sum(dim=(1, 2))
                elif method == 'var':
                    channel_scores = features.var(dim=(1, 2))
                else:
                    raise ValueError(f"Unknown selection method: {method}")
                
                # Select top k channels
                k = min(top_k, num_channels)
                top_indices = torch.topk(channel_scores, k).indices
                
                # Add selected channels to list
                for idx in top_indices:
                    selected.append((
                        layer_name,
                        idx.item(),
                        features[idx].cpu().numpy(),
                        0  # Default batch index for non-batched input
                    ))
        
        return selected
    
    def _render_grid(
        self,
        selected_features: List[Tuple[str, int, np.ndarray]],
        grid_size: Optional[Tuple[int, int]] = None,
        feature_size: Tuple[int, int] = (128, 128),
        show_stats: bool = True,
        colormap: str = 'viridis',
        **kwargs
    ) -> np.ndarray:
        """
        Render selected feature maps as a grid.
        
        Args:
            selected_features: List of (layer_name, channel_idx, feature_map) tuples
            grid_size: (rows, cols) for grid layout. Auto-calculated if None
            feature_size: Size to resize each feature map to
            show_stats: Whether to show statistics on each feature
            colormap: OpenCV colormap to apply
            **kwargs: Additional rendering options
            
        Returns:
            Rendered grid image as numpy array
        """
        if not selected_features:
            # Return empty image if no features
            return np.zeros((256, 256, 3), dtype=np.uint8)
        
        num_features = len(selected_features)
        
        # Calculate grid dimensions if not specified
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(num_features)))
            rows = int(np.ceil(num_features / cols))
        else:
            rows, cols = grid_size
        
        # Create grid image
        cell_h, cell_w = feature_size
        padding = 5
        text_height = 30 if show_stats else 20
        
        grid_h = rows * (cell_h + text_height + padding) + padding
        grid_w = cols * (cell_w + padding) + padding
        grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
        
        # Get colormap
        cmap = getattr(cv2, f'COLORMAP_{colormap.upper()}', cv2.COLORMAP_VIRIDIS)
        
        # Place each feature map in grid
        for idx, item in enumerate(selected_features):
            # Handle both old format (3-tuple) and new format (4-tuple with batch_idx)
            if len(item) == 4:
                layer_name, channel_idx, feature_map, batch_idx = item
            else:
                layer_name, channel_idx, feature_map = item
                batch_idx = 0
                
            row = idx // cols
            col = idx % cols
            
            # Calculate position
            y = row * (cell_h + text_height + padding) + padding
            x = col * (cell_w + padding) + padding
            
            # Normalize feature map to 0-255
            feat_norm = self._normalize_feature(feature_map)
            
            # Resize to cell size
            feat_resized = cv2.resize(feat_norm, (cell_w, cell_h))
            
            # Apply colormap
            feat_colored = cv2.applyColorMap(feat_resized, cmap)
            
            # Place in grid
            grid[y:y+cell_h, x:x+cell_w] = feat_colored
            
            # Add text annotations
            text_y = y + cell_h + 15
            
            # Layer and channel info
            label = f"{layer_name.split('.')[-1]} ch:{channel_idx}"
            cv2.putText(grid, label, (x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Statistics if requested
            if show_stats:
                stats_text = f"μ:{feature_map.mean():.1f} σ:{feature_map.std():.1f}"
                cv2.putText(grid, stats_text, (x, text_y + 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        
        return grid
    
    def _normalize_feature(self, feature: np.ndarray) -> np.ndarray:
        """
        Normalize feature map to 0-255 range for visualization.
        
        Args:
            feature: 2D feature map
            
        Returns:
            Normalized feature map as uint8
        """
        # Handle edge cases
        if feature.size == 0:
            return np.zeros_like(feature, dtype=np.uint8)
        
        # Normalize to 0-1
        f_min, f_max = feature.min(), feature.max()
        if f_max > f_min:
            feature_norm = (feature - f_min) / (f_max - f_min)
        else:
            feature_norm = np.zeros_like(feature)
        
        # Convert to 0-255
        feature_uint8 = (feature_norm * 255).astype(np.uint8)
        
        return feature_uint8
    
    def _compute_feature_stats(
        self,
        selected_features: List[Tuple[str, int, np.ndarray]]
    ) -> Dict[str, Any]:
        """
        Compute statistics for selected feature maps.
        
        Args:
            selected_features: List of selected feature maps
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'num_features': len(selected_features),
            'layers': {},
        }
        
        # Group by layer
        for item in selected_features:
            # Handle both old format (3-tuple) and new format (4-tuple with batch_idx)
            if len(item) == 4:
                layer_name, channel_idx, feature_map, batch_idx = item
            else:
                layer_name, channel_idx, feature_map = item
                
            if layer_name not in stats['layers']:
                stats['layers'][layer_name] = {
                    'channels': [],
                    'mean_activation': 0,
                    'max_activation': -float('inf'),
                    'min_activation': float('inf'),
                }
            
            layer_stats = stats['layers'][layer_name]
            layer_stats['channels'].append(channel_idx)
            layer_stats['mean_activation'] += feature_map.mean()
            layer_stats['max_activation'] = max(
                layer_stats['max_activation'], feature_map.max()
            )
            layer_stats['min_activation'] = min(
                layer_stats['min_activation'], feature_map.min()
            )
        
        # Average the mean activations
        for layer_stats in stats['layers'].values():
            num_channels = len(layer_stats['channels'])
            if num_channels > 0:
                layer_stats['mean_activation'] /= num_channels
        
        return stats
    
    def _combine_multimodal_results(
        self,
        results: Dict[str, FeatureMapResult]
    ) -> FeatureMapResult:
        """
        Combine feature map results from multiple modalities.
        
        Args:
            results: Dictionary mapping modality names to their results
            
        Returns:
            Combined FeatureMapResult
        """
        # Stack modality grids vertically
        grids = []
        combined_metadata = {
            'modalities': {},
            'combined': True
        }
        
        for modality, result in results.items():
            grids.append(result.feature_maps)
            combined_metadata['modalities'][modality] = result.metadata
        
        # Combine grids with modality labels
        combined_grid = self._stack_grids_with_labels(grids, list(results.keys()))
        
        return FeatureMapResult(
            feature_maps=combined_grid,
            metadata=combined_metadata
        )
    
    def _stack_grids_with_labels(
        self,
        grids: List[np.ndarray],
        labels: List[str]
    ) -> np.ndarray:
        """
        Stack multiple grids vertically with labels.
        
        Args:
            grids: List of grid images
            labels: List of labels for each grid
            
        Returns:
            Combined grid with labels
        """
        if not grids:
            return np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Find maximum width
        max_width = max(g.shape[1] for g in grids)
        label_height = 30
        
        # Pad grids to same width and add labels
        labeled_grids = []
        for grid, label in zip(grids, labels):
            h, w = grid.shape[:2]
            
            # Create labeled grid
            labeled = np.ones((h + label_height, max_width, 3), dtype=np.uint8) * 255
            
            # Center the grid horizontally if needed
            x_offset = (max_width - w) // 2
            labeled[label_height:, x_offset:x_offset+w] = grid
            
            # Add modality label
            cv2.putText(labeled, f"Modality: {label}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            labeled_grids.append(labeled)
        
        # Stack vertically
        combined = np.vstack(labeled_grids)
        
        return combined