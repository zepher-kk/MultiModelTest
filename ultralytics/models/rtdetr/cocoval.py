import csv
import torch
from ultralytics.models.yolo.multimodal.cocoval import MultiModalCOCOValidator
from ultralytics.utils import ops


class RTDETRMMCOCOValidator(MultiModalCOCOValidator):
    """COCO validator for RT-DETR multi-modal models."""
    
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize RT-DETR multi-modal COCO validator.
        
        Args:
            dataloader: Dataloader for validation dataset
            save_dir: Directory to save validation results
            pbar: Progress bar instance
            args: Validation arguments
            _callbacks: Callback functions
        """
        super().__init__(dataloader=dataloader, save_dir=save_dir, pbar=pbar, args=args, _callbacks=_callbacks)
    
    def init_metrics(self, model):
        """Initialize metrics for RT-DETR multi-modal COCO validation.
        
        Args:
            model: The RT-DETR model to extract metrics configuration from
        """
        # Call parent class init_metrics
        super().init_metrics(model)
        self.model = model
        
        # Initialize RT-DETR specific metrics
        self.end2end = getattr(model, "end2end", False)
        self.names = getattr(model, 'names', {})
        self.seen = 0
        self.jdict = []
        self.num_images_processed = 0
    
    def get_desc(self):
        """Return a formatted string for the progress bar description.
        
        Returns:
            str: Formatted string with column headers for validation output
        """
        return f"%22s" + "%11s" * 5 % ("Class", "Images", "Instances", "RTDETRMM", "COCO-mAP@.5:.95")
    
    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs.
        
        Args:
            preds: Raw predictions from RT-DETR model as a tuple.
        
        Returns:
            List[Dict[str, torch.Tensor]]: List of dictionaries for each image, each containing:
                - 'bboxes': Tensor of shape (N, 4) with bounding box coordinates
                - 'conf': Tensor of shape (N,) with confidence scores
                - 'cls': Tensor of shape (N,) with class indices
        """
        try:
            # Handle tuple input format for RT-DETR
            if not isinstance(preds, (list, tuple)):
                preds = [preds, None]
            
            # Extract batch size and dimensions
            bs, _, nd = preds[0].shape
            
            # Split predictions into bboxes and scores
            bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
            
            # Scale bboxes to image size
            bboxes *= self.args.imgsz
            
            # Initialize outputs
            outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
            
            # Process each image in the batch
            for i, bbox in enumerate(bboxes):
                # Convert from xywh to xyxy format
                bbox = ops.xywh2xyxy(bbox)
                
                # Get max score and class for each prediction
                score, cls = scores[i].max(-1)
                
                # Combine bbox, score, and class
                pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)
                
                # Sort by confidence to correctly get internal metrics
                pred = pred[score.argsort(descending=True)]
                
                # Filter predictions based on confidence threshold
                outputs[i] = pred[score > self.args.conf]
            
            # Return formatted results
            return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5]} for x in outputs]
            
        except Exception as e:
            # Handle any errors gracefully
            self.logger.warning(f"Error in postprocess: {str(e)}")
            # Return empty predictions on error
            return [{"bboxes": torch.zeros((0, 4)), "conf": torch.zeros(0), "cls": torch.zeros(0)}]
    
    def _prepare_pred(self, pred, pbatch):
        """Prepare predictions by scaling bounding boxes to original image dimensions.

        Args:
            pred (Dict[str, torch.Tensor]): Raw predictions containing 'cls', 'bboxes', and 'conf'.
            pbatch (Dict[str, torch.Tensor]): Prepared batch information containing 'ori_shape' and other metadata.

        Returns:
            (Dict[str, torch.Tensor]): Predictions scaled to original image dimensions.
        """
        cls = pred["cls"]
        if self.args.single_cls:
            cls *= 0
        bboxes = pred["bboxes"].clone()
        bboxes[..., [0, 2]] *= pbatch["ori_shape"][1] / self.args.imgsz  # native-space pred
        bboxes[..., [1, 3]] *= pbatch["ori_shape"][0] / self.args.imgsz  # native-space pred
        return {"bboxes": bboxes, "conf": pred["conf"], "cls": cls}
    
    def print_results(self):
        """Print COCO evaluation results and save CSV files.
        
        This method calls the parent class print_results which includes
        automatic CSV file generation through _save_csv_results().
        """
        # Call parent class print_results which handles everything including CSV saving
        super().print_results()