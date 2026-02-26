import numpy as np
import torch
import torch.nn as nn
import torchvision
import ssl
import shutil
from pathlib import Path
from torchvision.models.detection.keypoint_rcnn import (
        keypointrcnn_resnet50_fpn,
        KeypointRCNNPredictor
    )
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms

from huggingface_hub import HfApi

if __name__ == '__main__':
    from data import IMAGENET_MEAN, IMAGENET_STD
    import huggingface as _hf
else:
    from src.data import IMAGENET_MEAN, IMAGENET_STD
    import src.huggingface as _hf

# Bypass SSL certificate verification (temporary solution)
ssl._create_default_https_context = ssl._create_unverified_context

class FasterRCNN(nn.Module):
    """Keypoint R-CNN for keypoint detection based on Faster R-CNN"""
    def __init__(self, num_classes:int, num_keypoints:int, disable_box_regression: bool = True):
        super().__init__()
        # Load pre-trained Keypoint R-CNN model (trained on COCO)
        self.model = keypointrcnn_resnet50_fpn(weights="DEFAULT")
        
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        
        # Adjust Box-Predictor for our number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Adjust Keypoint-Predictor for our number of keypoints (3 instead of 17 like in COCO)
        in_features_kp = self.model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
        self.model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(
            in_features_kp,
            num_keypoints
        )
        
        self.disable_box_regression = disable_box_regression
        self.normalize = torchvision.transforms.Normalize(
            mean    = IMAGENET_MEAN, 
            std     = IMAGENET_STD
        )
    
    def forward(self, images, targets=None):
        losses = self.model(images, targets)
        
        # Remove Box Regression Losses, if enabled
        if self.disable_box_regression and isinstance(losses, dict):
            losses = {k: v for k, v in losses.items() if 'box_reg' not in k}
        
        return losses
    
    def _apply_nms(self, predictions, iou_threshold=0.3):
        """Apply Non-Maximum Suppression to predictions.
        
        Args:
            prediction: Dictionary with 'boxes', 'scores', 'labels', 'keypoints'
            iou_threshold: IoU threshold for NMS
        
        Returns:
            Filtered prediction dictionary
        """
        filtered_predictions = []

        for prediction in predictions:
            if len(prediction['boxes']) == 0:
                return prediction
            
            # Apply NMS on boxes
            keep_indices = nms(prediction['boxes'], prediction['scores'], iou_threshold)
            
            # Filter all prediction components
            filtered_prediction = {
                'boxes': prediction['boxes'][keep_indices].cpu(),
                'labels': prediction['labels'][keep_indices].cpu(),
                'scores': prediction['scores'][keep_indices].cpu(),
                'keypoints': prediction['keypoints'][keep_indices].cpu()
            }
            filtered_predictions.append(filtered_prediction)
        
        return filtered_predictions
    
    def predict(self, image, iou_threshold:float = 0.3):
        for param in self.parameters():
            device = param.device
            break

        self.eval()
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)         # convert to tensor
        else:
            image = image.clone()
        if len(image.shape) != 4:
            was_batch = False
            image = image.unsqueeze(0)          # add batch dimension
        else:
            was_batch = True
        if image.shape[1] != 3:                 # B, H, W, C
            image = image.permute(0, 3, 1, 2)   # B, C, H, W
        if image.dtype not in [torch.float32, torch.float64]:
            image = image / 255

        image = image.to(device)
        image = self.normalize(image)

        with torch.no_grad():
            predictions = self(image)
            predictions = self._apply_nms(predictions, iou_threshold=iou_threshold)
        
        if was_batch:
            return predictions
        return predictions[0]
        
    def to(self, device):
        self.model.to(device)
        self.device = device
        return super().to(device)

if __name__ == '__main__':
    RUN_ID = "ObjectDataset/FasterRCNN_1000"
    _hf.publish_model(run_id=RUN_ID, publish_name="DHSN-BottleOpener_1000")

    _hf.load_state_dict(publish_name="DHSN-BottleOpener_1000")