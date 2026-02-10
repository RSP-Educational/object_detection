import torch
import torch.nn as nn
import ssl
from torchvision.models.detection.keypoint_rcnn import (
        keypointrcnn_resnet50_fpn,
        KeypointRCNNPredictor
    )
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Bypass SSL certificate verification (temporary solution)
ssl._create_default_https_context = ssl._create_unverified_context

class FasterRCNN(nn.Module):
    """Keypoint R-CNN for keypoint detection based on Faster R-CNN"""
    def __init__(self, num_classes:int, num_keypoints:int, disable_box_regression: bool = True):
        super().__init__()
        # Load pre-trained Keypoint R-CNN model (trained on COCO)
        self.model = keypointrcnn_resnet50_fpn(weights="DEFAULT")
        
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
    
    def forward(self, images, targets=None):
        losses = self.model(images, targets)
        
        # Remove Box Regression Losses, if enabled
        if self.disable_box_regression and isinstance(losses, dict):
            losses = {k: v for k, v in losses.items() if 'box_reg' not in k}
        
        return losses