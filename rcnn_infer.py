import torch
import cv2 as cv
import numpy as np
from pathlib import Path
from torchvision.ops import nms

from src.data import ObjectDataset
from src.model import FasterRCNN
from src.run import Run

RUN_ID              = "ObjectDataset/FasterRCNN_800"
IMAGE_SIZE          = (800, 800)
DEVICE              = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESHOLD           = 0.75
NMS_THRESHOLD       = 0.3  # IoU threshold for NMS (lower = stricter)

COLOR_TARGET        = (0.5, 0.5, 1.0)
COLOR_PREDICTION    = (1.0, 0.5, 0.5)

def apply_nms(prediction, iou_threshold=0.3):
    """Apply Non-Maximum Suppression to predictions.
    
    Args:
        prediction: Dictionary with 'boxes', 'scores', 'labels', 'keypoints'
        iou_threshold: IoU threshold for NMS
    
    Returns:
        Filtered prediction dictionary
    """
    if len(prediction['boxes']) == 0:
        return prediction
    
    # Apply NMS on boxes
    keep_indices = nms(prediction['boxes'], prediction['scores'], iou_threshold)
    
    # Filter all prediction components
    filtered_prediction = {
        'boxes': prediction['boxes'][keep_indices],
        'labels': prediction['labels'][keep_indices],
        'scores': prediction['scores'][keep_indices],
        'keypoints': prediction['keypoints'][keep_indices]
    }
    
    return filtered_prediction

def get_point_tuples(pts):
    p1 = (int(round(pts[0, 0])), int(round(pts[0, 1])))
    p2 = (int(round(pts[1, 0])), int(round(pts[1, 1])))
    p3 = (int(round(pts[2, 0])), int(round(pts[2, 1])))
    return p1, p2, p3

def vis_prediction(image, target, prediction):
    def draw_object(img, pts, color, thickness = 2):
        p1, p2, p3 = get_point_tuples(pts)

        cv.line(img, p1, p2, color, thickness=thickness)
        cv.line(img, p2, p3, color, thickness=thickness)
        cv.line(img, p3, p1, color, thickness=thickness)

        pc = (int(round(p1[0] + (p2[0] - p1[0]) / 2)), int(round(p1[1] + (p2[1] - p1[1]) / 2)))
        cv.line(img, pc, p3, color, thickness=thickness)
    
    def draw_label(img, p, label, score):
        p = (p[0] + 5, p[1])
        cv.putText(img, f"{label} ({score:0.2f})", p, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=COLOR_PREDICTION, thickness=2)

    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]
    image = image.squeeze(0).cpu() * torch.tensor(std).view(3,1,1) + torch.tensor(mean).view(3,1,1)
    image = image.permute(1, 2, 0).cpu().numpy()

    t_keypoints = target['keypoints'][:, :, :2].cpu().numpy()

    for t_kpts in t_keypoints:
        draw_object(image, t_kpts, COLOR_TARGET)

    p_keypoints = prediction['keypoints'][:, :, :2].cpu().numpy()
    p_labels = prediction['labels'].cpu().numpy()
    p_scores = prediction['scores'].cpu().numpy()

    for score, label_idx, kpts in zip(p_scores, p_labels, p_keypoints):
        if score < THRESHOLD:
            continue
        draw_object(image, kpts, COLOR_PREDICTION)

        p1, p2, p3 = get_point_tuples(kpts)
        p = p1 if p1[0] > p2[0] and p1[0] > p3[0] else p2 if p2[0] > p3[0] else p3

        label = ObjectDataset.CLASS_LABELS[label_idx - 1]
        draw_label(image, p, label, score)

    cv.imshow('Preview', image)
    cv.waitKey()
    pass

if __name__ == '__main__':
    # DATASET
    ds_test = ObjectDataset(
        split                   = 'val',
        image_size              = IMAGE_SIZE,
        return_filename         = True
    )

    # MODEL
    model = FasterRCNN(
        num_classes             = ds_test.NUM_CLASSES,
        num_keypoints           = ds_test.NUM_KEYPOINTS,
        disable_box_regression  = False
    ).to(DEVICE)
    model.eval()

    # RUN
    run = Run(
        id      = RUN_ID,
        device  = DEVICE
    )
    run.load(model)

    indices = np.random.randint(0, len(ds_test), len(ds_test))

    # VISUALIZE DATA
    for i in indices:
        image, target, fname = ds_test[i]

        # Move data to device
        image = image.to(DEVICE).unsqueeze(0)
        target = {k: v.to(DEVICE) for k, v in target.items()}

        with torch.no_grad():
            prediction = model(image)[0]
            # Optional: Apply additional NMS if needed (model already applies NMS internally)
            prediction = apply_nms(prediction, iou_threshold=NMS_THRESHOLD)
            pass

        vis_prediction(image, target, prediction)
        pass