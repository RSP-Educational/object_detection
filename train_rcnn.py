import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from typing import Tuple
import src.data as data
import src.visualization as vis

class ObjectDataset(Dataset):
    def __init__(
            self,
            split:str,
            image_size:Tuple[int, int]=(1024, 1024),
            return_filename:bool = False
        ):
        super().__init__()

        assert split in ["train", "val", "test"], f"Split '{split}' is invalid. Expected 'train', 'val' or 'test'."
       
        self.split = split
        self.annotations = data.load_annotations(split=split)
        self.NUM_CLASSES = max([max(ann['classes'] + 1) for ann in self.annotations]) + 1 # inner +1 because index-based; outer +1 because background class = 0
        self.NUM_KEYPOINTS = self.annotations[0]['points'].shape[0]
        self.return_filename = return_filename

        self.image_size = image_size
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        try:
            import cv2 as cv
            import src.augmentation as aug

            ann = self.annotations[index]
            img = cv.imread(ann["img_file"]) / 255.
            if img is None:
                raise ValueError(f"Could not read image: {ann['img_file']}")
            img = cv.resize(img, self.image_size)

            keypoints = ann["points"]

            if self.split == "train":
                img, keypoints = aug.augment_perspective(img, keypoints, distortion=0.05, p=0.3)
                img, keypoints = aug.augment_rotate(img, keypoints, angle_shift=10, scale_factor=0.05, p=0.3)
                img = aug.augment_image_numpy(img)

            boxes = []
            N_obj = keypoints.shape[0] // 3
            keypoints_formatted = np.zeros((N_obj, 3, 3))
            
            
            # Konvertiere relative Koordinaten in absolute Pixelkoordinaten
            img_height, img_width = self.image_size[1], self.image_size[0]
            
            for i, kpts in enumerate(keypoints.copy().reshape(N_obj, 3, 2)):
                # Skaliere Keypoints auf absolute Pixelkoordinaten
                kpts_abs = kpts.copy()
                kpts_abs[:, 0] *= img_width   # x-Koordinaten
                kpts_abs[:, 1] *= img_height  # y-Koordinaten
                
                # Berechne Bounding Box aus den Keypoints (in Pixelkoordinaten)
                # WICHTIG: Margin proportional zur Bildgröße für bessere Generalisierung
                # und um sicherzustellen, dass Keypoints nach Augmentation nicht abgeschnitten werden
                margin_ratio = 0.15  # 15% der Keypoint-Spanne als Margin
                kpt_width = kpts_abs[:, 0].max() - kpts_abs[:, 0].min()
                kpt_height = kpts_abs[:, 1].max() - kpts_abs[:, 1].min()
                margin_x = max(30, kpt_width * margin_ratio)  # Minimum 30 Pixel
                margin_y = max(30, kpt_height * margin_ratio)
                
                xmin, xmax = kpts_abs[:, 0].min() - margin_x, kpts_abs[:, 0].max() + margin_x
                ymin, ymax = kpts_abs[:, 1].min() - margin_y, kpts_abs[:, 1].max() + margin_y
                
                # Clip auf Bildgrenzen
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(img_width, xmax), min(img_height, ymax)
                
                boxes.append([xmin, ymin, xmax, ymax])
                
                # Formatiere Keypoints für Keypoint R-CNN: [K, 3] wobei 3 = (x, y, visibility)
                # visibility: 0 = nicht annotiert, 1 = annotiert aber nicht sichtbar, 2 = annotiert und sichtbar
                kpts_with_visibility = np.concatenate([kpts_abs, np.ones((3, 1)) * 2], axis=1)  # alle Keypoints sind sichtbar
                keypoints_formatted[i] = kpts_with_visibility

            img = torch.tensor(img).permute(2, 0, 1).float()
            img = self.normalize(img)

            target = {
                "boxes": torch.tensor(boxes).float(),
                "keypoints": torch.tensor(keypoints_formatted).float(),  # Shape: [N, 3, 3] für N Objekte mit je 3 Keypoints
                "labels": torch.tensor(ann["classes"], dtype=torch.int64) + 1  # +1, because background class = 0
            }

            if self.return_filename:
                return img, target, ann["img_file"]
            return img, target
        except Exception as e:
            print(f"Exception: {e}")

    def collate_fn(batch):
        return tuple(zip(*batch))
    
    def get_weighted_sampler(self):
        """Erstellt Sampler basierend auf der seltensten Klasse pro Bild"""
        from collections import Counter
        
        # Zähle Objekte pro Klasse
        class_counts = Counter()
        for ann in self.annotations:
            for label in ann['classes']:
                class_counts[label] += 1
        
        print(f"\nClass distribution: {dict(class_counts)}")
        
        # Berechne Gewichte (inverse frequency)
        total = sum(class_counts.values())
        class_weights = {}
        for label, count in class_counts.items():
            class_weights[label] = total / (len(class_counts) * count)
        
        print(f"Class weights: {class_weights}")
        
        # Jedes Bild bekommt das HÖCHSTE Gewicht seiner Klassen (meistens sowieso nur eine Klasse pro Bild)
        sample_weights = []
        for ann in self.annotations:
            labels = ann['classes']
            # Nutze die seltenste Klasse im Bild
            weights = [class_weights[label] for label in labels]
            sample_weights.append(max(weights))
        
        return torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

# ds_train = ObjectDataset(split="train")
# imgs, points, titles = [], [], []
# for i in np.random.randint(0, len(ds_train), size=6):
#     img, target = ds_train[i]

#     imgs.append(img)
#     points.append(target["keypoints"])
#     titles.append("test")
    
#vis.plot_images_with_points(imgs, points, titles)
pass

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
        
        # HINWEIS: Der Keypoint-Loss verwendet intern OKS (Object Keypoint Similarity)
        # mit COCO-Sigmas. Diese sind für 17 Körper-Keypoints optimiert.
        # Für 3 geometrische Punkte wären andere Sigmas besser, aber torchvision
        # bietet keine einfache API um diese zu ändern.
        # LÖSUNG: Längeres Training + optimierte Box-Größen (siehe __getitem__)
        
        self.disable_box_regression = disable_box_regression
    
    def forward(self, images, targets=None):
        losses = self.model(images, targets)
        
        # Remove Box Regression Losses, if enabled
        if self.disable_box_regression and isinstance(losses, dict):
            losses = {k: v for k, v in losses.items() if 'box_reg' not in k}
        
        return losses
    
# IMPORTS
from tqdm import tqdm
from src.run import Run
from pathlib import Path
from torch.utils.data import DataLoader
import multiprocessing as mp

# TRAINING SETUP
RUN_ID              = "ObjectDataset/FasterRCNN_512_fixed"
EPOCHS              = 60  # Erhöhe Epochen - der Loss sinkt noch!
LEARNING_RATE       = 2e-4  # Leicht erhöht für schnelleres Keypoint-Learning
MIN_LEARNING_RATE   = 1e-6
BATCH_SIZE          = 2
IMAGE_SIZE          = (512, 512)#(1024, 1024)
DEVICE              = 'cuda' if torch.cuda.is_available() else 'cpu'
WARMUP_ITERARTIONS  = 800
BATCHES_PER_EPOCH   = 500000
PLOT_ITERATIONS     = 10
NUM_WORKERS         = 6

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)  # safer with OpenCV

    print(f"Running on device {DEVICE}")

    # DATA
    ds_train = ObjectDataset(split="train", image_size=IMAGE_SIZE)
    ds_val = ObjectDataset(split="val", image_size=IMAGE_SIZE)
    train_sampler = ds_train.get_weighted_sampler()

    train_loader = DataLoader(
        dataset             = ds_train,
        batch_size          = BATCH_SIZE,
        sampler             = train_sampler,
        num_workers         = NUM_WORKERS,
        persistent_workers  = NUM_WORKERS > 0,
        pin_memory          = True,
        collate_fn          = ObjectDataset.collate_fn
    )

    val_loader = DataLoader(
        dataset             = ds_val,
        batch_size          = BATCH_SIZE,
        collate_fn          = ObjectDataset.collate_fn,
        num_workers         = NUM_WORKERS,
        persistent_workers  = NUM_WORKERS > 0,
        pin_memory          = True,
        shuffle             = True
    )

    # MODEL - Verwende KeypointRCNN statt FasterRCNN für Keypoint-Detektion
    model = FasterRCNN(
        num_classes             = ds_train.NUM_CLASSES,
        num_keypoints           = ds_train.NUM_KEYPOINTS,
        disable_box_regression  = False
    )
    model.to(DEVICE)

    # OPTIMIZER
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=0.0001
    )

    # RUN
    run = Run(id=RUN_ID, device=DEVICE)
    run.load(model, optimizer)

    # LEARNING RATE SCHEDULER
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(EPOCHS * min(BATCHES_PER_EPOCH, len(train_loader))) - WARMUP_ITERARTIONS,
        eta_min=MIN_LEARNING_RATE
    )

    # TRAINING LOOP
    best_loss_val = min(run.get_values('loss', 'val'))
    if np.isnan(best_loss_val):
        best_loss_val = np.inf
    for epoch in range(run.epoch, EPOCHS):
        losses_epoch = { 'train': [], 'val': [] }

        # Training
        model.train()

        prog_train = tqdm(train_loader, leave=False, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for i, (images, targets) in enumerate(prog_train):
            _iter = i/min(BATCHES_PER_EPOCH, len(train_loader))
            # Move data to device
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            
            for key, value in loss_dict.items():
                run.append(key, 'train', _iter, value.item())
            run.append('loss', 'train', _iter, loss.item())
            losses_epoch['train'].append(loss.item())
            prog_train.set_postfix({'loss': f"{sum(losses_epoch['train']) / len(losses_epoch['train']):.4f}"})

            # Use warmup learning rate -> slightly increase to target learning rate
            # -> achieve more robust parameter updates at the beginning
            # -> at the beginning, parameters are quiet bad. Updating them to fast leads to instable gradients
            if run.global_iteration < WARMUP_ITERARTIONS:
                warmup_lr = LEARNING_RATE * (run.global_iteration / WARMUP_ITERARTIONS)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            # reset old gradients
            optimizer.zero_grad()

            # Backward pass -> compute gradients
            loss.backward()

            # Gradient Clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            # Update Parameters
            optimizer.step()

            # if warmup completed, use the learning rate scheduler, which slightly decreases the learning rate
            # following cosine function
            if run.global_iteration >= WARMUP_ITERARTIONS:
                lr_scheduler.step()
            run.append('lr', 'train', i/min(BATCHES_PER_EPOCH, len(train_loader)), optimizer.param_groups[0]['lr'])

            # increase global_iteration for each train step -> required for learning rate scheduling
            run.global_iteration += 1

            if i%PLOT_ITERATIONS == 0 and i > 0:
                run.plot(show_cv=False)

            if i + 1 >= BATCHES_PER_EPOCH:
                break
        
        # Validation
        prog_val = tqdm(val_loader, leave=False, desc=f"Epoch {epoch}/{EPOCHS} [Val]")
        for i, (images, targets) in enumerate(prog_val):
            _iter = i/min(BATCHES_PER_EPOCH, len(val_loader))

            # Move data to device
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            # Forward pass
            with torch.no_grad():   # do not compute gradients -> computational more efficient
                loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            for key, value in loss_dict.items():
                run.append(key, 'val', _iter, value.item())
            run.append('loss', 'val', _iter, loss.item())
            losses_epoch['val'].append(loss.item())
            prog_val.set_postfix({'loss': f"{sum(losses_epoch['val']) / len(losses_epoch['val']):.4f}"})

            if i%PLOT_ITERATIONS == 0 and i > 0:
                run.plot(show_cv=False)

            if i+1 >= BATCHES_PER_EPOCH:
                break
        
        
        print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {run.get_last_epoch_value('loss', 'train'):.4f}, Val Loss: {run.get_last_epoch_value('loss', 'val'):.4f}")
        
        run.plot(show_cv=False)
        run.epoch += 1

        # Save best model
        loss_val = run.get_last_epoch_value('loss', 'val')
        if loss_val < best_loss_val:
            best_loss_val = loss_val
            run.save(model, optimizer)
            print(f"✓ Saved checkpoint for epoch {run.epoch} with val loss {run.get_last_epoch_value('loss', 'val'):.4f}")

    print(f"\nTraining completed! Best validation loss: {best_loss_val:.4f}")