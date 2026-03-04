import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data import ObjectDataset
from src.model import FasterRCNN
from src.run import Run
from torch.utils.data import DataLoader
import multiprocessing as mp

# TRAINING SETUP
RUN_ID              = "ObjectDataset/FasterRCNN_1000"
EPOCHS              = 80
LEARNING_RATE       = 2e-4
MIN_LEARNING_RATE   = 1e-6
BATCH_SIZE          = 2
IMAGE_SIZE          = (1000, 1000)
DEVICE              = 'cuda' if torch.cuda.is_available() else 'cpu'
WARMUP_ITERARTIONS  = 800
BATCHES_PER_EPOCH   = 500000
PLOT_ITERATIONS     = 20
NUM_WORKERS         = 6#6

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
        #sampler             = train_sampler,
        shuffle             = True,
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
    best_loss_val = min(run._get_epoch_mean_values('loss', 'val', return_epochs=False))
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
            run.save(
                model       = model,
                optimizer   = optimizer,
                parameters  = {
                    'num_classes':      ds_train.NUM_CLASSES,
                    'num_keypoints':    ds_train.NUM_KEYPOINTS,
                    'image_size': IMAGE_SIZE
                }
            )
            print(f"✓ Saved checkpoint for epoch {run.epoch} with val loss {run.get_last_epoch_value('loss', 'val'):.4f}")

    print(f"\nTraining completed! Best validation loss: {best_loss_val:.4f}")