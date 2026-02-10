import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import io
from PIL import Image
import cv2 as cv

PLOT_COLORS = [
    'tab:blue',
    'tab:red',
    'tab:orange'
]

def _moving_average(data, window:int):
    _data = np.array(data)
    _mavg = np.zeros(len(data), dtype=np.float32)
    for i in range(len(data)):
        if i < window:
            _mavg[i] = np.mean(_data[:i+1])
        else:
            _mavg[i] = np.mean(_data[i-window:i])
    return _mavg

def _plt2img():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image = Image.open(buf)
    image_array = np.array(image)
    image_array = cv.cvtColor(image_array, cv.COLOR_RGBA2BGR)

    buf.close()
    return image_array

class Run:
    def __init__(
            self,
            id:str,
            device:str,
            mavg_epochs:int = 10
    ):
        self.id = id
        self.device = device
        self.mavg_epochs = mavg_epochs

        self.run_dir = Path('runs').joinpath(self.id)
        self.run_dir.mkdir(exist_ok=True, parents=True)

        self.plot_dir = self.run_dir.joinpath('plot')
        self.plot_dir.mkdir(exist_ok=True, parents=True)

        self.checkpoint_file = self.run_dir.joinpath('checkpoint.ckpt')

        self.data = {}

        self.epoch = 0
        self.global_iteration = 0

    def load(self, model:nn.Module, optimizer:optim.Optimizer = None):
        if self.checkpoint_file.exists():
            checkpoint = torch.load(self.checkpoint_file, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.data = checkpoint['data']
            self.global_iteration = checkpoint['global_iteration']
            self.epoch = checkpoint['epoch']
            best_val_loss = min(self.get_values('loss', 'val'))
            print(f"✓ Loaded checkpoint from epoch {self.epoch} with val loss {best_val_loss:.4f}")

    def save(self, model:nn.Module, optimizer:optim.Optimizer = None):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': None if optimizer is None else optimizer.state_dict(),
            'data': self.data,
            'epoch': self.epoch,
            'global_iteration': self.global_iteration
        }
        torch.save(checkpoint, self.checkpoint_file)

    def append(self, key:str, split:str, iteration:float, value:float):
        if key not in self.data:
            self.data[key] = {}
        if split not in self.data[key]:
            self.data[key][split] = {
                'iteration': [],
                'value': []
            }
        iter = self.epoch + iteration
        if iteration >= 1.:
            iter -= 1e-5
        self.data[key][split]['iteration'].append(iter)
        self.data[key][split]['value'].append(value)

    def get_values(self, key, split:str):
        if key not in self.data:
            return [np.nan]
        if split not in self.data[key]:
            return [np.nan]
        if len(self.data[key][split]) == 0:
            return [np.nan]
        return self.data[key][split]['value']
    
    def get_last_epoch_value(self, key, split):
        epochs, values = self._get_epoch_mean_values(key, split)
        return values[-1]
    
    def _get_epoch_mean_values(self, key:str, split:str):
        epochs = np.array(self.data[key][split]['iteration'], dtype=np.int32)
        epochs_unique = np.unique(epochs)
        values = np.array(self.data[key][split]['value'], dtype=np.float32)

        epochs_out, means_out = [], []
        for e in epochs_unique:
            epoch_mask = epochs == e
            epoch_values = values[epoch_mask]
            epoch_mean = np.mean(epoch_values)

            epochs_out.append(e.item())
            means_out.append(epoch_mean.item())
            pass
        return epochs_out, means_out
    
    def plot(self, show_cv:bool = False):
        for key in self.data:
            plt.figure(figsize=(8, 6))
            l = 0

            for i, split in enumerate(self.data[key]):
                x_iter = self.data[key][split]['iteration']
                y_iter = self.data[key][split]['value']
                x_epoch, y_epoch = self._get_epoch_mean_values(key, split)
                l = max(l, max(x_iter))
                plt.plot(x_iter, y_iter, color=PLOT_COLORS[i], alpha=0.4)
                plt.plot(x_epoch, y_epoch, color=PLOT_COLORS[i], label=split, alpha=1.0)

            plt.title(key)
            plt.xlabel('Epoch')
            plt.ylabel(key)
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
            plt.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgray')
            plt.xlim(0, max(0.01, l))
            plt.legend()

            plt.savefig(self.plot_dir.joinpath(f"{key}.png"))

            if show_cv:
                img = _plt2img()
                cv.imshow("Loss over Epochs", img)
                cv.waitKey(1)

            plt.close()