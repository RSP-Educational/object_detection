import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
            _mavg[i] = np.mean(_data[:i])
        else:
            _mavg[i] = np.mean(_data[i-window:i])
    return _mavg

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
        self.loss_plot_file = self.plot_dir.joinpath('losses.png')

        self.losses = {
            'train': [],
            'val': []
        }

    def load(self, model:nn.Module, optimizer:optim.Optimizer):
        if self.checkpoint_file.exists():
            checkpoint = torch.load(self.checkpoint_file)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.losses = checkpoint['losses']
            best_val_loss = min(self.get_values('val'))
            print(f"✓ Loaded checkpoint from epoch {self.epoch()-1} with val loss {best_val_loss:.4f}")

    def save(self, model:nn.Module, optimizer:optim.Optimizer):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': self.losses
        }
        torch.save(checkpoint, self.checkpoint_file)

    def append(self, split:str, value:float):
        self.losses[split].append(value)

    def epoch(self):
        return len(self.losses['train']) + 1

    def get_values(self, split:str):
        return self.losses[split]
    
    def plot(self):
        plt.figure(figsize=(8, 6))

        l = 0
        for i, (key, value) in enumerate(self.losses.items()):
            _mavg = _moving_average(value, self.mavg_epochs)
            l = max(l, len(value))
            plt.plot(value, color=PLOT_COLORS[i])
            plt.plot(_mavg, color=PLOT_COLORS[i], label=key, alpha=1.0)

        plt.title('Loss over Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.minorticks_on()
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgray')
        plt.xlim(0, l-1)
        plt.legend()

        plt.savefig(self.loss_plot_file)
        plt.close()