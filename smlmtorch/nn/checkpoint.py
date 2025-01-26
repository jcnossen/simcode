"""
GPT4 prompt
write a class that saves a pytorch model and optimizer state to a checkpoint, and has a function to load the latest checkpoint
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

class CheckpointManager:
    def __init__(self, model, optimizer, lr_scheduler, save_dir, max_saves=5):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_dir = os.path.abspath(save_dir)
        self.max_saves = max_saves

        print("CheckpointManager initialized: ", self.save_dir)


    def save(self, epoch):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict()
        }
        checkpoint_idx = (epoch - 1) % self.max_saves
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_{checkpoint_idx}.pt')
        self._remove_old_checkpoint(checkpoint_path)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def _remove_old_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"Removed old checkpoint: {checkpoint_path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        lr = self.optimizer.param_groups[0]['lr']
        print(f"Loaded checkpoint: {path}. Epoch: {checkpoint['epoch']}. Learning rate: {lr:.1e}")
        return checkpoint['epoch']

    def load_latest(self):
        latest_checkpoint = self.find_latest_checkpoint()
        if latest_checkpoint is not None:
            return self.load(latest_checkpoint)
        else:
            print("No checkpoint found.")
            return None

    def find_latest_checkpoint(self):
        checkpoints = [f for f in os.listdir(self.save_dir) if f.endswith('.pt')]
        if not checkpoints:
            return None
        latest_checkpoint = max(checkpoints, key=lambda f: int(f.split('_')[1].split('.')[0]))
        return os.path.join(self.save_dir, latest_checkpoint)


if __name__ == '__main__':

    # Example model and optimizer
    model = nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize CheckpointManager
    save_dir = 'checkpoints'
    max_saves = 3
    checkpoint_manager = CheckpointManager(model, optimizer, save_dir, max_saves)

    # Save checkpoints and test the max_saves functionality
    for epoch in range(1, 6):
        checkpoint_manager.save(epoch)

    # Load the latest checkpoint
    latest_epoch = checkpoint_manager.load_latest()
    print(f"Latest epoch: {latest_epoch}")
