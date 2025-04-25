#!/usr/bin/env python3
"""
Distributed Training Module with DeepSpeed support
"""

import os
import time
import logging
from datetime import timedelta
import torch
import deepspeed
from torch.utils.data.distributed import DistributedSampler

class DeepSpeedTrainer:
    def __init__(self, config):
        self.config = config
        self.rank, self.world_size, self.local_rank = self._init_distributed()
        self.device = self._setup_device()
        self.model = self._init_model()
        self._configure_deepspeed()

    def _init_distributed(self):
        """Safe distributed initialization handling both MPI and Slurm"""
        rank = int(os.environ.get('PMI_RANK', os.environ.get('SLURM_PROCID', 0)))
        world_size = int(os.environ.get('PMI_SIZE', os.environ.get('SLURM_NTASKS', 1)))
        local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', os.environ.get('SLURM_LOCALID', 0)))
        
        return rank, world_size, local_rank

    def _setup_device(self):
        """CUDA device initialization with safety checks"""
        torch.cuda.set_device(self.local_rank)
        device = torch.device(f'cuda:{self.local_rank}')
        
        # Verify CUDA availability
        try:
            torch.cuda.empty_cache()
            _ = torch.cuda.memory_allocated(device)
            return device
        except RuntimeError:
            raise RuntimeError(f'CUDA device {self.local_rank} not available')

    def _init_model(self):
        """Model initialization"""
        from model import FailurePredictor
        return FailurePredictor(self.config).to(self.device)

    def _configure_deepspeed(self):
        """Initialize DeepSpeed engine"""
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        
        self.model, self.optimizer, _, _ = deepspeed.initialize(
            args=self.config,
            model=self.model,
            model_parameters=parameters,
            config_params=self.config.get('deepspeed_config', {})
        )

    def train_epoch(self, dataloader, epoch):
        """Single epoch training with DeepSpeed"""
        self.model.train()
        sampler = dataloader.sampler
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        total_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True).float()

            # Forward pass
            outputs = self.model(inputs)
            loss = torch.nn.functional.binary_cross_entropy(outputs, targets)

            # Backward pass with DeepSpeed
            self.model.backward(loss)
            self.model.step()

            # Synchronize and log
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            total_loss += loss.item() / self.world_size

            if batch_idx % 50 == 0 and self.rank == 0:
                logging.info(f'Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f}')

        return total_loss / len(dataloader)

    @torch.no_grad()
    def validate(self, dataloader):
        """Distributed validation with metric synchronization"""
        self.model.eval()
        all_preds, all_targets = [], []

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            
            # Gather across processes
            preds = [torch.zeros_like(outputs) for _ in range(self.world_size)]
            targs = [torch.zeros_like(targets) for _ in range(self.world_size)]
            dist.all_gather(preds, outputs)
            dist.all_gather(targs, targets)
            
            all_preds.extend(preds)
            all_targets.extend(targs)

        return torch.cat(all_preds), torch.cat(all_targets)

    def run(self, train_loader, val_loader=None):
        """Main training loop with checkpointing"""
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(train_loader, epoch)
            
            if val_loader and (epoch % self.config['eval_every'] == 0):
                preds, targets = self.validate(val_loader)
                if self.rank == 0:
                    from utils import calculate_metrics
                    metrics = calculate_metrics(targets.cpu(), preds.cpu())
                    logging.info(f'Epoch {epoch} Validation - Loss: {train_loss:.4f} | AUC: {metrics["roc_auc"]:.4f}')

            # Save checkpoint on master
            if self.rank == 0 and (epoch % self.config['save_every'] == 0 or epoch == self.config['epochs'] - 1):
                self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch):
        """Save checkpoint using DeepSpeed's method"""
        client_state = {
            'epoch': epoch,
            'config': self.config
        }
        self.model.save_checkpoint(
            save_dir=os.path.join(self.config['output_dir'], 'checkpoints'),
            tag=f'epoch_{epoch}',
            client_state=client_state
        )

def train_main(config):
    """Main training entry point"""
    from data import create_dataloader
    from utils import setup_logging
    
    # Initialize logging
    logger = setup_logging(config['output_dir'], rank=config.get('local_rank', 0))
    
    # Initialize trainer
    trainer = DeepSpeedTrainer(config)
    
    # Create data loaders
    train_loader = create_dataloader(
        batch_size=config['batch_size'],
        split='train',
        rank=trainer.rank,
        world_size=trainer.world_size
    )
    
    val_loader = create_dataloader(
        batch_size=config['batch_size'],
        split='val',
        rank=trainer.rank,
        world_size=trainer.world_size
    ) if config.get('eval_every', 0) > 0 else None
    
    # Run training
    trainer.run(train_loader, val_loader)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    
    # Load config
    import json
    with open(args.config) as f:
        config = json.load(f)
    config['output_dir'] = args.output_dir
    config['local_rank'] = args.local_rank
    
    train_main(config)
