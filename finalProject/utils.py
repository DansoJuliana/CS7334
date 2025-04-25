import os
import time
import socket
import logging
import json
import numpy as np
import torch
import torch.nn as nn
from datetime import timedelta
import torch.distributed as dist
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, confusion_matrix, roc_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging(output_dir: str, rank: int = 0) -> logging.Logger:
    """Initialize distributed-safe logging"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, f'training_{time.strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    return logger

def init_distributed_mode() -> Tuple[int, int, int]:
    """Safe distributed initialization handling MPI/Slurm"""
    rank = int(os.environ.get('PMI_RANK', os.environ.get('SLURM_PROCID', 0)))
    world_size = int(os.environ.get('PMI_SIZE', os.environ.get('SLURM_NTASKS', 1)))
    local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', os.environ.get('SLURM_LOCALID', 0)))
    
    # Port conflict prevention
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = str(find_free_port())
    
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}',
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=5)
        )
    
    return rank, world_size, local_rank

def find_free_port() -> int:
    """Find an available network port"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def setup_device(local_rank: int) -> torch.device:
    """Initialize CUDA device with validation"""
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # Verify CUDA availability
    try:
        torch.cuda.empty_cache()
        _ = torch.cuda.memory_allocated(device)
        return device
    except RuntimeError:
        raise RuntimeError(f'CUDA device {local_rank} not available')

def sync_metrics(metrics: Dict[str, Any], world_size: int) -> Dict[str, Any]:
    """Average metrics across all processes"""
    if world_size == 1:
        return metrics
    
    metrics_tensor = torch.tensor(
        [v for v in metrics.values() if isinstance(v, (int, float))],
        device=torch.cuda.current_device()
    )
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
    metrics_tensor /= world_size
    
    synced = {}
    for i, k in enumerate([k for k in metrics.keys() if isinstance(metrics[k], (int, float))]):
        synced[k] = metrics_tensor[i].item()
    
    # Handle non-numeric metrics (e.g., confusion matrices)
    for k in metrics.keys():
        if not isinstance(metrics[k], (int, float)):
            synced[k] = metrics[k]
    
    return synced

def calculate_metrics(targets: torch.Tensor, preds: torch.Tensor) -> Dict[str, float]:
    """Compute binary classification metrics"""
    targets_np = targets.cpu().numpy()
    preds_np = preds.cpu().numpy()
    
    return {
        'roc_auc': roc_auc_score(targets_np, preds_np),
        'accuracy': accuracy_score(targets_np > 0.5, preds_np > 0.5),
        'pr_auc': average_precision_score(targets_np, preds_np)
    }

def plot_metrics(metrics: Dict[str, Any], output_dir: str, prefix: str = "") -> None:
    """Generate and save metric plots (only on rank 0)"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(metrics['targets'], metrics['preds'])
    plt.plot(fpr, tpr, label=f'ROC AUC = {metrics["roc_auc"]:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{prefix}roc_curve.png'))
    plt.close()
    
    # Confusion Matrix
    plt.figure(figsize=(6, 6))
    cm = confusion_matrix(metrics['targets'] > 0.5, metrics['preds'] > 0.5)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, f'{prefix}confusion_matrix.png'))
    plt.close()

def save_checkpoint(state: Dict[str, Any], filename: str, is_best: bool = False) -> None:
    """Atomic checkpoint saving"""
    temp_file = f'{filename}.tmp'
    torch.save(state, temp_file)
    os.replace(temp_file, filename)
    
    if is_best:
        best_file = os.path.join(os.path.dirname(filename), 'model_best.pth')
        os.link(filename, best_file)

def load_checkpoint(filename: str, model: nn.Module, optimizer=None, scaler=None) -> int:
    """Load checkpoint with validation"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint {filename} not found")
    
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    
    if optimizer is not None and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    if scaler is not None and 'scaler_state' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state'])
    
    return checkpoint.get('epoch', 0)

def log_distributed_env(logger: logging.Logger) -> None:
    """Log distributed environment details"""
    env_vars = {
        'Distributed': {
            'Initialized': dist.is_initialized(),
            'Backend': dist.get_backend() if dist.is_initialized() else None,
            'World Size': dist.get_world_size() if dist.is_initialized() else None,
            'Rank': dist.get_rank() if dist.is_initialized() else None
        },
        'Environment': {
            'MASTER_ADDR': os.environ.get('MASTER_ADDR'),
            'MASTER_PORT': os.environ.get('MASTER_PORT'),
            'LOCAL_RANK': os.environ.get('LOCAL_RANK'),
            'SLURM_PROCID': os.environ.get('SLURM_PROCID'),
            'PMI_RANK': os.environ.get('PMI_RANK')
        },
        'CUDA': {
            'Available': torch.cuda.is_available(),
            'Device Count': torch.cuda.device_count(),
            'Current Device': torch.cuda.current_device() if torch.cuda.is_available() else None
        }
    }
    
    logger.info("Environment Summary:\n" + json.dumps(env_vars, indent=2))

def barrier() -> None:
    """Distributed barrier with timeout"""
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()

def cleanup_distributed() -> None:
    """Clean up distributed process group if initialized"""
    if dist.is_initialized():
        dist.destroy_process_group()
