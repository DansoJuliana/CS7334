#!/usr/bin/env python3
"""
Main execution script with DeepSpeed support
"""

import os
import argparse
import json
import time
from datetime import datetime

import torch
import deepspeed
from utils import (
    setup_logging,
    init_distributed_mode,
    cleanup_distributed,
    plot_metrics,
    log_distributed_env
)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="DeepSpeed Training Runner")
    
    # Training configuration
    parser.add_argument("--config", type=str, default="configs/base.json",
                      help="Path to training config file")
    parser.add_argument("--output-dir", type=str, default="results",
                      help="Directory to save outputs")
    parser.add_argument("--run-id", type=str, default=None,
                      help="Unique identifier for this run")
    parser.add_argument("--resume", type=str, default=None,
                      help="Checkpoint path to resume from")
    
    # Distributed training
    parser.add_argument("--local-rank", type=int, default=0,
                      help="Local rank for distributed training")
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from JSON file"""
    with open(config_path) as f:
        config = json.load(f)
    
    # Set default values for required fields
    defaults = {
        "batch_size": 64,
        "epochs": 10,
        "lr": 1e-3,
        "eval_every": 1,
        "save_every": 1,
        "seed": 42
    }
    
    for key, value in defaults.items():
        config.setdefault(key, value)
    
    return config

def setup_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    """Initialize experiment environment and configuration"""
    # Initialize distributed mode
    rank, world_size, local_rank = init_distributed_mode()
    
    # Load configuration
    config = load_config(args.config)
    config.update({
        "world_size": world_size,
        "local_rank": local_rank,
        "output_dir": args.output_dir
    })
    
    # Set random seeds for reproducibility
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    
    # Set run ID if not provided
    if args.run_id is None and rank == 0:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if world_size > 1:
        torch.distributed.barrier()
    
    return config

def main():
    args = parse_args()
    
    # Initialize experiment
    config = setup_experiment(args)
    rank = config["local_rank"]
    
    # Setup logging
    logger = setup_logging(os.path.join(config["output_dir"], "logs"), rank=rank)
    if rank == 0:
        log_distributed_env(logger)
    
    try:
        # Initialize and run training
        from train import train_main
        results = train_main(config)
        
        # Save results and plots (master process only)
        if results and rank == 0:
            with open(os.path.join(config["output_dir"], f"results_{args.run_id}.json"), "w") as f:
                json.dump(results, f, indent=2)
            
            plot_metrics(
                metrics=results["metrics"],
                output_dir=os.path.join(config["output_dir"], "plots"),
                prefix=f"{args.run_id}_"
            )
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise
    
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
