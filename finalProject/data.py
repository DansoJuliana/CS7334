import pandas as pd
import subprocess
from io import StringIO
from typing import Optional, Dict, Iterator
import torch
from torch.utils.data import Dataset, DataLoader, distributed
from boto3 import UNSIGNED

class SupercloudDataLoader:
    """Handles streaming and batching from S3, with AWS CLI or boto3."""
    def __init__(self, use_aws_cli: bool = True):
        self.bucket = "mit-supercloud-dataset"
        self.prefix = "datacenter-challenge/202201/"
        self.use_aws_cli = use_aws_cli  # Fallback to boto3 if False
        if not use_aws_cli:
            import boto3
            self.s3 = boto3.client('s3', config=boto3.session.Config(signature_version=UNSIGNED))

    def _stream_csv(self, key: str) -> Optional[pd.DataFrame]:
        """Stream CSV from S3 using AWS CLI or boto3."""
        if self.use_aws_cli:
            cmd = f"aws s3 cp s3://{self.bucket}/{self.prefix}{key} - --no-sign-request"
            try:
                result = subprocess.run(cmd, shell=True, check=True, 
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                return pd.read_csv(StringIO(result.stdout))
            except subprocess.CalledProcessError as e:
                print(f"AWS CLI Error: {e.stderr}")
                return None
        else:
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=f"{self.prefix}{key}")
                return pd.read_csv(obj['Body'])
            except Exception as e:
                print(f"Boto3 Error: {str(e)}")
                return None

    def get_labeled_data(self, include_success: bool = False) -> Optional[pd.DataFrame]:
        """Merge critical files into a labeled dataset."""
        node_data = self._stream_csv("node-data.csv")
        slurm_log = self._stream_csv("slurm-log.csv")
        job_ids = self._stream_csv("labelled_jobids.csv")
        if None in [node_data, slurm_log, job_ids]:
            return None
        df = pd.merge(node_data, slurm_log, on="job_id")
        df = pd.merge(df, job_ids, on="job_id")
        return df if include_success else df[df["status"] == "fail"]

    def stream_time_series(self, job_id: str, metric_type: str = "cpu") -> Optional[pd.DataFrame]:
        """Stream CPU/GPU time-series for a job."""
        return self._stream_csv(f"{metric_type}/{job_id}.csv")

class SupercloudDataset(Dataset):
    """PyTorch Dataset for distributed training integration."""
    def __init__(self, loader: SupercloudDataLoader, include_success: bool = False):
        self.data = loader.get_labeled_data(include_success=include_success)
        if self.data is None:
            raise ValueError("Failed to load labeled data")
        
        # Convert status to binary labels (1 for failure, 0 for success)
        self.labels = torch.tensor(self.data["status"].eq("fail").astype(int).values, dtype=torch.long)
        
        # Convert features to tensor
        feature_cols = [col for col in self.data.columns if col not in ["job_id", "status"]]
        self.features = torch.tensor(self.data[feature_cols].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def create_dataloader(batch_size: int = 64, 
                     use_aws_cli: bool = True,
                     split: str = 'train',
                     rank: int = 0,
                     world_size: int = 1,
                     include_success: bool = False) -> DataLoader:
    """Create a DataLoader ready for distributed training."""
    loader = SupercloudDataLoader(use_aws_cli=use_aws_cli)
    dataset = SupercloudDataset(loader, include_success=include_success)
    
    sampler = distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=(split == 'train')
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
