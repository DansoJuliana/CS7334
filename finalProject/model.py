import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math

class FeatureEmbedding(nn.Module):
    """Embedding layer for categorical features"""
    def __init__(self, num_features: int, embedding_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(num_features, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

class TemporalAttention(nn.Module):
    """Attention mechanism for time-series features"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = math.sqrt(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention = torch.softmax(Q @ K.transpose(-2, -1) / self.scale, dim=-1)
        return attention @ V

class FailurePredictor(nn.Module):
    """
    Neural network for predicting node failures from system metrics.
    Architecture:
    1. Feature embedding for categorical variables
    2. Temporal attention for time-series metrics
    3. Dense layers for final prediction
    """
    
    def __init__(self, 
                 config: Optional[Dict] = None,
                 num_numerical_features: int = 10,
                 num_categorical_features: int = 3,
                 categorical_embed_dim: int = 32):
        super().__init__()
        
        # Default configuration
        self.config = config or {
            "hidden_dims": [256, 128, 64],
            "dropout_rate": 0.2,
            "use_attention": True,
            "temporal_window": 5
        }
        
        # Feature embedding
        self.categorical_embed = FeatureEmbedding(
            num_categorical_features, 
            categorical_embed_dim
        )
        
        # Temporal processing
        self.temporal_attention = TemporalAttention(
            num_numerical_features + categorical_embed_dim
        ) if self.config["use_attention"] else None
        
        # Main network
        input_dim = (num_numerical_features + categorical_embed_dim) * \
                   (self.config["temporal_window"] if self.config["use_attention"] else 1)
        
        layers = []
        for i, hidden_dim in enumerate(self.config["hidden_dims"]):
            layers.append(nn.Linear(
                input_dim if i == 0 else self.config["hidden_dims"][i-1],
                hidden_dim
            ))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.config["dropout_rate"]))
            input_dim = hidden_dim
            
        self.dense_layers = nn.Sequential(*layers)
        
        # Output
        self.output = nn.Linear(self.config["hidden_dims"][-1], 1)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, 
               numerical_features: torch.Tensor,
               categorical_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            numerical_features: (batch_size, seq_len, num_numerical_features)
            categorical_features: (batch_size, num_categorical_features)
        Returns:
            Prediction scores (batch_size, 1)
        """
        # Embed categorical features
        if categorical_features is not None:
            cat_embedded = self.categorical_embed(categorical_features.long())
            # Repeat embeddings across temporal dimension
            cat_embedded = cat_embedded.unsqueeze(1).repeat(
                1, numerical_features.size(1), 1)
            features = torch.cat([numerical_features, cat_embedded], dim=-1)
        else:
            features = numerical_features
        
        # Temporal processing
        if self.temporal_attention is not None:
            features = self.temporal_attention(features)
            features = features.flatten(start_dim=1)  # (batch_size, seq_len * feature_dim)
        else:
            features = features[:, -1, :]  # Just use most recent time step
        
        # Dense layers
        features = self.dense_layers(features)
        
        # Output
        return torch.sigmoid(self.output(features))

    def get_attention_weights(self, numerical_features: torch.Tensor) -> torch.Tensor:
        """Extract attention weights for interpretability"""
        if self.temporal_attention is None:
            raise ValueError("Model not configured with attention")
        
        Q = self.temporal_attention.query(numerical_features)
        K = self.temporal_attention.key(numerical_features)
        scale = math.sqrt(numerical_features.size(-1))
        
        return torch.softmax(Q @ K.transpose(-2, -1) / scale, dim=-1)

class ParallelizedWrapper(nn.Module):
    """
    Wrapper to handle model parallelization strategies.
    Actual parallelization is applied in train.py, this just provides the interface.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
