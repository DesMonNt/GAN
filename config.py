from dataclasses import dataclass

import torch


@dataclass
class Config:
    z_dim: int = 100
    batch_size: int = 64
    image_size: int = 64
    lr: float = 5e-5
    epochs: int = 100
    n_critic: int = 5
    clip_value: float = 0.01
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
