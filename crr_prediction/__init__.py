"""Module providing training methods for CRR sequences."""
from .train_meta_models import train_meta_models
from .train_fixed_models import train_fixed_models

__all__ = [
    "train_meta_models",
    "train_fixed_models"
]
