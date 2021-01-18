"""Training of MLP models."""
import ray
from crr_prediction import train_meta_models
from crr_prediction.meta_models import build_mlp_meta_model, build_cnn_meta_model
from crr_prediction.sequences import build_cnn_sequences, build_mlp_sequences
from meta_models.utils import patch_global_checkpoints_interval

if __name__ == "__main__":
    patch_global_checkpoints_interval()
    ray.init()
    train_meta_models(
        build_mlp_sequences,
        build_mlp_meta_model,
        model="MLP"
    ).to_csv("mlp.csv", index=False)
