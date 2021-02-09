"""Training of MLP models."""
import ray
from crr_prediction import train_fixed_models
from crr_prediction.baseline_models import fixed_ffnn
from crr_prediction.sequences import build_cnn_sequences, build_mlp_sequences
from meta_models.utils import patch_global_checkpoints_interval

if __name__ == "__main__":
    train_fixed_models(
        build_mlp_sequences,
        fixed_ffnn,
        model="FixedMLP",
        n_splits=20
    ).to_csv("fixed_mlp.csv", index=False)
