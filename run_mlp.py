"""Training of MLP models."""
from crr_prediction import train_meta_models
from crr_prediction.meta_models import build_mlp_meta_model, build_cnn_meta_model
from crr_prediction.sequences import build_cnn_sequences, build_mlp_sequences

if __name__ == "__main__":
    train_meta_models(
        build_mlp_sequences,
        build_mlp_meta_model
    ).to_csv("mlp.csv", index=False)
