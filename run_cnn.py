"""Training of cnn models."""
import ray
from crr_prediction import train_meta_models
from crr_prediction.meta_models import build_cnn_meta_model
from crr_prediction.sequences import build_cnn_sequences

if __name__ == "__main__":
    ray.init()
    train_meta_models(
        build_cnn_sequences,
        build_cnn_meta_model
    ).to_csv("cnn.csv", index=False)
