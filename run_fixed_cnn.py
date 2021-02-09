"""Training of cnn models."""
import ray
from crr_prediction import train_fixed_models
from crr_prediction.baseline_models import fixed_cnn
from crr_prediction.sequences import build_cnn_sequences
from meta_models.utils import patch_global_checkpoints_interval
from ucsc_genomes_downloader import Genome

if __name__ == "__main__":
    train_fixed_models(
        build_cnn_sequences,
        fixed_cnn,
        model="FixedCNN",
        n_splits=10,
        genome_assembly="hg38"
    ).to_csv("fixed_cnn.csv", index=False)
