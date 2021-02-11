"""Training of cnn models."""
import sys
from crr_prediction import train_fixed_models
from crr_prediction.baseline_models import fixed_cnn
from crr_prediction.sequences import build_cnn_sequences
from meta_models.utils import patch_global_checkpoints_interval
from ucsc_genomes_downloader import Genome

if __name__ == "__main__":
    if len(sys.argv) > 1:
        only_cell_line = sys.argv[1]
        path = f"{only_cell_line}_fixed_cnn.csv"
    else:
        only_cell_line = None
        path = "fixed_cnn.csv"
    train_fixed_models(
        build_cnn_sequences,
        fixed_cnn,
        model="FixedCNN",
        only_cell_line=only_cell_line,
        n_splits=10,
        genome=Genome("hg38")
    ).to_csv(path, index=False)
