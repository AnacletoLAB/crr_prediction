"""Module with genomic training sequence for the CNN."""
import pandas as pd
from ucsc_genomes_downloader import Genome
from keras_bed_sequence import BedSequence
from keras_mixed_sequence import MixedSequence, VectorSequence


def get_cnn_training_sequence(
    genome: Genome,
    y: pd.DataFrame,
    batch_size: int,
    seed: int
) -> MixedSequence:
    """Return training sequence for CNN.

    Parameters
    --------------------
    genome: Genome,
        Genome to extract the training sequences from.
    y: pd.DataFrame,
        Labels.
    batch_size: int,
        Size of the batches.
    seed: int,
        Random seed to reproduce the generated sequences.

    Returns
    ---------------------
    CNN genomic sequence.
    """
    return MixedSequence(
        BedSequence(
            genome,
            bed=y.reset_index()[y.index.names],
            batch_size=batch_size,
            seed=seed
        ),
        VectorSequence(
            y.values,
            batch_size=batch_size,
            seed=seed,
        )
    )
