"""Module with genomic training sequence for the CNN."""
import pandas as pd
from ucsc_genomes_downloader import Genome
from keras_bed_sequence import BedSequence
from keras_mixed_sequence import MixedSequence, VectorSequence


def get_cnn_training_sequence(
    genome: Genome,
    y: pd.DataFrame,
    batch_size: int,
    random_state: int
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
    random_state: int,
        Random state to reproduce the generated sequences.

    Returns
    ---------------------
    CNN genomic sequence.
    """
    return MixedSequence(
        BedSequence(
            genome,
            bed=y.reset_index()[y.index.names],
            batch_size=batch_size,
            random_state=random_state
        ),
        VectorSequence(
            y.values,
            batch_size=batch_size,
            random_state=random_state,
        )
    )
