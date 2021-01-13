"""Module with epigenomic training sequence for the MLP."""
import pandas as pd
from keras_mixed_sequence import MixedSequence, VectorSequence


def get_mlp_training_sequence(
    X: pd.DataFrame,
    y: pd.DataFrame,
    batch_size: int,
    seed: int
) -> MixedSequence:
    """Return training sequence for MLP.

    Parameters
    --------------------
    X: pd.DataFrame,
        Epigenomic data.
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
        VectorSequence(
            X.values,
            batch_size=batch_size,
            seed=seed
        ),
        VectorSequence(
            y.values,
            batch_size=batch_size,
            seed=seed,
        )
    )
