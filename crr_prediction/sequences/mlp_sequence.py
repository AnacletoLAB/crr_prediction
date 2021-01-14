"""Module with epigenomic training sequence for the MLP."""
import pandas as pd
import numpy as np
from keras_mixed_sequence import MixedSequence, VectorSequence


def get_mlp_training_sequence(
    X: np.ndarray,
    y: pd.DataFrame,
    batch_size: int,
    random_state: int
) -> MixedSequence:
    """Return training sequence for MLP.

    Parameters
    --------------------
    X: np.ndarray,
        Epigenomic data.
    y: pd.DataFrame,
        Labels.
    batch_size: int,
        Size of the batches.
    random_state: int,
        Random random_state to reproduce the generated sequences.

    Returns
    ---------------------
    CNN genomic sequence.
    """
    return MixedSequence(
        VectorSequence(
            X,
            batch_size=batch_size,
            random_state=random_state
        ),
        VectorSequence(
            y.values,
            batch_size=batch_size,
            random_state=random_state,
        )
    )
