"""Module building sequences for cnn model."""
from typing import Tuple, Dict
import pandas as pd
from keras_mixed_sequence import MixedSequence
from .build_cnn_sequences import get_cnn_training_sequence
from ucsc_genomes_downloader import Genome


def build_cnn_sequences(
    train_y: pd.DataFrame,
    test_y: pd.DataFrame,
    subtrain_y: pd.DataFrame,
    valid_y: pd.DataFrame,
    genome: Genome,
    batch_size: int,
    random_state: int,
    **kwargs: Dict
) -> Tuple[MixedSequence]:
    """Return quadruple with train, test, subtrain and validation sequences.

    Parameters
    ----------------------
    train_y: pd.DataFrame,
        Training label dataframe.
    test_y: pd.DataFrame,
        Test label dataframe.
    subtrain_y: pd.DataFrame,
        Subtraining label dataframe.
    valid_y: pd.DataFrame,
        Validation label dataframe.
    genome: Genome,
        Genome object from where to extract the sequences.
    batch_size: int,
        Batch size for the training sequences.
    random_state: int,
        Random state.
    **kwargs: Dict,
        Additional kwargs not used for this method.

    Returns
    ----------------------
    Quadruple of training sequences.
    """
    return (
        get_cnn_training_sequence(
            genome,
            train_y,
            batch_size=batch_size,
            random_state=random_state
        ),
        get_cnn_training_sequence(
            genome,
            test_y,
            batch_size=batch_size,
            random_state=random_state
        ),
        get_cnn_training_sequence(
            genome,
            subtrain_y,
            batch_size=batch_size,
            random_state=random_state
        ),
        get_cnn_training_sequence(
            genome,
            valid_y,
            batch_size=batch_size,
            random_state=random_state
        )
    )
