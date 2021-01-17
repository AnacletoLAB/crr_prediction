"""Module building sequences for mlp model."""
from typing import Tuple, Dict
import pandas as pd
from keras_mixed_sequence import MixedSequence
from epigenomic_dataset.utils import normalize_epigenomic_data
from .mlp_sequence import get_mlp_training_sequence


def build_mlp_sequences(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    subtrain_x: pd.DataFrame,
    valid_x: pd.DataFrame,
    train_y: pd.DataFrame,
    test_y: pd.DataFrame,
    subtrain_y: pd.DataFrame,
    valid_y: pd.DataFrame,
    batch_size: int,
    random_state: int,
    **kwargs: Dict
) -> Tuple[MixedSequence]:
    """Return quadruple with train, test, subtrain and validation sequences.

    Parameters
    ----------------------
    train_x: pd.DataFrame,
        Training input dataframe.
    test_x: pd.DataFrame,
        Test input dataframe.
    subtrain_x: pd.DataFrame,
        Subtraining input dataframe.
    valid_x: pd.DataFrame,
        Validation input dataframe.
    train_y: pd.DataFrame,
        Training label dataframe.
    test_y: pd.DataFrame,
        Test label dataframe.
    subtrain_y: pd.DataFrame,
        Subtraining label dataframe.
    valid_y: pd.DataFrame,
        Validation label dataframe.
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
    normalized_train_x, normalized_test_x = normalize_epigenomic_data(
        train_x,
        test_x
    )
    normalized_subtrain_x, normalized_valid_x = normalize_epigenomic_data(
        subtrain_x,
        valid_x
    )
    return (
        get_mlp_training_sequence(
            normalized_train_x,
            train_y,
            batch_size=batch_size,
            random_state=random_state
        ),
        get_mlp_training_sequence(
            normalized_test_x,
            test_y,
            batch_size=batch_size,
            random_state=random_state
        ),
        get_mlp_training_sequence(
            normalized_subtrain_x,
            subtrain_y,
            batch_size=batch_size,
            random_state=random_state
        ),
        get_mlp_training_sequence(
            normalized_valid_x,
            valid_y,
            batch_size=batch_size,
            random_state=random_state
        )
    )
