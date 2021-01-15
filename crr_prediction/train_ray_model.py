"""Module providing training method for meta models."""
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from keras_mixed_sequence import MixedSequence
from meta_models.meta_models import MetaModel
from extra_keras_metrics import get_standard_binary_metrics
from ray.tune.integration.keras import TuneReportCallback
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from .utils import enable_subgpu_training


def train_ray_model(
    config: Dict,
    train: Tuple[np.ndarray],
    validation:  Tuple[np.ndarray],
    meta_model: MetaModel,
    max_epochs: int,
    batch_size: int,
    patience: int,
    min_delta: float,
    verbose: bool = False,
    enable_ray_callback: bool = True,
    subgpu_training: bool = False
) -> pd.DataFrame:
    """Train the ray model.

    Parameters
    ---------------------
    config: Dict,
        Selected hyper-parameters.
    train: MixedSequence,
        Training sequence.
    validation: MixedSequence,
        Validation sequence.
    meta_model: MetaModel,
        MetaModel to fit.
    max_epochs: int,
        Maximum number of training epochs.
    batch_size: int,
        Batch size for the training process.
    patience: int,
        Patience for early stopping.
    min_delta: float,
        Minimum delta for early stopping.
    verbose: bool = False,
        Wether to show loading bars.
    enable_ray_callback: bool = True,
        Wether to enable the ray callback.
    subgpu_training: bool = False,
        Wether to enable subgpu training.

    Returns
    ----------------------
    Dataframe containing training history.
    """
    import silence_tensorflow.auto
    if subgpu_training:
        enable_subgpu_training()
    # Build the selected model from the meta model
    model = meta_model.build(**config)
    # Compile it
    model.compile(
        optimizer='nadam',
        loss="binary_crossentropy",
        # We add all the most common binary metrics
        metrics=get_standard_binary_metrics()
    )
    # Fitting the model
    return pd.DataFrame(model.fit(
        *train,
        validation_data=validation,
        epochs=max_epochs,
        batch_size=batch_size,
        verbose=verbose,
        shuffle=True,
        callbacks=[
            # We report the training performance at the end of each epoch
            *((TuneReportCallback(),) if enable_ray_callback else ()),
            # We kill the process when the training reaches a plateau
            EarlyStopping(
                monitor="loss",
                min_delta=min_delta,
                patience=patience,
                restore_best_weights=True
            ),
            # And if something very wrong happens and a NaN appears,
            # we terminate the execution.
            TerminateOnNaN()
        ]
    ).history)
