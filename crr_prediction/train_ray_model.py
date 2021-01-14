"""Module providing training method for meta models."""
from typing import Dict
from keras_mixed_sequence import MixedSequence
from meta_models.meta_models import MetaModel
from extra_keras_metrics import get_standard_binary_metrics
from ray.tune.integration.keras import TuneReportCallback
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN


def train_ray_model(
    train: MixedSequence,
    validation: MixedSequence,
    meta_model: MetaModel,
    max_epochs: int,
    patience: int,
    min_delta: float,
    config: Dict
):
    """Train the ray model.

    Parameters
    ---------------------
    train: MixedSequence,
        Training sequence.
    validation: MixedSequence,
        Validation sequence.
    meta_model: MetaModel,
        MetaModel to fit.
    max_epochs: int,
        Maximum number of training epochs.
    patience: int,
        Patience for early stopping.
    min_delta: float,
        Minimum delta for early stopping.
    config: Dict,
        Selected hyper-parameters.
    """
    import silence_tensorflow.auto
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
    model.fit(
        train,
        validation_data=validation,
        epochs=max_epochs,
        verbose=False,
        callbacks=[
            # We report the training performance at the end of each epoch
            TuneReportCallback(),
            # We kill the process when the training reaches a plateau
            EarlyStopping(
                monitor="loss",
                min_delta=min_delta,
                patience=patience
            ),
            # And if something very wrong happens and a NaN appears,
            # we terminate the execution.
            TerminateOnNaN()
        ]
    )
