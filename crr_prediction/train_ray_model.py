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
    **kwargs: Dict
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
    **kwargs: Dict,
        Selected hyper-parameters.
    """
    model = meta_model.build(**kwargs)
    model.compile(
        optimizer='nadam',
        loss="binary_crossentropy",
        metrics=get_standard_binary_metrics()
    )
    model.fit(
        train,
        validation_data=validation,
        epochs=max_epochs,
        verbose=False,
        callbacks=[
            TuneReportCallback(metrics=[
                "{}{}".format(sub, metric)
                for metric in model.metrics_names
                for sub in ("", "val_")
            ], ),
            EarlyStopping(
                monitor="loss",
                min_delta=min_delta,
                patience=patience
            ),
            TerminateOnNaN()
        ]
    )
