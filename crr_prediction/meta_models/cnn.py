"""Module providing meta-cnn model."""
from meta_models.meta_models import CNN2DMetaModel


def build_cnn_meta_model(window_size: int) -> CNN2DMetaModel:
    """Return CNN meta model.

    Parameters
    -------------------------
    window_size: int,
        Training window size.

    Returns
    -------------------------
    CNN1D model.
    """
    return CNN2DMetaModel(
        blocks=3,
        input_shape=(window_size, 4),
        target_shape=(window_size, 4, 1),
        meta_layer_kwargs=dict(
            batch_normalization=False,
            dropout=False,
            residual=True,
            max_filters=128,
            min_x_strides=2,
            l1_regularization=1e-4,
            l2_regularization=1e-4,
            activity_regularizer=True,
            kernel_regularizer=True,
            bias_regularizer=True,
        ),
        top_ffnn_meta_model_kwargs=dict(
            blocks=3,
            meta_layer_kwargs=dict(
                max_units=64,
                batch_normalization=True,
                residual=True,
                dropout=True,
                l1_regularization=1e-4,
                l2_regularization=1e-4,
                activity_regularizer=True,
                kernel_regularizer=True,
                bias_regularizer=True,
            ),
        )
    )
