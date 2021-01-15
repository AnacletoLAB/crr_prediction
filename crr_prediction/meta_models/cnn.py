"""Module providing meta-cnn model."""
from meta_models.meta_models import CNN1DMetaModel


def build_cnn_meta_model(window_size: int) -> CNN1DMetaModel:
    """Return CNN meta model.

    Parameters
    -------------------------
    window_size: int,
        Training window size.

    Returns
    -------------------------
    CNN1D model.
    """
    return CNN1DMetaModel(
        blocks=3,
        input_shape=(window_size, 4),
        meta_layer_kwargs=dict(
            batch_normalization=False,
            residual=True,
            max_kernel_size=10,
            min_kernel_size=2,
            max_filters=256
        ),
        top_ffnn_meta_model_kwargs=dict(
            blocks=3,
            meta_layer_kwargs=dict(
                max_units=256,
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
