"""Module providing meta-mlp model."""
from meta_models.meta_models import FFNNMetaModel


def build_mlp_meta_model(features_number: int) -> FFNNMetaModel:
    """Return CNN meta model.

    Parameters
    -------------------------
    features_number: int,
        Number of input features.

    Returns
    -------------------------
    CNN1D model.
    """
    return FFNNMetaModel(
        input_shape=(features_number,),
        meta_layer_kwargs=dict(
            max_units=64,
            activity_regularizer=True,
            kernel_regularizer=True,
            bias_regularizer=True,
            batch_normalization=True,
            dropout=True
        ),
    )
