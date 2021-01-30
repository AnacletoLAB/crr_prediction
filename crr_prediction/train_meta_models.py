"""Method to train the meta models on all cell lines."""
from typing import Dict, Callable, Tuple
import os
import silence_tensorflow.auto
import pandas as pd
from tqdm.auto import tqdm
from epigenomic_dataset import load_all_tasks
from epigenomic_dataset.utils import normalize_epigenomic_data, get_cell_lines
from crr_prediction.baseline_models import deep_enhancers
from crr_prediction.meta_models import build_cnn_meta_model, build_mlp_meta_model
from ucsc_genomes_downloader import Genome
from meta_models.tuner import RayHyperOptTuner
from meta_models.utils import stratified_holdouts, get_minimum_gpu_rate_per_trial, enable_subgpu_training
from multiprocessing import cpu_count
from cache_decorator import Cache


@Cache(
    cache_path="results/{model}/performance/{task}/{cell_line}/{holdout_number}.csv.gz",
    args_to_ignore=[
        "train_x", "test_x", "train_y", "test_y",
        "build_sequences", "build_meta_model", "genome", "total_threads"
    ]
)
def train(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    train_y: pd.DataFrame,
    test_y: pd.DataFrame,
    build_sequences: Callable,
    build_meta_model: Callable,
    model: str,
    task: str,
    cell_line: str,
    holdout_number: int,
    random_state: int = 42,
    valid_size: float = 0.2,
    batch_size: int = 256,
    num_samples: int = 200,
    random_search_steps: int = 100,
    resolution: int = 10,
    total_threads: int = None,
    genome: Genome = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return tuple with all details of the training and tuning process.

    Parameters
    ---------------------
    train_x: pd.DataFrame,
        Training input dataframe.
    test_x: pd.DataFrame,
        Test input dataframe.
    train_y: pd.DataFrame,
        Train output dataframe.
    test_y: pd.DataFrame,
        Test output dataframe.
    build_sequences: Callable,
        Method to use to generate the training sequences.
    build_meta_model: Callable,
        Method to build the meta model.
    model: str,
        Name of the meta model.
    task: str,
        Name of the task.
    cell_line: str,
        Name of the cell line.
    holdout_number: int,
        Number of the holdout to compute.
    random_state: int = 42,
        Random state to reproduce holdout.
    valid_size: float = 0.2,
        Size for the validation data.
    batch_size: int = 256,
        Batch size for the sequences.
    num_samples: int = 200,
        Number of samples for the optimization.
    random_search_steps: int = 100,
        Number of initial random sampled points.
    resolution: int = 10,
        Resolution for the range of hyper-parameters.
    total_threads: int = None,
        Number of threads to use.
    genome: Genome = None,
        Genome to use for the bed sequences.

    Returns
    ---------------------
    Dataframe with model performance.
    """
    subtrain_x, valid_x, subtrain_y, valid_y = stratified_holdouts(
        random_state=random_state,
        train_size=1-valid_size,
        X=train_x,
        y=train_y
    )
    train, test, subtrain, valid = build_sequences(
        train_x=train_x,
        test_x=test_x,
        subtrain_x=subtrain_x,
        valid_x=valid_x,
        train_y=train_y,
        test_y=test_y,
        subtrain_y=subtrain_y,
        valid_y=valid_y,
        genome=genome,
        batch_size=batch_size,
        random_state=random_state
    )
    tuner = RayHyperOptTuner(
        meta_model=build_meta_model(subtrain.features),
        resolution=resolution
    )
    tuning_analysis = tuner.tune(
        train=subtrain.rasterize(verbose=False),
        validation_data=valid.rasterize(verbose=False),
        name=f"{task}-{holdout_number}",
        num_samples=num_samples,
        random_search_steps=random_search_steps,
        total_threads=total_threads,
        verbose=False
    )

    os.makedirs(
        f"results/{model}/tuning_analyses/{task}/{cell_line}", exist_ok=True)
    tuning_analysis.to_csv(
        f"results/{model}/tuning_analyses/{task}/{cell_line}/{holdout_number}.csv.gz",
        index=False
    )

    history = tuner.fit(train=train.rasterize(verbose=False))

    os.makedirs(
        f"results/{model}/training_histories/{task}/{cell_line}", exist_ok=True)
    history.to_csv(
        f"results/{model}/training_histories/{task}/{cell_line}/{holdout_number}.csv.gz",
        index=False
    )

    train_performance = tuner.evaluate(train, verbose=False)
    test_performance = tuner.evaluate(test, verbose=False)
    subtrain_performance = tuner.evaluate(subtrain, verbose=False)
    valid_performance = tuner.evaluate(valid, verbose=False)

    metadata = {
        "task": task,
        "cell_line": cell_line,
        "holdout_number": holdout_number,
        **tuner.optimal_configuration
    }

    return pd.DataFrame([
        {
            "run_type": "train",
            **metadata,
            **train_performance,
        },
        {
            "run_type": "test",
            **metadata,
            **test_performance,
        },
        {
            "run_type": "subtrain",
            **metadata,
            **subtrain_performance,
        },
        {
            "run_type": "valid",
            **metadata,
            **valid_performance,
        }
    ])


def train_meta_models(
    build_sequences: Callable,
    build_meta_model: Callable,
    model: str,
    window_size: int = 256,
    n_splits: int = 10,
    random_state: int = 42,
    test_size: float = 0.2,
    valid_size: float = 0.2,
    batch_size: int = 256,
    num_samples: int = 200,
    random_search_steps: int = 100,
    resolution: int = 10,
    total_threads: int = None,
    genome: Genome = None
) -> pd.DataFrame:
    """Run full suite of experiments on the given metamodel.

    Parameters
    -------------------
    build_sequences: Callable,
        Method to use to generate the training sequences.
    build_meta_model: Callable,
        Method to build the meta model.
    model: str,
        Name of the meta model.
    window_size: int = 256,
        Window size.
    n_splits: int = 10,
        Number of random holdouts
    holdout_number: int,
        Number of the holdout to compute.
    random_state: int = 42,
        Random state to reproduce holdout.
    test_size: float = 0.2,
        Size for the test data.
    valid_size: float = 0.2,
        Size for the validation data.
    batch_size: int = 256,
        Batch size for the sequences.
    num_samples: int = 200,
        Number of samples for the optimization.
    random_search_steps: int = 100,
        Number of initial random sampled points.
    resolution: int = 10,
        Resolution for the range of hyper-parameters.
    total_threads: int = None,
        Number of threads to use.
    genome: Genome = None,
        Genome to use for the bed sequences.

    Returns
    -------------------
    DataFrame with all performance.
    """
    if total_threads is None:
        total_threads = cpu_count()
    enable_subgpu_training()
    all_performance = []
    for cell_line in tqdm(get_cell_lines(), desc="Cell lines"):
        for (X, y), task in load_all_tasks(
            cell_line=cell_line,
            window_size=window_size,
        ):
            for holdout_number, train_x, test_x, train_y, test_y in stratified_holdouts(
                n_splits=n_splits,
                random_state=random_state,
                train_size=1-test_size,
                X=X,
                y=y,
                task_name=task
            ):
                all_performance.append(train(
                    train_x, test_x, train_y, test_y,
                    build_sequences, build_meta_model,
                    model=model,
                    task=task,
                    cell_line=cell_line,
                    holdout_number=holdout_number,
                    random_state=random_state,
                    valid_size=valid_size,
                    batch_size=batch_size,
                    num_samples=num_samples,
                    random_search_steps=random_search_steps,
                    resolution=resolution,
                    total_threads=total_threads,
                    genome=genome
                ))
    return pd.concat(all_performance)
