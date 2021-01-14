"""Module providing stratified random holdouts generator."""
from typing import Generator
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from tqdm.auto import tqdm
from sanitize_ml_labels import sanitize_ml_labels


def stratified_holdouts(
    n_splits: int,
    random_state: int,
    train_size: float,
    X: pd.DataFrame,
    y: pd.DataFrame,
    task_name: str = "",
    verbose: bool = True,
    leave: bool = False
) -> Generator:
    """Return generator for stratified random holdouts.
    
    Parameters
    -----------------------
    n_splits: int,
        Number of holdouts.
    random_state: int,
        The random state to reproduce the sampled holdouts.
    train_size: float,
        The rate of values to leave in the training set.
    X: pd.DataFrame,
        The input data.
    y: pd.DataFrame,
        The output data labels.
    task_name: str = "",
        Name of the task to be shown in the loading bar.
    verbose: bool = True,
        Wether to show the loading bar.
        By default, True.
    leave: bool = False,
        Wether to leave the loading bar.
        By default, False.

    Returns
    -----------------------
    Generator with stratified random holdouts.
    """
    return (
        (
            holdout_number,
            X.iloc[train_idx],
            X.iloc[test_idx],
            y.iloc[train_idx],
            y.iloc[test_idx]
        )
        for holdout_number, (train_idx, test_idx) in tqdm(
            enumerate(StratifiedShuffleSplit(
                n_splits=n_splits,
                train_size=0.8,
                random_state=random_state
            ).split(X, y)),
            desc="Computing holdouts for task {}".format(
                sanitize_ml_labels(task_name)
            ),
            total=n_splits,
            disable=not verbose,
            leave=leave
        )
    )
