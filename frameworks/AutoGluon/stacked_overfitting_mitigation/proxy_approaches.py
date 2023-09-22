import numpy as np
import logging

from shutil import rmtree

import pandas as pd
from sklearn.model_selection import train_test_split

from autogluon.tabular import TabularPredictor
from .utils import _check_stacked_overfitting_from_leaderboard

logger = logging.getLogger(__name__)


def stacked_overfitting_proxy_model(train_data, label, problem_type, split_random_state=42):
    """Approximates whether one should use stacking or not."""

    # add duplicates to the code?

    classification_problem = problem_type in ["binary", "multiclass"]
    if problem_type == "binary":
        metric = "roc_auc"
    elif problem_type == "multiclass":
        metric = "log_loss"
    elif problem_type == "regression":
        metric = "mse"
    else:
        raise ValueError(f"Unsupported Problem Type: {problem_type}")

    inner_train_data, outer_val_data = train_test_split(
        train_data, test_size=1 / 9, random_state=split_random_state, stratify=train_data[label] if classification_problem else None
    )

    # --- Proxy Model Definition ---
    # In essence, the idea is to find a cheap model that can spot the leak quickly with high confidence.
    predictor_para = dict(
        eval_metric=metric,
        label=label,
        verbosity=0,
        problem_type=problem_type,
        learner_kwargs=dict(random_state=0),
    )
    fit_para = dict(
        hyperparameters={
            1: {"RF": [{}], "KNN": [{}]},
            2: {"RF": [{}], "LR": [{}]},
        },
        num_stack_levels=1,
        num_bag_sets=1,
        num_bag_folds=8,
        fit_weighted_ensemble=False,
    )

    logger.info(f"Start running Proxy Model on data, RS: {split_random_state}")
    predictor = TabularPredictor(**predictor_para)
    predictor.fit(train_data=inner_train_data, **fit_para)
    val_leaderboard = predictor.leaderboard(outer_val_data, silent=True).reset_index(drop=True)
    stacked_overfitting, *_ = _check_stacked_overfitting_from_leaderboard(val_leaderboard)
    rmtree(predictor.path)

    # Additional indicators / flags to not use stacking
    lr_score = val_leaderboard.loc[val_leaderboard["model"] == "LinearModel_BAG_L2", "score_test"].iloc[0]
    rf_score = val_leaderboard.loc[val_leaderboard["model"] == "RandomForest_BAG_L2", "score_test"].iloc[0]
    stacked_overfitting = stacked_overfitting or (lr_score > rf_score)

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        logger.info(val_leaderboard[["model", "score_test", "score_val"]].sort_values(by="score_val", ascending=False))

    return stacked_overfitting


def determine_stacked_overfitting(train_data, label, problem_type):
    rng = np.random.RandomState(42)
    use_stacking_opinions = []
    for _ in range(5):
        use_stacking_opinions.append(stacked_overfitting_proxy_model(train_data, label, problem_type, split_random_state=rng.randint(0, 2**32)))
    logger.info(f"Proxy Opinions: {use_stacking_opinions}")
    use_stacking = not any(use_stacking_opinions)

    return use_stacking


def verify_stacking_settings(use_stacking, fit_para):
    fit_para = fit_para.copy()

    if use_stacking:
        fit_para["num_stack_levels"] = fit_para.get("num_stack_levels", 1)
    else:
        fit_para["num_stack_levels"] = 0

    return fit_para

