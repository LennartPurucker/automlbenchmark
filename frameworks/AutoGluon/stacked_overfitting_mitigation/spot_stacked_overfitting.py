import numpy as np
import pandas as pd
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


def get_non_leaking_and_leaking_model_names(leaderboard: pd.DataFrame) -> Tuple[List[str], List[str], bool]:
    leaderboard = leaderboard.copy()

    NON_LEAKING_MODEL_NAMES_IN_AUTOGLUON = ["WeightedEnsemble_L2", "WeightedEnsemble_BAG_L2", "WeightedEnsemble_ALL_L2", "WeightedEnsemble_FULL_L2"]

    non_leaking = []
    leaking = []
    for model_name in set(leaderboard["model"]):

        if (model_name in NON_LEAKING_MODEL_NAMES_IN_AUTOGLUON) or model_name.endswith("L1"):
            non_leaking.append(model_name)
        else:
            leaking.append(model_name)

    leaking_models_exist = len(leaking) > 0

    return non_leaking, leaking, leaking_models_exist


def get_best_val_models(leaderboard):
    non_leaking_names, leaking_names, leaking_models_exist = get_non_leaking_and_leaking_model_names(leaderboard)

    best_non_leaking_model = leaderboard[leaderboard["model"].isin(non_leaking_names)].sort_values(by="score_val", ascending=False).iloc[0].loc["model"]

    best_leaking_model = None
    if leaking_names:
        best_leaking_model = leaderboard[leaderboard["model"].isin(leaking_names)].sort_values(by="score_val", ascending=False).iloc[0].loc["model"]

    return best_non_leaking_model, best_leaking_model, leaking_models_exist


def _check_so_for_models(best_non_leaking_model, best_leaking_model, leaderboard):
    score_non_leaking_oof = leaderboard.loc[leaderboard["model"] == best_non_leaking_model, "score_val"].iloc[0]
    score_non_leaking_test = leaderboard.loc[leaderboard["model"] == best_non_leaking_model, "score_test"].iloc[0]

    score_leaking_oof = leaderboard.loc[leaderboard["model"] == best_leaking_model, "score_val"].iloc[0]
    score_leaking_test = leaderboard.loc[leaderboard["model"] == best_leaking_model, "score_test"].iloc[0]

    # l1 worse val score than l2+
    stacked_overfitting = score_non_leaking_oof < score_leaking_oof
    # l2+ worse test score than L1
    stacked_overfitting = stacked_overfitting and (score_non_leaking_test >= score_leaking_test)

    return stacked_overfitting


def spot_based_on_best_models(leaderboard):
    best_non_leaking_model, best_leaking_model, leaking_models_exist = get_best_val_models(leaderboard)

    if not leaking_models_exist:
        return False

    return _check_so_for_models(best_non_leaking_model, best_leaking_model, leaderboard)


def spot_on_all_model_combinations(leaderboard):
    non_leaking_names, leaking_names, leaking_models_exist = get_non_leaking_and_leaking_model_names(leaderboard)
    best_non_leaking_model = leaderboard[leaderboard["model"].isin(non_leaking_names)].sort_values(by="score_val", ascending=False).iloc[0].loc["model"]

    stacked_overfitting = False
    for leaking_model_name in leaking_names:
        stacked_overfitting = stacked_overfitting or _check_so_for_models(best_non_leaking_model, leaking_model_name, leaderboard)
        if stacked_overfitting:
            break

    return stacked_overfitting


def spot_stacking_is_worse(leaderboard):
    non_leaking_names, leaking_names, leaking_models_exist = get_non_leaking_and_leaking_model_names(leaderboard)

    # if no stacking models were fit, do not try to do L2 at refit
    if not leaking_models_exist:
        return True

    score_non_leaking_test = leaderboard.loc[leaderboard["model"].isin(non_leaking_names), "score_test"].max()
    score_leaking_test = leaderboard.loc[leaderboard["model"].isin(leaking_names), "score_test"].max()
    return score_non_leaking_test >= score_leaking_test


def _check_stacked_overfitting_from_leaderboard(leaderboard, dynamic_stacking_variant):
    if dynamic_stacking_variant == "default":
        stacked_overfitting = spot_based_on_best_models(leaderboard)
    elif dynamic_stacking_variant == "safe":
        stacked_overfitting = spot_on_all_model_combinations(leaderboard)
    elif dynamic_stacking_variant == "aggressive":
        stacked_overfitting = spot_on_all_model_combinations(leaderboard)
        stacked_overfitting = stacked_overfitting or spot_stacking_is_worse(leaderboard)
    else:
        raise ValueError(f"Unknown dynamic stacking variant. Got: {dynamic_stacking_variant}")

    return stacked_overfitting
