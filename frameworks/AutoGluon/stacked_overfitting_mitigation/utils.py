import numpy as np
import pandas as pd

def get_best_val_models(leaderboard):
    leaderboard = leaderboard.copy()
    non_leaking = ["WeightedEnsemble_L2", "WeightedEnsemble_BAG_L2"]
    for non_leaker in non_leaking:
        leaderboard["model"] = leaderboard["model"].str.replace(non_leaker, non_leaker.replace("L2", "L1"))
    best_l1_model = leaderboard[leaderboard["model"].str.endswith("L1")].sort_values(by="score_val", ascending=False).iloc[0].loc["model"]
    leaking_models_exist = any(m.endswith("L2") for m in leaderboard["model"])

    if leaking_models_exist:
        # Get best models per layer
        best_l2_model = leaderboard[~leaderboard["model"].str.endswith("L1")].sort_values(by="score_val", ascending=False).iloc[0].loc["model"]
    else:
        best_l2_model = None

    # -- Revert back
    if best_l1_model in [x.replace("L2", "L1") for x in non_leaking]:
        best_l1_model = best_l1_model.replace("L1", "L2")
    for non_leaker in non_leaking:
        leaderboard["model"] = leaderboard["model"].str.replace(non_leaker.replace("L2", "L1"), non_leaker)

    return best_l1_model, best_l2_model, leaking_models_exist


def _check_stacked_overfitting_from_leaderboard(leaderboard):
    best_l1_model, best_l2_model, leaking_models_exist = get_best_val_models(leaderboard)

    score_l1_oof = leaderboard.loc[leaderboard["model"] == best_l1_model, "score_val"].iloc[0]
    score_l1_test = leaderboard.loc[leaderboard["model"] == best_l1_model, "score_test"].iloc[0]

    if leaking_models_exist:
        score_l2_oof = leaderboard.loc[leaderboard["model"] == best_l2_model, "score_val"].iloc[0]
        score_l2_test = leaderboard.loc[leaderboard["model"] == best_l2_model, "score_test"].iloc[0]

        # l1 worse val score than l2+
        stacked_overfitting = score_l1_oof < score_l2_oof
        # l2+ worse test score than L1
        stacked_overfitting = stacked_overfitting and (score_l1_test >= score_l2_test)

    else:
        # Stacked Overfitting is impossible
        score_l2_oof = np.nan
        score_l2_test = np.nan
        stacked_overfitting = False

    return stacked_overfitting, score_l1_oof, score_l2_oof, score_l1_test, score_l2_test


def get_label_train_data(fit_para, predictor_para):
    fit_para = fit_para.copy()
    train_data = pd.read_parquet(fit_para.get("train_data"))
    label = predictor_para["label"]
    fit_para.pop("train_data")

    return train_data, label, fit_para
