from shutil import rmtree
import logging
import pandas as pd
import time
from sklearn.model_selection import train_test_split

from autogluon.tabular import TabularPredictor
from stacked_overfitting_mitigation.utils import _check_stacked_overfitting_from_leaderboard

logger = logging.getLogger(__name__)


def _verify_stacking_settings(use_stacking, fit_para):
    fit_para = fit_para.copy()

    if use_stacking:
        fit_para["num_stack_levels"] = fit_para.get("num_stack_levels", 1)
    else:
        fit_para["num_stack_levels"] = 0

    return fit_para


def use_holdout(
        train_data, label, predictor_para, fit_para, refit_autogluon=False, select_on_holdout=False, dynamic_stacking=False, dynamic_fix=False,
        ges_holdout=False, holdout_seed=42, fix_predictor_para=None
):
    """A function to run different configurations of AutoGluon with a holdout set to avoid stacked overfitting.


    Parameters
    ----------
    refit_autogluon:
        If True, refit autogluon on all available data. Note, this is not a default AutoGluon refit (e.g. without bagging) but running default AutoGluon again.
    select_on_holdout
        If True, we select and set the best model based on the score on the holdout data. If we refit, we stick to the selection from the holdout data.
    dynamic_stacking
        If True, we dynamic select whether to use stacking for the refit based on whether we observed stacked overfitting on the holdout data.
    ges_holdout
        If True, we compute a weight vector, using greedy ensemble selection (GES), on the holdout data. Moreover, we set the best model to the
        weighted ensemble with this weight vector. Note, we check both the L2 and L3 weighted ensemble and use the better one in the end.
        If we refit, we stick to the weights computed on the holdout data.
    """

    # Get holdout
    classification_problem = predictor_para["problem_type"] in ["binary", "multiclass"]
    inner_train_data, outer_val_data = train_test_split(
        train_data, test_size=1 / 9, random_state=holdout_seed, stratify=train_data[label] if classification_problem else None
    )

    logger.info(f"Start running AutoGluon on subset of data data")
    predictor = TabularPredictor(**predictor_para)
    predictor.fit(train_data=inner_train_data, **fit_para)

    # -- Obtain info from holdout
    val_leaderboard = predictor.leaderboard(outer_val_data, silent=True).reset_index(drop=True)
    best_model_on_holdout = val_leaderboard.loc[val_leaderboard["score_test"].idxmax(), "model"]
    stacked_overfitting, *_ = _check_stacked_overfitting_from_leaderboard(val_leaderboard)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        logger.info(val_leaderboard.sort_values(by="score_val", ascending=False))

    logger.info(f"Stacked overfitting in this run: {stacked_overfitting}")

    if ges_holdout:
        # Obtain best GES weights on holdout data
        ges_train_data = predictor.transform_features(outer_val_data.drop(columns=[label]))
        ges_label = predictor.transform_labels(outer_val_data[label])
        l1_ges = predictor.fit_weighted_ensemble(base_models_level=1, new_data=[ges_train_data, ges_label], name_suffix="HOL1")[0]
        l2_ges = predictor.fit_weighted_ensemble(base_models_level=2, new_data=[ges_train_data, ges_label], name_suffix="HOL2")[0]
        l1_ges = predictor._trainer.load_model(l1_ges)
        l2_ges = predictor._trainer.load_model(l2_ges)

        if l1_ges.val_score >= l2_ges.val_score:  # if they are equal, we prefer the simpler model (e.g. lower layer)
            ho_weights = l1_ges._get_model_weights()
            weights_level = 2
        else:
            ho_weights = l2_ges._get_model_weights()
            weights_level = 3

    if dynamic_stacking:
        fit_para = _verify_stacking_settings(use_stacking=not stacked_overfitting, fit_para=fit_para)

    if dynamic_fix and stacked_overfitting:
        # Enable fix if we spotted SO and use stacking
        predictor_para = fix_predictor_para
        fit_para = _verify_stacking_settings(use_stacking=True, fit_para=fit_para)

    # Refit and reselect
    if refit_autogluon:
        rmtree(predictor.path)  # clean up
        time.sleep(5)  # wait for the folder to be correctly deleted and log messages

        predictor = TabularPredictor(**predictor_para)
        predictor.fit(train_data=train_data, **fit_para)

    if select_on_holdout:
        predictor.set_model_best(best_model_on_holdout, save_trainer=True)

    if ges_holdout:
        final_ges = f"WeightedEnsemble_L{weights_level}"

        bm_names = []
        weights = []
        for bm_name, weight in ho_weights.items():
            bm_names.append(bm_name)
            weights.append(weight)

        # Update weights of GES
        f_ges = predictor._trainer.load_model(final_ges)
        f_ges.models[0].base_model_names = bm_names
        f_ges.models[0].weights_ = weights
        predictor._trainer.save_model(f_ges)

        # Set GES to be best model
        predictor.set_model_best(final_ges, save_trainer=True)

    return predictor, val_leaderboard[["model", "score_test"]].rename({"score_test": "unbiased_score_val"}, axis=1)
