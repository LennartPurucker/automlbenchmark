from shutil import rmtree
import gc
import logging
import pandas as pd
import time
from sklearn.model_selection import train_test_split
import concurrent.futures

from autogluon.tabular import TabularPredictor
from stacked_overfitting_mitigation.utils import _check_stacked_overfitting_from_leaderboard, get_label_train_data
from stacked_overfitting_mitigation.oof_selection import get_preselected_fit_hps

logger = logging.getLogger(__name__)


def _verify_stacking_settings(use_stacking, fit_para):
    fit_para = fit_para.copy()

    if use_stacking:
        fit_para["num_stack_levels"] = fit_para.get("num_stack_levels", 1)
    else:
        fit_para["num_stack_levels"] = 0

    return fit_para


def _first_fit(para):
    # due to multiprocessing code
    classification_problem, holdout_seed, predictor_para, time_limit_fit_1, fit_para, refit_autogluon, select_oof_predictions, dynamic_stacking = para

    train_data, label, fit_para = get_label_train_data(fit_para, predictor_para)

    inner_train_data, outer_val_data = train_test_split(
        train_data, test_size=1 / 9, random_state=holdout_seed,
        stratify=train_data[label] if classification_problem else None
    )

    # Remove memory footprint
    del train_data
    time.sleep(1)
    gc.collect()

    logger.info(f"Start running AutoGluon on subset of data")
    predictor = TabularPredictor(**predictor_para)
    predictor.fit(train_data=inner_train_data, time_limit=time_limit_fit_1, **fit_para)

    # -- Obtain info from holdout
    val_leaderboard = predictor.leaderboard(outer_val_data, silent=True).reset_index(drop=True)
    best_model_on_holdout = val_leaderboard.loc[val_leaderboard["score_test"].idxmax(), "model"]
    stacked_overfitting, *_ = _check_stacked_overfitting_from_leaderboard(val_leaderboard)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        logger.info(val_leaderboard.sort_values(by="score_val", ascending=False))
    val_leaderboard = val_leaderboard.reset_index()

    logger.info(f"Stacked overfitting in this run: {stacked_overfitting}")

    if select_oof_predictions:
        model_hps, disable_stacking, importance_df = get_preselected_fit_hps(outer_val_data, label, predictor)
    else:
        model_hps, disable_stacking, importance_df = 'default', False, None

    # FIXME: move this up later but keep here for now such that we get import df once for all cases
    if select_oof_predictions and (not (stacked_overfitting and dynamic_stacking)):
        model_hps, disable_stacking = 'default', False

    if refit_autogluon:
        rmtree(predictor.path)  # clean up
        time.sleep(5)  # wait for the folder to be correctly deleted and log messages
        del predictor
        del inner_train_data
        del outer_val_data
        time.sleep(1)  # Wait for system calls
        gc.collect()
        predictor = None

    return predictor, best_model_on_holdout, stacked_overfitting, val_leaderboard, model_hps, disable_stacking, importance_df


def _second_fit(time_limit_fit_2, predictor_para, fit_para):
    predictor = TabularPredictor(**predictor_para)
    predictor.fit(time_limit=time_limit_fit_2, **fit_para)
    return predictor


def use_holdout(predictor_para, fit_para, refit_autogluon=False, select_on_holdout=False,
                dynamic_stacking=False, dynamic_fix=False,
                ges_holdout=False, holdout_seed=42, fix_predictor_para=None, dynamic_stacking_limited=False,
                select_oof_predictions=False,
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
    dynamic_fix
        If True, we dynamically enable a fix if we spotted stacked overfitting.
    ges_holdout (!OLD)
        If True, we compute a weight vector, using greedy ensemble selection (GES), on the holdout data. Moreover, we set the best model to the
        weighted ensemble with this weight vector. Note, we check both the L2 and L3 weighted ensemble and use the better one in the end.
        If we refit, we stick to the weights computed on the holdout data.
    dynamic_stacking_limited
        Limit dynamic stacking to 4h in total.
    select_oof_predictions
        Compute feature importance of OOF predictions and filter useless/hurtful OOF prediction columns.
    """

    # Get holdout
    classification_problem = predictor_para["problem_type"] in ["binary", "multiclass"]

    time_limit = fit_para.pop("time_limit", 60)  # in seconds
    if refit_autogluon:
        if dynamic_stacking_limited:
            time_start = time.time()
            time_limit_fit_1 = int(time_limit * 1 / 4)
        else:
            time_limit_fit_1 = time_limit
            time_limit_fit_2 = time_limit
    else:
        time_limit_fit_1 = time_limit
        time_limit_fit_2 = 0

    first_fit_para = [classification_problem, holdout_seed, predictor_para, time_limit_fit_1, fit_para, refit_autogluon, select_oof_predictions, dynamic_stacking]
    # first fit in subprocess for memory safety
    with concurrent.futures.ProcessPoolExecutor() as executor:
        f = executor.submit(_first_fit, first_fit_para)
        ret = f.result()

    predictor, best_model_on_holdout, stacked_overfitting, val_leaderboard, model_hps, disable_stacking, importance_df = ret

    if refit_autogluon and dynamic_stacking_limited:
        time_spend_fit_1 = int(time.time() - time_start)
        time_limit_fit_2 = time_limit - time_spend_fit_1
        logger.info(
            f"Spend {time_spend_fit_1} seconds for first fit. Dynamic Refit for: {time_limit_fit_2} seconds (non-dynamic would have been: {int(time_limit * 3 / 4)})")

    if dynamic_stacking:
        logger.info(f"Check if stacking used for refit or not")
        fit_para = _verify_stacking_settings(use_stacking=not stacked_overfitting, fit_para=fit_para)

    if dynamic_fix and stacked_overfitting:
        logger.info(f"Enable Dynamic Fix")
        # Enable fix if we spotted SO and use stacking
        predictor_para = fix_predictor_para
        fit_para = _verify_stacking_settings(use_stacking=True, fit_para=fit_para)

    if select_oof_predictions:
        fit_para['hyperparameters'] = model_hps

        # Disable stacking if no need for it due to selection
        if disable_stacking:
            fit_para = _verify_stacking_settings(use_stacking=False, fit_para=fit_para)

    # Refit and reselect
    if refit_autogluon:
        logger.info(f"Refit with the following configs, predictor: {predictor_para} and fit: {fit_para}")
        predictor = _second_fit(time_limit_fit_2, predictor_para, fit_para)

    if select_on_holdout:
        predictor.set_model_best(best_model_on_holdout, save_trainer=True)

    return predictor, val_leaderboard[["model", "score_test"]].rename({"score_test": "unbiased_score_val"}, axis=1), importance_df

# ! ---- old
#     if ges_holdout:
#         # Obtain best GES weights on holdout data
#         ges_train_data = predictor.transform_features(outer_val_data.drop(columns=[label]))
#         ges_label = predictor.transform_labels(outer_val_data[label])
#         l1_ges = \
#             predictor.fit_weighted_ensemble(base_models_level=1, new_data=[ges_train_data, ges_label],
#                                             name_suffix="HOL1")[
#                 0]
#         l2_ges = \
#             predictor.fit_weighted_ensemble(base_models_level=2, new_data=[ges_train_data, ges_label],
#                                             name_suffix="HOL2")[
#                 0]
#         l1_ges = predictor._trainer.load_model(l1_ges)
#         l2_ges = predictor._trainer.load_model(l2_ges)
#
#         if l1_ges.val_score >= l2_ges.val_score:  # if they are equal, we prefer the simpler model (e.g. lower layer)
#             ho_weights = l1_ges._get_model_weights()
#             weights_level = 2
#         else:
#             ho_weights = l2_ges._get_model_weights()
#             weights_level = 3
#     if ges_holdout:
#         final_ges = f"WeightedEnsemble_L{weights_level}"
#
#         bm_names = []
#         weights = []
#         for bm_name, weight in ho_weights.items():
#             bm_names.append(bm_name)
#             weights.append(weight)
#
#         # Update weights of GES
#         f_ges = predictor._trainer.load_model(final_ges)
#         f_ges.models[0].base_model_names = bm_names
#         f_ges.models[0].weights_ = weights
#         predictor._trainer.save_model(f_ges)
#
#         # Set GES to be best model
#         predictor.set_model_best(final_ges, save_trainer=True)


# !old --- usage
#     # not_supported_pt = ("clean_oof_predictions" in so_mitigation) and (predictor_para["problem_type"] in ["multiclass", "regression"]) # or not_supported_pt
#     from stacked_overfitting_mitigation.holdout_approaches import use_holdout
#
#     if so_mitigation == "ho_select":
#         mitigate_para = dict(select_on_holdout=True)
#     elif so_mitigation == "ho_select_refit":
#         mitigate_para = dict(refit_autogluon=True, select_on_holdout=True)
#     elif so_mitigation == "ho_dynamic_stacking":
#         mitigate_para = dict(refit_autogluon=True, dynamic_stacking=True)
#     elif so_mitigation == "ho_dynamic_stacking_select_oof":
#         mitigate_para = dict(refit_autogluon=True, dynamic_stacking=True, select_oof_predictions=True)
#     elif so_mitigation == "ho_dynamic_stacking_limited":
#         mitigate_para = dict(refit_autogluon=True, dynamic_stacking=True, dynamic_stacking_limited=True)
#     elif so_mitigation == "ho_ges_weights":
#         mitigate_para = dict(refit_autogluon=True, ges_holdout=True)
#     elif so_mitigation == "dynamic_clean_oof_predictions":
#         fix_predictor_para = predictor_para.copy()
#         fix_predictor_para["learner_kwargs"] = dict(clean_oof_predictions=True)
#         mitigate_para = dict(refit_autogluon=True, dynamic_fix=True, fix_predictor_para=fix_predictor_para)
#     else:
#         raise ValueError(f"Unknown SO mitigation: {so_mitigation}")
#
#     predictor, ho_lb, ho_importance_df = use_holdout(predictor_para, fit_para, **mitigate_para)
