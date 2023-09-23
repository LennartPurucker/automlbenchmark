from autogluon.core.metrics import get_metric
from autogluon.tabular import TabularPredictor
import logging
from functools import partial

from stacked_overfitting_mitigation.utils import get_best_val_models

log = logging.getLogger(__name__)


def no_holdout(train_data, label, predictor_para, fit_para):
    predictor = TabularPredictor(**predictor_para)
    predictor.fit(train_data=train_data, **fit_para)

    problem_type = predictor_para["problem_type"]
    if problem_type in ["binary", "multiclass"]:
        alt_metric = "log_loss"
        pred = partial(predictor.predict_proba, as_multiclass=False)
    else:
        alt_metric = "mse"
        pred = partial(predictor.predict)


    # Decide between L1 and L2 model based on heuristic
    leaderboard = predictor.leaderboard(silent=True)

    # Determine best l1 and l2 model
    best_l1_model, best_l2_model, leaking_models_exist = get_best_val_models(leaderboard)
    score_l1_oof = leaderboard.loc[leaderboard["model"] == best_l1_model, "score_val"].iloc[0]

    if (not leaking_models_exist) or (score_l1_oof >= leaderboard.loc[leaderboard["model"] == best_l2_model, "score_val"].iloc[0]):
        return predictor

    # -- Obtain reproduction scores
    X = predictor.transform_features(train_data.drop(columns=[label]))
    y = predictor.transform_labels(train_data[label])
    
    l2_repo_oof = pred(X, model=best_l2_model, as_reproduction_predictions_args=dict(y=y))

    l1_models = [model_name for model_name in leaderboard["model"] if model_name.endswith("BAG_L1")]
    model_name_to_oof = {model_name: predictor.get_oof_pred_proba(model=model_name, as_multiclass=False) for model_name in l1_models}
    l2_true_repo_oof = pred(X, model=best_l2_model,  as_reproduction_predictions_args=dict(y=y, model_name_to_oof=model_name_to_oof))

    eval_metric = get_metric(alt_metric, problem_type=problem_type)
    am_score_l2_repo = eval_metric(y, l2_repo_oof)
    am_score_l2_true_repo = eval_metric(y, l2_true_repo_oof)

    # -- Detect leak and if we think the leak happens set best model to be the L1 model
    if am_score_l2_repo < am_score_l2_true_repo:
        log.info(f"Detected leak, switching to {best_l1_model}")
        predictor.set_model_best(best_l1_model, save_trainer=True)
    log.info(f"No leak detected")

    return predictor
