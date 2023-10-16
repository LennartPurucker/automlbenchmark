import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from autogluon.tabular.configs.hyperparameter_configs import hyperparameter_config_dict
from autogluon.tabular.trainer.model_presets.presets_custom import get_preset_custom
import logging

logger = logging.getLogger(__name__)


def get_importance_diff_holdout_oof(holdout_data: pd.DataFrame, label: str,
                                    predictor: TabularPredictor,
                                    model: str,
                                    num_shuffle_sets: int = 1) -> pd.DataFrame:
    model_loaded = predictor._trainer.load_model(model)
    oof_features = model_loaded.feature_metadata.get_features(required_special_types=["stack"])

    X_holdout_inner = predictor.transform_features(data=holdout_data, model=model)
    y_holdout_inner = predictor.transform_labels(labels=holdout_data[label])

    # predict on all input data - predictions are the individual predictions of each fold model
    # -> final importance average over all predictions
    child_holdout_importance = model_loaded.compute_feature_importance(X=X_holdout_inner, y=y_holdout_inner,
                                                                       num_shuffle_sets=num_shuffle_sets,
                                                                       from_children=True, silent=False,
                                                                       features=oof_features)

    # predict on all input data - prediction are the average of all fold models' predictions
    # -> final importance average over all predictions
    holdout_importance = model_loaded.compute_feature_importance(X=X_holdout_inner, y=y_holdout_inner,
                                                                 num_shuffle_sets=num_shuffle_sets, silent=False,
                                                                 features=oof_features)



    diff_importance = holdout_importance[["importance"]].copy()
    diff_importance["importance_children"] = child_holdout_importance["importance"]

    if not (model_loaded._child_oof or not model_loaded._bagged_mode):
        # This is the correct way (probably I should make this less complicated)
        _, y_train_inner = predictor.load_data_internal()
        X_train_inner = predictor._learner.get_inputs_to_stacker(model=model)

        # predict on each split as for OOF - predictions are the individual predictions of the split's fold model
        # -> final importance average over all predictions
        oof_importance = model_loaded.compute_feature_importance(X=X_train_inner, y=y_train_inner,
                                                                 num_shuffle_sets=num_shuffle_sets, is_oof=True,
                                                                 silent=False, features=oof_features)
        diff_importance["importance_oof"] = oof_importance["importance"]
    else:
        diff_importance["importance_oof"] = np.nan

    diff_importance["drop"] = (diff_importance["importance_children"] <= 0) | (diff_importance["importance"] <= 0)
    diff_importance = diff_importance.sort_values(by=["importance"], ascending=False)
    diff_importance['model'] = model
    return diff_importance


def _get_allowed_models(holdout_data, label, predictor, model):
    importance_df = get_importance_diff_holdout_oof(holdout_data, label, predictor, model, 10)
    allowed_oof_features = [x for x in list(importance_df[~importance_df['drop']].index) if ("BAG_L1" in x)]
    logger.info(f"Allowed features for {model}: {allowed_oof_features}")
    return allowed_oof_features, importance_df.reset_index()


def get_preselected_fit_hps(holdout_data: pd.DataFrame, label: str, predictor: TabularPredictor):
    l2_models = predictor.get_model_names(level=2)
    postfix = "_BAG_L2"
    l2_models = [bm for bm in l2_models if bm.endswith(postfix)]
    logger.info(f"L2 models from first fit: {l2_models}")

    reverse_model_names = {
        "NeuralNetTorch": ["NN_TORCH", {}],
        "LightGBM": ["GBM", {}],
        "LightGBMLarge": ["GBM", get_preset_custom("GBMLarge", predictor.problem_type)[0]],
        "LightGBMXT": ["GBM", {"extra_trees": True, "ag_args": {"name_suffix": "XT"}}],
        "CatBoost": ["CAT", {}],
        "XGBoost": ["XGB", {}],
        "NeuralNetFastAI": ["FASTAI", {}],
        "RandomForestGini": ["RF", {"criterion": "gini",
                                    "ag_args": {"name_suffix": "Gini", "problem_types": ["binary", "multiclass"]}}],
        "RandomForestEntr": ["RF", {"criterion": "entropy",
                                    "ag_args": {"name_suffix": "Entr", "problem_types": ["binary", "multiclass"]}}],
        "RandomForestMSE": ["RF", {"criterion": "squared_error",
                                   "ag_args": {"name_suffix": "MSE", "problem_types": ["regression", "quantile"]}}],
        "ExtraTreesGini": ["XT", {"criterion": "gini",
                                  "ag_args": {"name_suffix": "Gini", "problem_types": ["binary", "multiclass"]}}],
        "ExtraTreesEntr": ["XT", {"criterion": "entropy",
                                  "ag_args": {"name_suffix": "Entr", "problem_types": ["binary", "multiclass"]}}],
        "ExtraTreesMSE": ["XT", {"criterion": "squared_error",
                                 "ag_args": {"name_suffix": "MSE", "problem_types": ["regression", "quantile"]}}],
        "KNeighborsUnif": ["KNN", {"weights": "uniform", "ag_args": {"name_suffix": "Unif"}}],
        "KNeighborsDist": ["KNN", {"weights": "distance", "ag_args": {"name_suffix": "Dist"}}]
    }
    reverse_model_names = {k + postfix: v for k, v in reverse_model_names.items()}

    if l2_models:
        hps = dict()

        missing_l2_models = [m for m in reverse_model_names.keys() if m not in l2_models]
        allowed_models_collection = []
        importance_df_all = None
        for l2_model in l2_models:
            default_key, default_hp = reverse_model_names[l2_model]

            allowed_models, tmp_importance_df = _get_allowed_models(holdout_data, label, predictor, l2_model)

            if importance_df_all is None:
                importance_df_all = tmp_importance_df
            else:
                importance_df_all = pd.concat([importance_df_all, tmp_importance_df], axis=0)

            if not allowed_models:
                continue

            default_hp["allowed_oof_features"] = allowed_models
            allowed_models_collection.append(allowed_models)

            if default_key not in hps:
                hps[default_key] = []
            hps[default_key].append(default_hp)

        always_allowed_models = list(set.intersection(*map(set, allowed_models_collection)))
        if not always_allowed_models:
            always_allowed_models = 'do-not-stack-model'

        for l2_model in missing_l2_models:
            default_key, default_hp = reverse_model_names[l2_model]
            default_hp["allowed_oof_features"] = always_allowed_models

            if default_key not in hps:
                hps[default_key] = []
            hps[default_key].append(default_hp)

        if allowed_models_collection:
            # at least some models use oof predictions
            disable_stacking = False
        else:
            # no model is planing on using l2 predictions, hence do not stack
            disable_stacking = True

    else:
        hps = hyperparameter_config_dict['default'].copy()

        # could be due to crash or similar that no L2 model exist, hence, refitting with all data and time
        #   could fix it. Thus, do not set to False.
        disable_stacking = False
        importance_df_all = None

    logger.info(f"OOF selection decided for: disable stacking {disable_stacking} and HPs: {hps}")

    return hps, disable_stacking, importance_df_all



