import logging
import os
import shutil
import warnings
import sys
import tempfile
from typing import Union

warnings.simplefilter("ignore")

if sys.platform == 'darwin':
    os.environ['OMP_NUM_THREADS'] = '1'

import matplotlib
import pandas as pd
matplotlib.use('agg')  # no need for tk

from autogluon.tabular import TabularPredictor, TabularDataset
import autogluon.core.metrics as metrics
from autogluon.tabular.version import __version__

from frameworks.shared.callee import call_run, result, measure_inference_times
from frameworks.shared.utils import Timer

from ag_utils.save_artifacts import ArtifactSaver
from ag_utils.zs_portfolio import get_hyperparameters_from_zeroshot_framework

log = logging.getLogger(__name__)


def _fit_autogluon(so_mitigation, predictor_para, fit_para):
    if so_mitigation is None:
        return _fit_autogluon_default(predictor_para, fit_para), None

    train_data = pd.read_parquet(fit_para.get('train_data'))
    label = predictor_para["label"]

    if so_mitigation == "proxy":
        from stacked_overfitting_mitigation.proxy_approaches import determine_stacked_overfitting, verify_stacking_settings
        use_stacking = determine_stacked_overfitting(train_data, label, predictor_para["problem_type"])
        fit_para = verify_stacking_settings(use_stacking, fit_para)
        return _fit_autogluon_default(predictor_para, fit_para), None

    # remove for other methods as they set train data manually
    fit_para.pop('train_data')

    if so_mitigation == "heuristic":
        from stacked_overfitting_mitigation.heuristic_approaches import no_holdout
        return no_holdout(train_data, label, predictor_para, fit_para), None

    from stacked_overfitting_mitigation.holdout_approaches import use_holdout
    if so_mitigation == "ho_select":
        mitigate_para = dict(select_on_holdout=True)
    elif so_mitigation == "ho_select_refit":
        mitigate_para = dict(refit_autogluon=True, select_on_holdout=True)
    elif so_mitigation == "ho_dynamic_stacking":
        mitigate_para = dict(refit_autogluon=True, dynamic_stacking=True)
    elif so_mitigation == "ho_ges_weights":
        mitigate_para = dict(refit_autogluon=True, ges_holdout=True)
    else:
        raise ValueError(f"Unknown SO mitigation: {so_mitigation}")

    predictor, ho_lb = use_holdout(train_data, label, predictor_para, fit_para, **mitigate_para)

    return predictor, ho_lb


def _fit_autogluon_default(predictor_para, fit_para):
    predictor = TabularPredictor(**predictor_para)
    predictor.fit(**fit_para)
    return predictor


def run(dataset, config):
    log.info(f"\n**** AutoGluon [v{__version__}] ****\n")
    log_pip_freeze()

    metrics_mapping = dict(
        acc=metrics.accuracy,
        auc=metrics.roc_auc,
        f1=metrics.f1,
        logloss=metrics.log_loss,
        mae=metrics.mean_absolute_error,
        mse=metrics.mean_squared_error,
        r2=metrics.r2,
        rmse=metrics.root_mean_squared_error,
    )

    perf_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if perf_metric is None:
        # TODO: figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        log.warning("Performance metric %s not supported.", config.metric)

    is_classification = config.type == 'classification'
    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
    presets = training_params.get("presets", [])
    so_mitigation = training_params.pop("so_mitigation", None)
    presets = presets if isinstance(presets, list) else [presets]
    if preset_with_refit_full := (set(presets) & {"good_quality", "high_quality"}):
        preserve = 0.9
        preset = next(iter(preset_with_refit_full))
        msg = (
            f"Detected `{preset}` preset, reducing `max_runtime_seconds` "
            f"from {config.max_runtime_seconds}s to "
            f"{preserve * config.max_runtime_seconds}s to account for `refit_full` "
            f"call after fit, which can take up to ~15% of total time. "
            "See https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.refit_full.html"
        )
        log.info(msg)
        config.max_runtime_seconds = preserve * config.max_runtime_seconds

    train_path, test_path = dataset.train.path, dataset.test.path
    label = dataset.target.name
    problem_type = dataset.problem_type

    models_dir = tempfile.mkdtemp() + os.sep  # passed to AG

    _zeroshot_framework = config.framework_params.get('_zeroshot_framework', None)
    if _zeroshot_framework is not None:
        _hyperparameters = get_hyperparameters_from_zeroshot_framework(
            zeroshot_framework=_zeroshot_framework,
            config=config,
        )
        if _hyperparameters is not None:
            training_params['hyperparameters'] = _hyperparameters

    with Timer() as training:
        predictor_para = dict(
            label=label,
            eval_metric=perf_metric.name,
            path=models_dir,
            problem_type=problem_type,
        )
        fit_para = dict(
            train_data=train_path,
            time_limit=config.max_runtime_seconds,
            **training_params
        )
        predictor, ho_leaderboard = _fit_autogluon(so_mitigation, predictor_para, fit_para)

    # FIXME how to save ho_leaderboard?
    artifact_saver = ArtifactSaver(predictor=predictor, config=config)

    log.info(f"Finished fit in {training.duration}s.")

    # Log failures
    if hasattr(predictor, 'get_model_failures'):  # version >0.8.2
        model_failures_df = predictor.get_model_failures()
        num_model_failures = len(model_failures_df)
        log.info(f'num_model_failures: {num_model_failures}')
        if num_model_failures > 0:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                log.info(model_failures_df)
    else:
        model_failures_df = None
    artifact_saver.cache_post_fit(model_failures_df=model_failures_df)

    # Persist model in memory that is going to be predicting to get correct inference latency
    # max_memory=0.4 will be future default: https://github.com/autogluon/autogluon/pull/3338
    predictor.persist_models('best', max_memory=0.4)

    def inference_time_classification(data: Union[str, pd.DataFrame]):
        return None, predictor.predict_proba(data, as_multiclass=True)

    def inference_time_regression(data: Union[str, pd.DataFrame]):
        return predictor.predict(data, as_pandas=False), None

    infer = inference_time_classification if is_classification else inference_time_regression
    inference_times = {}
    if config.measure_inference_time:
        inference_times["file"] = measure_inference_times(infer, dataset.inference_subsample_files)
        test_data = pd.read_parquet(dataset.test.path)
        inference_times["df"] = measure_inference_times(
            infer,
            [(1, test_data.sample(1, random_state=i)) for i in range(100)],
        )
        log.info(f"Finished inference time measurements.")

    test_data = TabularDataset(test_path)
    with Timer() as predict:
        predictions, probabilities = infer(test_data)
    if is_classification:
        predictions = probabilities.idxmax(axis=1).to_numpy()

    prob_labels = probabilities.columns.values.astype(str).tolist() if probabilities is not None else None
    log.info(f"Finished predict in {predict.duration}s.")

    _leaderboard_extra_info = config.framework_params.get('_leaderboard_extra_info', False)  # whether to get extra model info (very verbose)
    _leaderboard_test = config.framework_params.get('_leaderboard_test', False)  # whether to compute test scores in leaderboard (expensive)
    leaderboard_kwargs = dict(silent=True, extra_info=_leaderboard_extra_info)
    # Disabled leaderboard test data input by default to avoid long running computation, remove 7200s timeout limitation to re-enable
    if _leaderboard_test:
        leaderboard_kwargs['data'] = test_data

    leaderboard = predictor.leaderboard(**leaderboard_kwargs)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        log.info(leaderboard)

    num_models_trained = len(leaderboard)
    if predictor._trainer.model_best is not None:
        num_models_ensemble = len(predictor._trainer.get_minimum_model_set(predictor._trainer.model_best))
    else:
        num_models_ensemble = 1

    artifact_saver.cache_post_predict(leaderboard=leaderboard, test_data=test_data)
    shutil.rmtree(predictor.path, ignore_errors=True)

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  probabilities=probabilities,
                  probabilities_labels=prob_labels,
                  target_is_encoded=False,
                  models_count=num_models_trained,
                  models_ensemble_count=num_models_ensemble,
                  training_duration=training.duration,
                  predict_duration=predict.duration,
                  inference_times=inference_times,)


def log_pip_freeze():
    try:
        from pip._internal.operations import freeze
        pip_dependencies = freeze.freeze()
        log_pip_str = '\n===== pip freeze =====\n'
        for p in pip_dependencies:
            log_pip_str += f'{p}\n'
        log_pip_str += '======================\n'
        log.info(log_pip_str)
    except:
        pass


if __name__ == '__main__':
    call_run(run)
