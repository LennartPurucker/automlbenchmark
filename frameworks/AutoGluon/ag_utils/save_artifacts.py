from __future__ import annotations

import logging
import os
import shutil

import pandas as pd

from autogluon.common.savers import save_pd, save_pkl
from autogluon.tabular import TabularPredictor

from frameworks.shared.callee import touch
from frameworks.shared.utils import zip_path

from ag_utils.calc_infer_speed import get_infer_speed_real
from ag_utils.zs_artifact import get_zeroshot_artifact

log = logging.getLogger(__name__)


def get_save_path(config, suffix: str, create_dir: bool = True, as_dir: bool = False) -> str:
    path = os.path.join(config.output_dir, suffix)
    if create_dir:
        touch(path, as_dir=as_dir)
    return path


class ArtifactSaver:
    def __init__(self, predictor: TabularPredictor, config):
        self.predictor = predictor
        self.config = config
        self.artifacts = self.config.framework_params.get('_save_artifacts', ['leaderboard', 'model_failures'])

    @property
    def path_leaderboard(self) -> str:
        return get_save_path(self.config, "leaderboard.csv")

    @property
    def path_ho_leaderboard(self) -> str:
        return get_save_path(self.config, "ho_leaderboard.csv")

    @property
    def path_ho_oof_importance(self) -> str:
        return get_save_path(self.config, "ho_oof_importance.csv")

    @property
    def path_info_dir(self) -> str:
        return get_save_path(self.config, "info", as_dir=True)

    @property
    def path_info(self) -> str:
        return os.path.join(self.path_info_dir, "info.pkl")

    @property
    def path_file_sizes(self) -> str:
        return os.path.join(self.path_info_dir, "file_sizes.csv")

    @property
    def path_model_failures(self) -> str:
        return get_save_path(self.config, "model_failures.csv")

    @property
    def path_infer_speed(self) -> str:
        return get_save_path(self.config, "infer_speed.csv")

    @property
    def path_zeroshot_dir(self) -> str:
        return get_save_path(self.config, 'zeroshot', as_dir=True)

    @property
    def path_zeroshot(self) -> str:
        return os.path.join(self.path_zeroshot_dir, "zeroshot_metadata.pkl")

    @property
    def path_predictor(self) -> str:
        return get_save_path(self.config, "models.zip")

    def save_leaderboard(self, leaderboard: pd.DataFrame):
        save_pd.save(path=self.path_leaderboard, df=leaderboard)

    def save_ho_leaderboard(self, ho_leaderboard: pd.DataFrame):
        save_pd.save(path=self.path_ho_leaderboard, df=ho_leaderboard)

    def save_ho_oof_importance(self, ho_oof_importance: pd.DataFrame):
        save_pd.save(path=self.path_ho_oof_importance, df=ho_oof_importance)

    def save_info(self):
        ag_size_df = self.predictor.get_size_disk_per_file().to_frame().reset_index(names='file')
        save_pd.save(path=self.path_file_sizes, df=ag_size_df)
        ag_info = self.predictor.info()
        save_pkl.save(path=self.path_info, object=ag_info)

    def save_model_failures(self, model_failures_df: pd.DataFrame):
        save_pd.save(path=self.path_model_failures, df=model_failures_df)

    def save_infer_speed(self, test_data: pd.DataFrame):
        infer_speed_df = get_infer_speed_real(predictor=self.predictor, test_data=test_data)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            log.info(infer_speed_df)
        save_pd.save(path=self.path_infer_speed, df=infer_speed_df)

    def save_zeroshot(self, test_data: pd.DataFrame):
        zeroshot_dict = get_zeroshot_artifact(predictor=self.predictor, test_data=test_data)
        save_pkl.save(path=self.path_zeroshot, object=zeroshot_dict)

    def save_predictor(self, delete_utils: bool = True):
        if delete_utils:
            shutil.rmtree(os.path.join(self.predictor.path, "utils"), ignore_errors=True)
        zip_path(self.predictor.path, self.path_predictor)

    def cache_post_fit(self, model_failures_df: pd.DataFrame | None, ho_leaderboard: pd.DataFrame | None = None,
                       ho_oof_importance: pd.DataFrame | None = None):
        artifacts = self.artifacts
        try:
            if 'model_failures' in artifacts and model_failures_df is not None:
                self.save_model_failures(model_failures_df=model_failures_df)

            if 'ho_leaderboard' in artifacts and ho_leaderboard is not None:
                self.save_ho_leaderboard(ho_leaderboard=ho_leaderboard)

            if 'ho_oof_importance' in artifacts and ho_oof_importance is not None:
                self.save_ho_oof_importance(ho_oof_importance=ho_oof_importance)
        except:
            log.warning("Error when saving post-fit artifacts.", exc_info=True)

    def cache_post_predict(self, leaderboard: pd.DataFrame, test_data: pd.DataFrame):
        artifacts = self.artifacts
        try:
            if 'leaderboard' in artifacts:
                self.save_leaderboard(leaderboard=leaderboard)

            if 'info' in artifacts:
                self.save_info()

            if 'infer_speed' in artifacts:
                self.save_infer_speed(test_data=test_data)

            if 'zeroshot' in artifacts:
                self.save_zeroshot(test_data=test_data)

            if 'models' in artifacts:
                self.save_predictor(delete_utils=True)
        except Exception:
            log.warning("Error when saving post-predict artifacts.", exc_info=True)


def save_artifacts(predictor: TabularPredictor, leaderboard: pd.DataFrame, config, test_data: pd.DataFrame,
                   ho_leaderboard: pd.DataFrame, ho_oof_importance: pd.DataFrame):
    artifacts = config.framework_params.get('_save_artifacts', ['leaderboard'])
    try:
        if 'leaderboard' in artifacts:
            save_pd.save(path=get_save_path(config, "leaderboard.csv"), df=leaderboard)

        if 'ho_leaderboard' in artifacts:
            save_pd.save(path=get_save_path(config, "ho_leaderboard.csv"), df=ho_leaderboard)

        if 'ho_oof_importance' in artifacts:
            save_pd.save(path=get_save_path(config, "ho_oof_importance.csv"), df=ho_oof_importance)

        if 'info' in artifacts:
            info_path = get_save_path(config, 'info', as_dir=True)
            ag_info = predictor.info()
            ag_size_df = predictor.get_size_disk_per_file().to_frame().reset_index(names='file')
            save_pd.save(path=os.path.join(info_path, "file_sizes.csv"), df=ag_size_df)
            save_pkl.save(path=os.path.join(info_path, "info.pkl"), object=ag_info)

        if 'infer_speed' in artifacts:
            infer_speed_df = get_infer_speed_real(predictor=predictor, test_data=test_data)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                log.info(infer_speed_df)
            save_pd.save(path=get_save_path(config, "infer_speed.csv"), df=infer_speed_df)

        if 'zeroshot' in artifacts:
            zeroshot_path = get_save_path(config, 'zeroshot', as_dir=True)
            zeroshot_dict = get_zeroshot_artifact(predictor=predictor, test_data=test_data)
            save_pkl.save(path=os.path.join(zeroshot_path, "zeroshot_metadata.pkl"), object=zeroshot_dict)

        if 'models' in artifacts:
            shutil.rmtree(os.path.join(predictor.path, "utils"), ignore_errors=True)
            zip_path(predictor.path, get_save_path(config, "models.zip"))
    except Exception:
        log.warning("Error when saving artifacts.", exc_info=True)
