import logging

import numpy as np
import pandas as pd

from ..._utils import score_with_y_pred_proba
from ....scheduler.seq_scheduler import LocalSequentialScheduler
import autogluon.core as ag  # FIXME: Don't do absolute import

logger = logging.getLogger(__name__)


# FIXME: Only works for binary at present
def compute_optimal_oof_noise(X, y, X_val, y_val, oof_pred_proba_init, val_pred_proba, problem_type, metric, sample_weight=None, sample_weight_val=None, quantile_levels=None):
    from autogluon.tabular.models import RFModel  # FIXME: RFModel creates circular dependence on Tabular

    score_oof_kwargs = dict(y=y, problem_type=problem_type, metric=metric, sample_weight=sample_weight, quantile_levels=quantile_levels)
    score_val_kwargs = dict(y=y_val, problem_type=problem_type, metric=metric, sample_weight=sample_weight_val, quantile_levels=quantile_levels)

    l1_score = score_with_y_pred_proba(y_pred_proba=oof_pred_proba_init, **score_oof_kwargs)
    l1_score_val = score_with_y_pred_proba(y_pred_proba=val_pred_proba, **score_val_kwargs)

    print(l1_score)
    print(l1_score_val)

    noise_init = np.random.rand(len(oof_pred_proba_init))

    X_val_l2 = pd.concat([X_val.reset_index(drop=True), val_pred_proba], axis=1)  # FIXME: Ensure unique col names

    @ag.args(noise_scale=ag.space.Real(0, 0.5, default=0))
    def train_fn(args, reporter):
        noise_scale = args.noise_scale
        # for noise_scale in [0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        rf = RFModel(path='', name='', problem_type=problem_type, eval_metric=metric, hyperparameters={'n_estimators': 50})
        # rf = KNNModel(path='', name='', problem_type=self.problem_type, eval_metric=self.eval_metric, hyperparameters={'weights': 'distance', 'n_neighbors': 200})
        oof_pred_proba = oof_pred_proba_init.copy()
        noise = noise_init
        noise = noise * noise_scale * 2
        noise = noise - np.mean(noise)
        oof_pred_proba[0] += noise

        X_l2 = pd.concat([X.reset_index(drop=True), oof_pred_proba], axis=1)  # FIXME: Ensure unique col names

        rf.fit(X=X_l2, y=y)
        l2_oof_pred_proba = rf.get_oof_pred_proba(X=X_l2, y=y)
        l2_val_pred_proba = rf.predict_proba(X=X_val_l2)
        l2_score = score_with_y_pred_proba(y_pred_proba=l2_oof_pred_proba, **score_oof_kwargs)
        l2_score_val = score_with_y_pred_proba(y_pred_proba=l2_val_pred_proba, **score_val_kwargs)

        l1_score_val_noise = score_with_y_pred_proba(y_pred_proba=oof_pred_proba, **score_oof_kwargs)
        print()
        print(f'noise: {noise_scale}')

        score_diff_l1_noise = l1_score_val_noise - l1_score
        score_diff = l2_score - l1_score
        score_diff_val = l2_score_val - l1_score_val

        print(f'oof_noise_diff: {score_diff_l1_noise}')
        print(f'oof           : {score_diff}')
        print(f'val           : {score_diff_val}')
        # print(score_diff_val)

        noise_score = score_diff_val - abs(score_diff_l1_noise * 0.1)
        print(f'noise score   : {noise_score}')

        reporter(epoch=1, accuracy=noise_score)

    scheduler = LocalSequentialScheduler(
        train_fn,
        resource={'num_cpus': 'all', 'num_gpus': 0},
        num_trials=30,
        reward_attr='accuracy',
        time_attr='epoch',
        checkpoint=None
    )

    scheduler.run()

    print('result:')
    print(scheduler.get_best_config())
    print(scheduler.get_best_reward())

    oof_noise_scale = scheduler.get_best_config()['noise_scale']

    oof_noise = noise_init
    oof_noise = oof_noise * oof_noise_scale * 2
    oof_noise = oof_noise - np.mean(oof_noise)

    return oof_noise_scale, oof_noise
