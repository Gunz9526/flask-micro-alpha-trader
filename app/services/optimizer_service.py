import os
import optuna
import numpy as np
import pandas as pd
from flask import current_app
import json
from .ai_service import LightGBMStrategy, XGBoostStrategy

class HyperparameterOptimizer:
    def __init__(self, symbol: str, bars_df: pd.DataFrame):
        self.symbol = symbol
        self.bars_df = bars_df
        self.params_dir = "best_params"
        os.makedirs(self.params_dir, exist_ok=True)

    def _objective(self, trial, model_strategy):
        if isinstance(model_strategy, LightGBMStrategy):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, 50),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 40),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            }
        elif isinstance(model_strategy, XGBoostStrategy):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'subsample': trial.suggest_float('subsample', 0.5, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            }
        else:
            return float('inf')

        result = model_strategy.train(self.bars_df, override_params=params)
        
        return result.get('rmse', float('inf'))

    def optimize(self, model_strategy, n_trials=100):
        model_name = model_strategy.__class__.__name__
        current_app.logger.info(f"[{self.symbol}] {model_name} 최적화 시작 ({n_trials}회 시도)")
        
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self._objective(trial, model_strategy), n_trials=n_trials, timeout=600)

        best_params = study.best_params
        current_app.logger.info(f"[{self.symbol}] {model_name} 최적화 완료. Best RMSE: {study.best_value:.4f}")
        
        params_path = os.path.join(self.params_dir, f"{self.symbol}_{model_name}.json")
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        
        return {"best_params": best_params, "best_rmse": study.best_value}