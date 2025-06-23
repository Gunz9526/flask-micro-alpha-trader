import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from flask import current_app
from typing import Dict, Tuple, List
import joblib
import os
from datetime import datetime



class SimpleScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self
    
    def transform(self, X):
        return (X - self.mean_) / (self.std_ + 1e-8)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class BaseModel:
    def __init__(self):
        self.model = None
        self.scaler = SimpleScaler()
        self.is_trained = False
        self.min_data_points = 100
        self.features = [
            'returns_5d', 'returns_20d', 'rsi', 'bb_width',
            'volume_ratio', 'volatility_20d', 'momentum_score', 'trend_strength'
        ]

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['returns_1d'] = df['close'].pct_change()
        df['returns_5d'] = df['close'].pct_change(5)
        df['returns_20d'] = df['close'].pct_change(20)
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = (100 - (100 / (1 + rs)))

        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_width'] = (4 * std_20) / (sma_20 + 1e-8)

        df['volume_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-8)
        df['volatility_20d'] = df['returns_1d'].rolling(20).std()
        df['momentum_score'] = (df['close'] / df['close'].rolling(10).mean() - 1)
        
        sma_5 = df['close'].rolling(5).mean()
        df['trend_strength'] = (sma_5 - sma_20) / (sma_20 + 1e-8)
        
        df['target'] = df['close'].pct_change().shift(-1)
        return df

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df = self.calculate_features(df)
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True)
        df.fillna(0, inplace=True)

        if len(df) < self.min_data_points:
            raise ValueError(f"데이터 부족: {len(df)} < {self.min_data_points}")
        
        X = df[self.features].values
        y = df['target'].values
        
        X = self.scaler.fit_transform(X)
        y = np.clip(y, np.percentile(y, 2), np.percentile(y, 98))
        
        return X[:-1], y[:-1]

    def train(self, df: pd.DataFrame, override_params: Dict = None) -> Dict:
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> Dict:
        raise NotImplementedError

class LightGBMStrategy(BaseModel):
    def __init__(self, symbol:str, random_state=42):
        super().__init__()
        self.model_type = 'lgbm'
        self.symbol = symbol
        self.lgb_params = {
            'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
            'n_estimators': 500, 'num_leaves': 20, 'learning_rate': 0.05,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1,
            'lambda_l1': 0.1, 'lambda_l2': 0.1, 'verbose': -1,
            'random_state': random_state, 'n_jobs': -1, 'max_depth': 7, 'min_child_samples': 20
        }

    def train(self, df: pd.DataFrame, override_params: Dict = None) -> Dict:
        try:
            X, y = self.prepare_data(df)
            train_size = int(len(X) * 0.8)
            X_train, X_val, y_train, y_val = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
            
            optimize_params = self.lgb_params.copy()
            if override_params:
                optimize_params.update(override_params)
            else:
                params_path = os.path.join("best_params", f"{self.symbol}_LightGBMStrategy.json")
                if os.path.exists(params_path):
                    try:
                        with open(params_path, 'r') as f:
                            best_params = json.load(f)
                            optimize_params.update(best_params)
                            current_app.logger.info(f"[{self.symbol}] 파일에서 최적화된 LGBM 파라미터 로드")
                    except Exception as e:
                        current_app.logger.warning(f"[{self.symbol}] 파라미터 파일 로드 실패: {e}")

            optimize_params['random_state'] = self.lgb_params['random_state']
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            self.model = lgb.train(optimize_params, train_data,
                                   valid_sets=[val_data], callbacks=[lgb.early_stopping(20, verbose=False)])
            
            y_pred = self.model.predict(X_val)
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            direction_acc = np.mean(np.sign(y_val) == np.sign(y_pred))
            
            correlation = 0.0
            if len(y_val) > 1 and np.std(y_val) > 1e-8 and np.std(y_pred) > 1e-8:
                corr_matrix = np.corrcoef(y_val, y_pred)
                if not np.isnan(corr_matrix[0, 1]):
                    correlation = corr_matrix[0, 1]
            
            self.is_trained = True
            return {"status": "success", "direction_accuracy": float(direction_acc), 
                    "correlation": float(correlation), "rmse": float(rmse),
                    "validation_passed": bool(direction_acc > 0.5 and correlation > 0.0)}
        except Exception as e:
            return {"status": "error", "message": str(e), "rmse": float('inf')}

    def predict(self, df: pd.DataFrame) -> Dict:
        if not self.is_trained: return {"predicted_return": 0.0}
        try:
            df_features = self.calculate_features(df)
            latest_features_raw = df_features[self.features].iloc[-1:].values
            if np.isnan(latest_features_raw).any(): return {"predicted_return": 0.0}
            
            latest_features = self.scaler.transform(latest_features_raw)
            predicted_return = self.model.predict(latest_features)[0]
            
            return {"predicted_return": float(predicted_return), 
                    "latest_features": dict(zip(self.features, latest_features_raw[0]))}
        except Exception as e:
            current_app.logger.error(f"LGBM 예측 실패: {e}", exc_info=True)
            return {"predicted_return": 0.0}

class XGBoostStrategy(BaseModel):
    def __init__(self, symbol: str, random_state=42):
        super().__init__()
        self.model_type = 'xgb'
        self.symbol = symbol
        self.xgb_params = {
            'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'seed': random_state,
            'learning_rate': 0.05, 'max_depth': 5,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'n_jobs': -1
        }

    def train(self, df: pd.DataFrame, override_params: Dict = None) -> Dict:
        try:
            X, y = self.prepare_data(df)
            
            train_size = int(len(X) * 0.8)
            X_train, X_val, y_train, y_val = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            optimize_params = self.xgb_params.copy()
            if override_params:
                optimize_params.update(override_params)
            else:
                params_path = os.path.join("best_params", f"{self.symbol}_XGBoostStrategy.json")
                if os.path.exists(params_path):
                    try:
                        with open(params_path, 'r') as f:
                            best_params = json.load(f)
                            optimize_params.update(best_params)
                            current_app.logger.info(f"[{self.symbol}] 파일에서 최적화된 XGB 파라미터 로드")
                    except Exception as e:
                        current_app.logger.warning(f"[{self.symbol}] 파라미터 파일 로드 실패: {e}")
            
            optimize_params['seed'] = self.xgb_params['seed']
            
            self.model = xgb.train(
                params=optimize_params,
                dtrain=dtrain,
                num_boost_round=300,
                evals=[(dval, 'validation')],
                early_stopping_rounds=20,
                verbose_eval=False
            )
            
            y_pred = self.model.predict(dval)
            
            if np.isnan(y_pred).any():
                raise ValueError("XGBoost 모델이 NaN 값을 예측했습니다.")

            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            direction_acc = np.mean(np.sign(y_val) == np.sign(y_pred))
            
            correlation = 0.0
            if len(y_val) > 1 and np.std(y_val) > 1e-8 and np.std(y_pred) > 1e-8:
                corr_matrix = np.corrcoef(y_val, y_pred)
                if not np.isnan(corr_matrix[0, 1]):
                    correlation = corr_matrix[0, 1]

            self.is_trained = True
            return {"status": "success", "direction_accuracy": float(direction_acc), 
                    "correlation": float(correlation), "rmse": float(rmse),
                    "validation_passed": bool(direction_acc > 0.5 and correlation > 0.0 and np.isfinite(rmse))}
        except Exception as e:
            current_app.logger.error(f"XGBoost 학습 실패: {e}", exc_info=True)
            return {"status": "error", "message": str(e), "rmse": float('inf')}

    def predict(self, df: pd.DataFrame) -> Dict:
        if not self.is_trained: return {"predicted_return": 0.0}
        try:
            df_features = self.calculate_features(df)
            latest_features_raw = df_features[self.features].iloc[-1:].values
            if np.isnan(latest_features_raw).any(): return {"predicted_return": 0.0}
            
            latest_features = self.scaler.transform(latest_features_raw)
            
            dpredict = xgb.DMatrix(latest_features)
            predicted_return = self.model.predict(dpredict)[0]
            
            return {"predicted_return": float(predicted_return), 
                    "latest_features": dict(zip(self.features, latest_features_raw[0]))}
        except Exception as e:
            current_app.logger.error(f"XGBoost 예측 실패: {e}", exc_info=True)
            return {"predicted_return": 0.0}

class AIService:
    def __init__(self):
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
        self.model_classes = {
            'lgbm_1': {'class': LightGBMStrategy, 'random_state': 42},
            'lgbm_2': {'class': LightGBMStrategy, 'random_state': 84},
            'xgb_1': {'class': XGBoostStrategy, 'random_state': 42},
            'xgb_2': {'class': XGBoostStrategy, 'random_state': 84}
        }

    def train_strategy(self, symbol: str, bars_df: pd.DataFrame, best_params_map: Dict = None) -> Dict:
        if best_params_map is None:
            best_params_map = {}
            
        current_app.logger.info(f"[{symbol}] AI 하이브리드 앙상블 모델 학습 시작")
        final_results = {}
        
        current_app.logger.info(f"[{symbol}] 기존 모델 파일 정리 시작...")
        for model_name in self.model_classes.keys():
            model_path = os.path.join(self.models_dir, f"{symbol}_{model_name}.pkl")
            scaler_path = os.path.join(self.models_dir, f"{symbol}_{model_name}_scaler.pkl")
            if os.path.exists(model_path): os.remove(model_path)
            if os.path.exists(scaler_path): os.remove(scaler_path)

        for model_name, config in self.model_classes.items():
            StrategyClass = config['class']
            random_state = config['random_state']
            strategy = StrategyClass(symbol=symbol, random_state=random_state)
            
            override_params = best_params_map.get(strategy.model_type)
            # current_app.logger.info(f"[{symbol}] {model_name} 모델 학습 시작 (랜덤 시드: {random_state}, 파라미터: {override_params})")
            result = strategy.train(bars_df, override_params=override_params)
            final_results[model_name] = result
            
            if result.get("status") == "success" and result.get("validation_passed"):
                joblib.dump(strategy.model, os.path.join(self.models_dir, f"{symbol}_{model_name}.pkl"))
                joblib.dump(strategy.scaler, os.path.join(self.models_dir, f"{symbol}_{model_name}_scaler.pkl"))
        
        valid_results = [r for r in final_results.values() if r.get("validation_passed")]
        if not valid_results:
            current_app.logger.warning(f"[{symbol}] 모든 앙상블 모델 검증 실패")
            return {"status": "failure", "reason": "all_models_failed_validation"}

        avg_accuracy = np.mean([r['direction_accuracy'] for r in valid_results])
        current_app.logger.info(f"[{symbol}] 앙상블 모델 저장 완료 (성공: {len(valid_results)}/{len(self.model_classes)}, 평균 정확도: {avg_accuracy:.3f})")
        return {"status": "success", "average_accuracy": float(avg_accuracy), "successful_models": len(valid_results)}

    def get_trading_signal(self, symbol: str, bars_df: pd.DataFrame) -> Dict:
        predictions = []
        for model_name, config in self.model_classes.items():
            model_path = os.path.join(self.models_dir, f"{symbol}_{model_name}.pkl")
            scaler_path = os.path.join(self.models_dir, f"{symbol}_{model_name}_scaler.pkl")
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                StrategyClass = config['class']
                random_state = config['random_state']
                strategy = StrategyClass(symbol=symbol, random_state=random_state)
                strategy.model = joblib.load(model_path)
                strategy.scaler = joblib.load(scaler_path)
                strategy.is_trained = True
                prediction = strategy.predict(bars_df)
                predictions.append(prediction)

        if not predictions:
            return {"signal": "HOLD", "reason": "no_models"}

        predicted_returns = [p['predicted_return'] for p in predictions]
        avg_return = np.mean(predicted_returns)
        std_return = np.std(predicted_returns)
        
        confidence = 1.0 - min(std_return / 0.01, 0.9) if std_return > 0 else 0.95

        current_app.logger.info(f"[{symbol}] 앙상블 예측 상세: 개별 예측값={predicted_returns}, 평균={avg_return:.4f}, 표준편차={std_return:.4f}, 최종 신뢰도={confidence:.3f}")

        buy_threshold = current_app.config.get('AI_SIGNAL_BUY_THRESHOLD', 0.005)
        sell_threshold = current_app.config.get('AI_SIGNAL_SELL_THRESHOLD', -0.005)
        confidence_threshold = current_app.config.get('AI_CONFIDENCE_THRESHOLD', 0.6)

        if avg_return > buy_threshold and confidence > confidence_threshold: signal = "BUY"
        elif avg_return < -0.005 and confidence > confidence_threshold: signal = "SELL"
        else: signal = "HOLD"

        return {"signal": signal, "confidence": confidence, "predicted_return": avg_return,
                "latest_features": predictions[0].get('latest_features', {}),
                "reason": "ensemble_prediction", "timestamp": datetime.now().isoformat()}