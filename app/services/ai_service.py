import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
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
        df = self.calculate_features(df).dropna()
        if len(df) < self.min_data_points:
            raise ValueError(f"데이터 부족: {len(df)} < {self.min_data_points}")
        
        X = df[self.features].values
        y = df['target'].values
        
        X = self.scaler.fit_transform(X)
        y = np.clip(y, np.percentile(y, 2), np.percentile(y, 98))
        
        return X[:-1], y[:-1]

    def train(self, df: pd.DataFrame) -> Dict:
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> Dict:
        raise NotImplementedError

class LightGBMStrategy(BaseModel):
    def __init__(self, random_state=42):
        super().__init__()
        self.lgb_params = {
            'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
            'n_estimators': 500, 'num_leaves': 20, 'learning_rate': 0.05,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1,
            'lambda_l1': 0.1, 'lambda_l2': 0.1, 'verbose': -1,
            'random_state': random_state, 'n_jobs': -1, 'max_depth': 7, 'min_child_samples': 20
        }

    def train(self, df: pd.DataFrame) -> Dict:
        try:
            X, y = self.prepare_data(df)
            train_size = int(len(X) * 0.8)
            X_train, X_val, y_train, y_val = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            self.model = lgb.train(self.lgb_params, train_data,
                                   valid_sets=[val_data], callbacks=[lgb.early_stopping(20, verbose=False)])
            
            y_pred = self.model.predict(X_val)
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            direction_acc = np.mean(np.sign(y_val) == np.sign(y_pred))
            correlation = np.corrcoef(y_val, y_pred)[0, 1] if len(y_val) > 1 else 0
            
            self.is_trained = True
            return {"status": "success", "direction_accuracy": float(direction_acc), 
                    "correlation": float(correlation), "rmse": float(rmse),
                    "validation_passed": bool(direction_acc > 0.5 and correlation > 0.0)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

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
        except:
            return {"predicted_return": 0.0}

class XGBoostStrategy(BaseModel):
    def __init__(self, random_state=42):
        super().__init__()
        self.xgb_params = {
            'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'seed': random_state,
            'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 5,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'n_jobs': -1
        }

    def train(self, df: pd.DataFrame) -> Dict:
        try:
            X, y = self.prepare_data(df)
            train_size = int(len(X) * 0.8)
            X_train, X_val, y_train, y_val = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
            
            self.model = xgb.XGBRegressor(**self.xgb_params)
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
            
            y_pred = self.model.predict(X_val)
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            direction_acc = np.mean(np.sign(y_val) == np.sign(y_pred))
            correlation = np.corrcoef(y_val, y_pred)[0, 1] if len(y_val) > 1 else 0
            
            self.is_trained = True
            return {"status": "success", "direction_accuracy": float(direction_acc), 
                    "correlation": float(correlation), "rmse": float(rmse),
                    "validation_passed": bool(direction_acc > 0.5 and correlation > 0.0)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

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
        except:
            return {"predicted_return": 0.0}

class AIService:
    def __init__(self):
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
        self.model_prototypes = {
            'lgbm_1': LightGBMStrategy(random_state=42),
            'lgbm_2': LightGBMStrategy(random_state=84),
            'xgb_1': XGBoostStrategy(random_state=42)
        }

    def train_strategy(self, symbol: str, bars_df: pd.DataFrame) -> Dict:
        current_app.logger.info(f"[{symbol}] AI 하이브리드 앙상블 모델 학습 시작")
        final_results = {}
        for model_name, strategy in self.model_prototypes.items():
            result = strategy.train(bars_df)
            final_results[model_name] = result
            if result.get("status") == "success" and result.get("validation_passed"):
                joblib.dump(strategy.model, os.path.join(self.models_dir, f"{symbol}_{model_name}.pkl"))
                joblib.dump(strategy.scaler, os.path.join(self.models_dir, f"{symbol}_{model_name}_scaler.pkl"))
        
        valid_results = [r for r in final_results.values() if r.get("validation_passed")]
        if not valid_results:
            current_app.logger.warning(f"[{symbol}] 모든 앙상블 모델 검증 실패")
            return {"status": "failure", "reason": "all_models_failed_validation"}

        avg_accuracy = np.mean([r['direction_accuracy'] for r in valid_results])
        current_app.logger.info(f"[{symbol}] 앙상블 모델 저장 완료 (성공: {len(valid_results)}/{len(self.model_prototypes)}, 평균 정확도: {avg_accuracy:.3f})")
        return {"status": "success", "average_accuracy": float(avg_accuracy), "successful_models": len(valid_results)}

    def get_trading_signal(self, symbol: str, bars_df: pd.DataFrame) -> Dict:
        predictions = []
        for model_name, strategy_prototype in self.model_prototypes.items():
            model_path = os.path.join(self.models_dir, f"{symbol}_{model_name}.pkl")
            scaler_path = os.path.join(self.models_dir, f"{symbol}_{model_name}_scaler.pkl")
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                strategy = strategy_prototype
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

        current_app.logger.info(f"[{symbol}] 앙상블 예측 상세: "
                                f"개별 예측값={predicted_returns}, "
                                f"평균={avg_return:.4f}, 표준편차={std_return:.4f}, "
                                f"최종 신뢰도={confidence:.3f}")

        buy_threshold = current_app.config.get('AI_SIGNAL_BUY_THRESHOLD', 0.005)
        sell_threshold = current_app.config.get('AI_SIGNAL_SELL_THRESHOLD', -0.005)
        confidence_threshold = current_app.config.get('AI_CONFIDENCE_THRESHOLD', 0.6)

        if avg_return > buy_threshold and confidence > confidence_threshold: signal = "BUY"
        elif avg_return < sell_threshold and confidence > confidence_threshold: signal = "SELL"
        else: signal = "HOLD"

        return {"signal": signal, "confidence": confidence, "predicted_return": avg_return,
                "latest_features": predictions[0].get('latest_features', {}),
                "reason": "ensemble_prediction", "timestamp": datetime.now().isoformat()}