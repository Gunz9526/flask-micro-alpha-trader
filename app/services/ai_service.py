import numpy as np
import pandas as pd
import lightgbm as lgb
from flask import current_app
from typing import Dict, Tuple
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

class AIStrategy:    
    def __init__(self):
        self.model = None
        self.scaler = SimpleScaler()
        self.is_trained = False
        self.min_data_points = 50
        
        self.features = [
            'returns_1d', 'returns_5d', 'returns_20d',
            'rsi', 'bb_position', 'volume_ratio',
            'volatility_20d', 'momentum_score'
        ]
        
        self.lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 8,
            'learning_rate': 0.15,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': 1,
            'max_depth': 4,
            'min_data_in_leaf': 10
        }
        
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            
            df['returns_1d'] = df['close'].pct_change()
            df['returns_5d'] = df['close'].pct_change(5)
            df['returns_20d'] = df['close'].pct_change(20)
            
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-8)
            df['rsi'] = (100 - (100 / (1 + rs))) / 100
            
            sma_20 = df['close'].rolling(20).mean()
            std_20 = df['close'].rolling(20).std()
            df['bb_position'] = (df['close'] - sma_20) / (2 * std_20 + 1e-8)
            
            df['volume_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-8)
            
            df['volatility_20d'] = df['returns_1d'].rolling(20).std()
            
            df['momentum_score'] = (df['close'] / df['close'].rolling(10).mean() - 1)
            
            df['target'] = df['returns_1d'].shift(-1)
            
            return df
            
        except Exception as e:
            current_app.logger.error(f"피처 계산 오류: {e}")
            return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df = self.calculate_features(df)
        df = df.dropna()
        
        if len(df) < self.min_data_points:
            raise ValueError(f"데이터 부족: {len(df)} < {self.min_data_points}")
        
        X = df[self.features].values
        y = df['target'].values
        
        X = self.scaler.fit_transform(X)
        
        y = np.clip(y, np.percentile(y, 5), np.percentile(y, 95))
        
        return X[:-1], y[:-1]
    
    def train_model(self, df: pd.DataFrame) -> Dict:
        try:
            X, y = self.prepare_data(df)
            
            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            self.model = lgb.train(
                self.lgb_params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data, val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=10, verbose=False),
                    lgb.log_evaluation(0)
                ]
            )
            
            y_pred = self.model.predict(X_val)
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            direction_acc = np.mean(np.sign(y_val) == np.sign(y_pred))
            
            self.is_trained = True
            
            return {
                "status": "success",
                "rmse": float(rmse),
                "direction_accuracy": float(direction_acc),
                "training_samples": int(len(X_train)),
                "validation_passed": bool(direction_acc > 0.52)
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def predict(self, df: pd.DataFrame) -> Dict:
        if not self.is_trained:
            return {"signal": "HOLD", "confidence": 0.0, "reason": "model_not_trained"}
        
        try:
            df_features = self.calculate_features(df)
            latest_features = df_features[self.features].iloc[-1:].values
            
            if np.isnan(latest_features).any():
                return {"signal": "HOLD", "confidence": 0.0, "reason": "missing_features"}
            
            latest_features = self.scaler.transform(latest_features)
            predicted_return = self.model.predict(latest_features)[0]
            
            confidence = min(abs(predicted_return) * 10, 0.8)
            
            if predicted_return > 0.015 and confidence > 0.65:
                signal = "BUY"
            elif predicted_return < -0.015 and confidence > 0.65:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            return {
                "signal": signal,
                "confidence": confidence,
                "predicted_return": predicted_return,
                "reason": "model_prediction"
            }
            
        except Exception as e:
            return {"signal": "HOLD", "confidence": 0.0, "reason": str(e)}

class AIService:
    def __init__(self):
        self.strategy = AIStrategy()
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
    
    def train_strategy(self, symbol: str, bars_df: pd.DataFrame) -> Dict:
        current_app.logger.info(f"{symbol} AI 모델 학습 시작")
        
        try:
            result = self.strategy.train_model(bars_df)
            
            if result.get("status") == "success" and result.get("validation_passed"):
                model_path = os.path.join(self.models_dir, f"{symbol}_model.pkl")
                scaler_path = os.path.join(self.models_dir, f"{symbol}_scaler.pkl")
                
                joblib.dump(self.strategy.model, model_path)
                joblib.dump(self.strategy.scaler, scaler_path)
                
                current_app.logger.info(f"{symbol} 모델 저장 완료 (정확도: {result['direction_accuracy']:.3f})")
            else:
                current_app.logger.warning(f"{symbol} 모델 검증 실패")
            
            return result
            
        except Exception as e:
            current_app.logger.error(f"{symbol} 학습 실패: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_trading_signal(self, symbol: str, bars_df: pd.DataFrame) -> Dict:
        try:
            model_path = os.path.join(self.models_dir, f"{symbol}_model.pkl")
            scaler_path = os.path.join(self.models_dir, f"{symbol}_scaler.pkl")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.strategy.model = joblib.load(model_path)
                self.strategy.scaler = joblib.load(scaler_path)
                self.strategy.is_trained = True
            else:
                return {
                    "symbol": symbol,
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "reason": "no_model"
                }
            
            result = self.strategy.predict(bars_df)
            result["symbol"] = symbol
            result["timestamp"] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            return {
                "symbol": symbol,
                "signal": "HOLD",
                "confidence": 0.0,
                "reason": str(e)
            }
