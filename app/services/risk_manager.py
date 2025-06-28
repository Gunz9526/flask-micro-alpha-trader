import json
import os
from datetime import datetime, date
from typing import Dict, List, Optional
from flask import current_app
import threading

class InMemoryRiskManager:
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.max_position_size = current_app.config.get('RISK_MAX_POSITION_SIZE', 0.10)
        self.max_portfolio_exposure = current_app.config.get('RISK_MAX_PORTFOLIO_EXPOSURE', 0.80)
        self.max_positions = current_app.config.get('RISK_MAX_POSITIONS', 7)
        self.stop_loss_pct = current_app.config.get('RISK_STOP_LOSS_PCT', -0.05)
        self.take_profit_pct = current_app.config.get('RISK_TAKE_PROFIT_PCT', 0.10)
        self.volatility_target = current_app.config.get('RISK_VOLATILITY_TARGET', 0.015)
        self.max_daily_loss = current_app.config.get('RISK_MAX_DAILY_LOSS', -0.02)
        
        self._trading_halted = False
        self._daily_pnl = 0.0
        self._last_portfolio_value = None
        self._current_date = date.today()
        self._trade_count_today = 0
        self._max_daily_trades = 99999  # 임시 조치
        
        self.state_file = "risk_state.json"
        self._load_state()
        self._initialized = True
        
        current_app.logger.info(f"InMemory RiskManager 초기화 완료 - 거래중단: {self._trading_halted}")

    def _load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    
                    # 날짜 확인 - 새로운 날이면 리셋
                    saved_date = state.get('date')
                    if saved_date != str(self._current_date):
                        current_app.logger.info("새로운 날 - 리스크 상태 초기화")
                        self._reset_daily_state()
                    else:
                        self._trading_halted = state.get('trading_halted', False)
                        self._daily_pnl = state.get('daily_pnl', 0.0)
                        self._last_portfolio_value = state.get('last_portfolio_value')
                        self._trade_count_today = state.get('trade_count_today', 0)
                        
                        current_app.logger.info(f"리스크 상태 로드: PnL={self._daily_pnl:.4f}, 거래횟수={self._trade_count_today}")
        except Exception as e:
            current_app.logger.warning(f"리스크 상태 로드 실패, 초기화: {e}")
            self._reset_daily_state()

    def _save_state(self):
        try:
            state = {
                'date': str(self._current_date),
                'trading_halted': self._trading_halted,
                'daily_pnl': self._daily_pnl,
                'last_portfolio_value': self._last_portfolio_value,
                'trade_count_today': self._trade_count_today,
                'updated_at': datetime.now().isoformat()
            }
            
            temp_file = f"{self.state_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            os.replace(temp_file, self.state_file)
            current_app.logger.debug("리스크 상태 저장 완료")
            
        except Exception as e:
            current_app.logger.error(f"리스크 상태 저장 실패: {e}")

    def _reset_daily_state(self):
        self._trading_halted = False
        self._daily_pnl = 0.0
        self._trade_count_today = 0
        self._current_date = date.today()
        self._save_state()

    @property
    def is_trading_halted(self) -> bool:
        # 날짜 체크
        if date.today() != self._current_date:
            self._reset_daily_state()
        return self._trading_halted

    @is_trading_halted.setter
    def is_trading_halted(self, value: bool):
        self._trading_halted = value
        self._save_state()

    def check_position_risk(self, confidence: float, volatility: float) -> Dict:
        try:
            if self.is_trading_halted:
                return {"approved": False, "reason": "trading_halted"}
            
            if self._trade_count_today >= self._max_daily_trades:
                return {"approved": False, "reason": f"daily_trade_limit_exceeded ({self._max_daily_trades})"}
            
            volatility_factor = max(volatility / self.volatility_target, 1.0)
            base_size = self.max_position_size * confidence
            adjusted_size = base_size / volatility_factor
            position_size = min(adjusted_size, self.max_position_size)
            
            return {
                "approved": True, 
                "position_size": position_size,
                "volatility_factor": volatility_factor,
                "trades_today": self._trade_count_today
            }
            
        except Exception as e:
            current_app.logger.error(f"포지션 리스크 체크 오류: {e}")
            return {"approved": False, "reason": f"risk_check_error: {str(e)}"}

    def check_portfolio_risk(self, new_position_value: float, current_positions: List[Dict], portfolio_value: float) -> Dict:
        try:
            if len(current_positions) >= self.max_positions:
                return {"approved": False, "reason": f"max_positions_exceeded ({self.max_positions})"}
            
            total_exposure = sum(pos.get('market_value', 0) for pos in current_positions)
            new_exposure_ratio = (total_exposure + new_position_value) / portfolio_value
            
            if new_exposure_ratio > self.max_portfolio_exposure:
                return {
                    "approved": False, 
                    "reason": f"max_exposure_exceeded ({self.max_portfolio_exposure:.0%})",
                    "current_exposure": new_exposure_ratio
                }
            
            return {
                "approved": True,
                "portfolio_exposure": new_exposure_ratio,
                "available_positions": self.max_positions - len(current_positions)
            }
            
        except Exception as e:
            current_app.logger.error(f"포트폴리오 리스크 체크 오류: {e}")
            return {"approved": False, "reason": f"portfolio_risk_error: {str(e)}"}

    def should_close_position(self, position: Dict) -> Dict:
        try:
            unrealized_pnl_pct = position.get('unrealized_plpc', 0)
            
            if unrealized_pnl_pct <= self.stop_loss_pct:
                return {
                    "should_close": True, 
                    "reason": "stop_loss",
                    "close_ratio": 1.0,
                    "pnl_pct": unrealized_pnl_pct
                }
            
            if unrealized_pnl_pct >= self.take_profit_pct:
                return {
                    "should_close": True, 
                    "reason": "take_profit",
                    "close_ratio": 0.5,
                    "pnl_pct": unrealized_pnl_pct
                }
            
            return {"should_close": False, "pnl_pct": unrealized_pnl_pct}
            
        except Exception as e:
            current_app.logger.error(f"포지션 청산 체크 오류: {e}")
            return {"should_close": False, "error": str(e)}
    
    def check_daily_limits(self, portfolio_value: float) -> Dict:
        try:
            if self.is_trading_halted:
                return {
                    "approved": False, 
                    "reason": "trading_halted_due_to_daily_loss",
                    "daily_pnl": self._daily_pnl
                }
            
            self.update_daily_pnl(portfolio_value)
            
            if self._daily_pnl < self.max_daily_loss:
                self.is_trading_halted = True
                current_app.logger.critical(f"!!! 일일 최대 손실 한도 초과: {self._daily_pnl:.2%} < {self.max_daily_loss:.2%} !!!")
                return {
                    "approved": False, 
                    "reason": "daily_loss_limit_exceeded",
                    "daily_pnl": self._daily_pnl,
                    "limit": self.max_daily_loss
                }
            
            return {
                "approved": True,
                "daily_pnl": self._daily_pnl,
                "trades_today": self._trade_count_today,
                "trades_remaining": self._max_daily_trades - self._trade_count_today
            }
            
        except Exception as e:
            current_app.logger.error(f"일일 한도 체크 오류: {e}")
            return {"approved": False, "reason": f"daily_limit_error: {str(e)}"}

    def update_daily_pnl(self, portfolio_value: float):
        try:
            if self._last_portfolio_value is not None:
                pnl = (portfolio_value - self._last_portfolio_value) / self._last_portfolio_value
                self._daily_pnl = pnl
            
            self._last_portfolio_value = portfolio_value
            self._save_state()
            
        except Exception as e:
            current_app.logger.error(f"일일 PnL 업데이트 오류: {e}")

    def record_trade(self):
        self._trade_count_today += 1
        self._save_state()
        current_app.logger.info(f"거래 기록: 오늘 {self._trade_count_today}번째 거래")

    def reset_daily_limits(self):
        self._reset_daily_state()
        current_app.logger.info("일일 거래 한도 및 PnL 초기화 완료")

    def get_status(self) -> Dict:
        return {
            "trading_halted": self._trading_halted,
            "daily_pnl": self._daily_pnl,
            "daily_pnl_percent": f"{self._daily_pnl:.2%}" if self._daily_pnl else "0.00%",
            "trade_count_today": self._trade_count_today,
            "max_daily_trades": self._max_daily_trades,
            "trades_remaining": self._max_daily_trades - self._trade_count_today,
            "current_date": str(self._current_date),
            "limits": {
                "max_position_size": f"{self.max_position_size:.1%}",
                "max_portfolio_exposure": f"{self.max_portfolio_exposure:.1%}",
                "max_positions": self.max_positions,
                "max_daily_loss": f"{self.max_daily_loss:.1%}",
                "stop_loss": f"{self.stop_loss_pct:.1%}",
                "take_profit": f"{self.take_profit_pct:.1%}"
            }
        }

RiskManager = InMemoryRiskManager