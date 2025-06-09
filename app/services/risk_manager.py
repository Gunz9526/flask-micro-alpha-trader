from typing import Dict, List, Optional
from flask import current_app
import numpy as np
from datetime import datetime, timedelta

class RiskManager:
    def __init__(self):
        self.max_portfolio_risk = 0.02
        self.max_position_size = 0.05
        self.max_sector_exposure = 0.3
        self.max_correlation = 0.7
        self.stop_loss = 0.05
        self.max_drawdown = 0.1
        
        self.daily_pnl = []
        self.positions_history = []
        self.is_trading_halted = False
        
    def check_position_risk(self, symbol: str, signal: str, confidence: float, 
                          current_price: float, portfolio_value: float) -> Dict:
        try:
            base_size = self.max_position_size * confidence
            
            volatility_factor = self._get_volatility_factor(symbol)
            adjusted_size = base_size / max(volatility_factor, 1.0)
            
            position_size = min(adjusted_size, self.max_position_size)
            
            investment_amount = portfolio_value * position_size
            shares = int(investment_amount / current_price)
            
            return {
                "approved": True,
                "shares": shares,
                "position_size": position_size,
                "investment_amount": shares * current_price,
                "risk_adjusted": volatility_factor > 1.5
            }
            
        except Exception as e:
            current_app.logger.error(f"포지션 리스크 체크 오류: {e}")
            return {"approved": False, "reason": str(e)}
    
    def check_portfolio_risk(self, new_position: Dict, current_positions: List[Dict]) -> Dict:
        try:
            total_exposure = sum(pos.get('market_value', 0) for pos in current_positions)
            new_exposure = new_position.get('investment_amount', 0)
            
            if (total_exposure + new_exposure) / new_position.get('portfolio_value', 1) > 0.8:
                return {"approved": False, "reason": "max_exposure_exceeded"}
            
            if len(current_positions) >= 10:
                return {"approved": False, "reason": "max_positions_exceeded"}
            
            symbol = new_position.get('symbol')
            if any(pos.get('symbol') == symbol for pos in current_positions):
                return {"approved": False, "reason": "duplicate_position"}
            
            return {"approved": True}
            
        except Exception as e:
            return {"approved": False, "reason": str(e)}
    
    def check_daily_limits(self, portfolio_value: float) -> Dict:
        if self.is_trading_halted:
            return {"approved": False, "reason": "trading_halted"}
        
        if len(self.daily_pnl) > 0:
            today_pnl = self.daily_pnl[-1] if self.daily_pnl else 0
            if today_pnl / portfolio_value < -self.max_portfolio_risk:
                self.is_trading_halted = True
                return {"approved": False, "reason": "daily_loss_limit_exceeded"}
        
        return {"approved": True}
    
    def update_pnl(self, portfolio_value: float, previous_value: float):
        if previous_value > 0:
            daily_return = (portfolio_value - previous_value) / previous_value
            self.daily_pnl.append(daily_return)
            
            if len(self.daily_pnl) > 30:
                self.daily_pnl.pop(0)
    
    def _get_volatility_factor(self, symbol: str) -> float:
        high_vol_symbols = ['TSLA', 'AMC', 'GME', 'NVDA']
        if symbol in high_vol_symbols:
            return 2.0
        return 1.0
    
    def should_close_position(self, position: Dict) -> Dict:
        try:
            unrealized_pnl_pct = position.get('unrealized_plpc', 0)
            
            if unrealized_pnl_pct < -self.stop_loss:
                return {
                    "should_close": True,
                    "reason": "stop_loss",
                    "urgency": "high"
                }
            
            if unrealized_pnl_pct > 0.2:
                return {
                    "should_close": True,
                    "reason": "take_profit",
                    "urgency": "medium",
                    "partial": True,
                    "close_ratio": 0.5
                }
            
            return {"should_close": False}
            
        except Exception as e:
            return {"should_close": False, "error": str(e)}
    
    def reset_daily_limits(self):
        self.is_trading_halted = False
        current_app.logger.info("일일 거래 한도 초기화")