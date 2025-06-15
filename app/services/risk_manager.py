from typing import Dict, List, Optional
from flask import current_app
import numpy as np
from datetime import datetime, timedelta
import redis


class RiskManager:
    def __init__(self):
        self.max_position_size = float('0.10')  # 종목당 최대 10%
        self.max_portfolio_exposure = float('0.80')  # 총 투자 비중 80%
        self.max_positions = float('7')  # 최대 보유 종목 수 7개
        self.stop_loss = float('0.05')  # -5% 손절
        self.take_profit = float('0.10')  # +10% 익절
        self.volatility_target = float('0.015')  # 목표 변동성 1.5%
    
        self.redis_client = redis.from_url(current_app.config['REDIS_URL'])

        

    def check_position_risk(self, symbol: str, confidence: float, portfolio_value: float, volatility: float) -> Dict:
        try:
            volatility_factor = max(volatility / self.volatility_target, 1.0)
            
            base_size = self.max_position_size * confidence
            
            adjusted_size = base_size / volatility_factor
            
            position_size = min(adjusted_size, self.max_position_size)
            
            return {"approved": True, "position_size": position_size}
        
        except Exception as e:
            current_app.logger.error(f"포지션 리스크 체크 오류: {e}")
            return {"approved": False, "reason": str(e)}

    def check_portfolio_risk(self, new_position_value: float, current_positions: List[Dict], portfolio_value: float) -> Dict:
        if len(current_positions) >= self.max_positions:
            return {"approved": False, "reason": f"max_positions_exceeded ({self.max_positions})"}
        
        total_value = sum(pos.get('market_value', 0) for pos in current_positions)
        if (total_value + new_position_value) / portfolio_value > self.max_portfolio_exposure:
            return {"approved": False, "reason": f"max_exposure_exceeded ({self.max_portfolio_exposure:.0%})"}
            
        return {"approved": True}

    def should_close_position(self, position: Dict) -> Dict:
        unrealized_pnl_pct = position.get('unrealized_plpc', 0)
        
        if unrealized_pnl_pct < self.stop_loss_pct:
            return {"should_close": True, "reason": "stop_loss"}
            
        if unrealized_pnl_pct > self.take_profit_pct:
            return {"should_close": True, "reason": "take_profit", "close_ratio": 0.5}
            
        return {"should_close": False}
    
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
    
    def reset_daily_limits(self):
        self.is_trading_halted = False
        current_app.logger.info("일일 거래 한도 초기화")