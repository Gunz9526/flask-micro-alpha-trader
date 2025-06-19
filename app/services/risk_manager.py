import redis
from flask import current_app
from typing import Dict, List

class RiskManager:
    def __init__(self):
        self.max_position_size = current_app.config.get('RISK_MAX_POSITION_SIZE', 0.10)
        self.max_portfolio_exposure = current_app.config.get('RISK_MAX_PORTFOLIO_EXPOSURE', 0.80)
        self.max_positions = current_app.config.get('RISK_MAX_POSITIONS', 7)
        self.stop_loss_pct = current_app.config.get('RISK_STOP_LOSS_PCT', -0.05)
        self.take_profit_pct = current_app.config.get('RISK_TAKE_PROFIT_PCT', 0.10)
        self.volatility_target = current_app.config.get('RISK_VOLATILITY_TARGET', 0.015)
        self.max_daily_loss = current_app.config.get('RISK_MAX_DAILY_LOSS', -0.02) # 일일 최대 손실률

        self.redis_client = redis.from_url(current_app.config['REDIS_URL'])

    @property
    def is_trading_halted(self) -> bool:
        halted = self.redis_client.get('trading_halted')
        return halted == b'1'

    @is_trading_halted.setter
    def is_trading_halted(self, value: bool):
        self.redis_client.set('trading_halted', '1' if value else '0', ex=86400) # ex=seconds

    def check_position_risk(self, confidence: float, volatility: float) -> Dict:
        try:
            volatility_factor = max(volatility / self.volatility_target, 1.0)
            
            base_size = self.max_position_size * confidence
            
            adjusted_size = base_size / volatility_factor
            
            position_size = min(adjusted_size, self.max_position_size)
            
            return {"approved": True, "position_size": position_size}
        except Exception as e:
            current_app.logger.error(f"포지션 리스크 체크 오류: {e}", exc_info=True)
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
            return {"approved": False, "reason": "trading_halted_due_to_daily_loss"}
        
        today_pnl_str = self.redis_client.get('daily_pnl')
        if today_pnl_str:
            today_pnl = float(today_pnl_str)
            if (today_pnl / portfolio_value) < self.max_daily_loss:
                self.is_trading_halted = True
                current_app.logger.critical("!!! 일일 최대 손실 한도 초과. 모든 신규 거래를 중단합니다. !!!")
                return {"approved": False, "reason": "daily_loss_limit_exceeded"}
        
        return {"approved": True}

    def update_daily_pnl(self, portfolio_value: float):
        redis_key = 'last_day_portfolio_value'
        previous_value_str = self.redis_client.get(redis_key)
        
        if previous_value_str:
            previous_value = float(previous_value_str)
            daily_return = (portfolio_value - previous_value) / previous_value
            self.redis_client.set('daily_pnl', daily_return)
        
        # 다음 날 계산을 위해 현재 가치를 저장합니다.
        self.redis_client.set(redis_key, portfolio_value)

    def reset_daily_limits(self):]
        self.is_trading_halted = False
        self.redis_client.set('daily_pnl', 0.0)
        current_app.logger.info("일일 거래 한도 및 PnL 초기화 완료")