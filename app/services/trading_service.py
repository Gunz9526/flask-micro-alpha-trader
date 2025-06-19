from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError
from flask import current_app
from typing import Dict, Any, List
from .alpaca_service import AlpacaService
from .risk_manager import RiskManager
from .database_service import DatabaseService

class TradingService:
    
    def __init__(self):
        self.alpaca_service = AlpacaService()
        self.trading_client = self.alpaca_service.trading_client
        self.risk_manager = RiskManager()
        self.db_service = DatabaseService()
        
    def execute_ai_signal(self, symbol: str, signal: str, confidence: float, ai_signal_result: Dict) -> Dict[str, Any]:
        try:
            current_app.logger.info(f"시그널 실행 시작: {symbol} {signal} ({confidence:.1%})")
            
            confidence_threshold = current_app.config.get('AI_CONFIDENCE_THRESHOLD', 0.6)
            if confidence < confidence_threshold:
                return {"status": "skipped", "reason": "low_confidence", "confidence": confidence}
            
            account_info = self.alpaca_service.get_account_info()
            if not account_info:
                return {"status": "error", "reason": "account_info_failed"}
            
            portfolio_value = float(account_info['portfolio_value'])
            
            daily_check = self.risk_manager.check_daily_limits(portfolio_value)
            if not daily_check["approved"]:
                return {"status": "blocked", "reason": daily_check["reason"]}
            
            if signal == "BUY":
                return self._execute_buy_order(symbol, confidence, account_info, ai_signal_result)
            elif signal == "SELL":
                return self._execute_sell_order(symbol, confidence)
            else:
                return {"status": "hold", "reason": "hold_signal"}
                
        except Exception as e:
            current_app.logger.error(f"시그널 실행 오류: {e}", exc_info=True)
            return {"status": "error", "reason": str(e)}
    
    def _execute_buy_order(self, symbol: str, confidence: float, account_info: Dict, ai_signal_result: Dict) -> Dict:
        try:
            portfolio_value = float(account_info['portfolio_value'])
            current_price = self.alpaca_service.get_current_price(symbol)
            if not current_price:
                return {"status": "error", "reason": "price_unavailable"}

            latest_volatility = ai_signal_result.get('latest_features', {}).get('volatility_20d', 0.02)
            
            position_risk = self.risk_manager.check_position_risk(
                confidence=confidence, 
                volatility=latest_volatility
            )

            if not position_risk.get("approved"):
                return {"status": "blocked", "reason": position_risk.get("reason", "position_risk_failed")}

            position_size = position_risk["position_size"]
            investment_amount = portfolio_value * position_size
            shares = int(investment_amount / current_price)
            
            if shares <= 0:
                return {"status": "skipped", "reason": "insufficient_shares_after_risk_adj"}

            current_positions = self.get_positions().get("positions", [])
            portfolio_risk = self.risk_manager.check_portfolio_risk(investment_amount, current_positions, portfolio_value)
            
            if not portfolio_risk.get("approved"):
                return {"status": "blocked", "reason": portfolio_risk.get("reason")}
            
            return self._place_market_order(symbol, "BUY", shares)
        except Exception as e:
            current_app.logger.error(f"매수 주문 실행 오류: {e}", exc_info=True)
            return {"status": "error", "reason": str(e)}
    
    def _execute_sell_order(self, symbol: str, confidence: float) -> Dict:
        try:
            positions = self.get_positions()
            if positions.get("status") != "success":
                return {"status": "error", "reason": "position_query_failed"}
            
            target_position = next((pos for pos in positions["positions"] if pos["symbol"] == symbol and float(pos["quantity"]) > 0), None)
            
            if not target_position:
                return {"status": "skipped", "reason": "no_position_to_sell"}
            
            sell_shares = max(1, int(float(target_position["quantity"]) * confidence))
            
            return self._place_market_order(symbol, "SELL", sell_shares)
        except Exception as e:
            current_app.logger.error(f"매도 주문 실행 오류: {e}", exc_info=True)
            return {"status": "error", "reason": str(e)}
    
    def _place_market_order(self, symbol: str, side: str, qty: int) -> Dict:
        try:
            order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            
            order_request = MarketOrderRequest(
                symbol=symbol, qty=qty, side=order_side, time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_request)
            
            current_app.logger.info(f"주문 제출 성공: ID={order.id}, {symbol} {side} {qty}주")
            return {"status": "success", "order_id": str(order.id), "symbol": symbol,
                    "side": side, "quantity": qty, "order_status": str(order.status)}
        except APIError as e:
            current_app.logger.error(f"주문 API 오류: {e}")
            return {"status": "error", "reason": f"api_error: {str(e)}"}
        except Exception as e:
            current_app.logger.error(f"주문 실행 오류: {e}", exc_info=True)
            return {"status": "error", "reason": str(e)}
    
    def get_positions(self) -> Dict[str, Any]:
        try:
            positions = self.trading_client.get_all_positions()
            position_list = [{
                "symbol": p.symbol, "quantity": float(p.qty), "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl), "unrealized_plpc": float(p.unrealized_plpc),
                "cost_basis": float(p.cost_basis)
            } for p in positions]
            
            return {"status": "success", "positions": position_list, "total_positions": len(position_list)}
        except Exception as e:
            current_app.logger.error(f"포지션 조회 오류: {e}", exc_info=True)
            return {"status": "error", "reason": str(e)}
    
    def check_stop_losses(self) -> List[Dict]:
        results = []
        try:
            positions = self.get_positions()
            if positions.get("status") != "success": return results
            
            for position in positions["positions"]:
                close_decision = self.risk_manager.should_close_position(position)
                if close_decision.get("should_close"):
                    qty_to_sell = int(float(position["quantity"]) * close_decision.get("close_ratio", 1.0))
                    if qty_to_sell > 0:
                        result = self._place_market_order(position["symbol"], "SELL", qty_to_sell)
                        result["reason"] = close_decision.get("reason")
                        results.append(result)
            return results
        except Exception as e:
            current_app.logger.error(f"손절매 체크 오류: {e}", exc_info=True)
            return results