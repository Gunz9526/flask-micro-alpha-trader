from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.common.exceptions import APIError
from flask import current_app
from typing import Dict, Any, Optional, List
from .alpaca_service import AlpacaService
from .risk_manager import RiskManager

class TradingService:
    
    def __init__(self):
        self.alpaca_service = AlpacaService()
        self.trading_client = self.alpaca_service.trading_client
        self.risk_manager = RiskManager()
        
    def execute_ai_signal(self, symbol: str, signal: str, confidence: float) -> Dict[str, Any]:
        try:
            current_app.logger.info(f"시그널 실행 시작: {symbol} {signal} ({confidence:.1%})")
            
            if confidence < 0.65:
                return {
                    "status": "skipped",
                    "reason": "low_confidence",
                    "confidence": confidence
                }
            
            account_info = self.alpaca_service.get_account_info()
            if not account_info:
                return {"status": "error", "reason": "account_info_failed"}
            
            portfolio_value = float(account_info['portfolio_value'])
            
            daily_check = self.risk_manager.check_daily_limits(portfolio_value)
            if not daily_check["approved"]:
                return {"status": "blocked", "reason": daily_check["reason"]}
            
            if signal == "BUY":
                return self._execute_buy_order(symbol, confidence, account_info)
            elif signal == "SELL":
                return self._execute_sell_order(symbol, confidence)
            else:
                return {"status": "hold", "reason": "hold_signal"}
                
        except Exception as e:
            current_app.logger.error(f"시그널 실행 오류: {e}")
            return {"status": "error", "reason": str(e)}
    
    def _execute_buy_order(self, symbol: str, confidence: float, account_info: Dict) -> Dict:
        try:
            portfolio_value = float(account_info['portfolio_value'])
            current_price = self.alpaca_service.get_current_price(symbol)
            
            if not current_price:
                return {"status": "error", "reason": "price_unavailable"}
            
            position_risk = self.risk_manager.check_position_risk(
                symbol, "BUY", confidence, current_price, portfolio_value
            )
            
            if not position_risk["approved"]:
                return {"status": "blocked", "reason": "position_risk_failed"}
            
            current_positions = self.get_positions()["positions"]
            
            portfolio_risk = self.risk_manager.check_portfolio_risk(
                {
                    "symbol": symbol,
                    "investment_amount": position_risk["investment_amount"],
                    "portfolio_value": portfolio_value
                },
                current_positions
            )
            
            if not portfolio_risk["approved"]:
                return {"status": "blocked", "reason": portfolio_risk["reason"]}
            
            shares = position_risk["shares"]
            if shares <= 0:
                return {"status": "skipped", "reason": "insufficient_shares"}
            
            order_result = self._place_market_order(symbol, "BUY", shares)
            
            if order_result["status"] == "success":
                order_result["risk_info"] = position_risk
            
            return order_result
            
        except Exception as e:
            return {"status": "error", "reason": str(e)}
    
    def _execute_sell_order(self, symbol: str, confidence: float) -> Dict:
        try:
            positions = self.get_positions()
            if positions["status"] != "success":
                return {"status": "error", "reason": "position_query_failed"}
            
            target_position = None
            for pos in positions["positions"]:
                if pos["symbol"] == symbol and pos["quantity"] > 0:
                    target_position = pos
                    break
            
            if not target_position:
                return {"status": "skipped", "reason": "no_position"}
            
            close_decision = self.risk_manager.should_close_position(target_position)
            
            if close_decision.get("partial", False):
                sell_ratio = close_decision.get("close_ratio", confidence)
            else:
                sell_ratio = confidence
            
            sell_shares = max(1, int(target_position["quantity"] * sell_ratio))
            
            return self._place_market_order(symbol, "SELL", sell_shares)
            
        except Exception as e:
            return {"status": "error", "reason": str(e)}
    
    def _place_market_order(self, symbol: str, side: str, qty: int) -> Dict:
        try:
            order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_request)
            
            return {
                "status": "success",
                "order_id": str(order.id),
                "symbol": symbol,
                "side": side,
                "quantity": qty,
                "order_status": str(order.status)
            }
            
        except APIError as e:
            current_app.logger.error(f"주문 API 오류: {e}")
            return {"status": "error", "reason": f"api_error: {str(e)}"}
        except Exception as e:
            current_app.logger.error(f"주문 실행 오류: {e}")
            return {"status": "error", "reason": str(e)}
    
    def get_positions(self) -> Dict[str, Any]:
        try:
            positions = self.trading_client.get_all_positions()
            
            position_list = []
            for position in positions:
                pos_data = {
                    "symbol": position.symbol,
                    "quantity": float(position.qty),
                    "market_value": float(position.market_value),
                    "unrealized_pl": float(position.unrealized_pl),
                    "unrealized_plpc": float(position.unrealized_plpc),
                    "cost_basis": float(position.cost_basis)
                }
                position_list.append(pos_data)
            
            return {
                "status": "success",
                "positions": position_list,
                "total_positions": len(position_list)
            }
            
        except Exception as e:
            return {"status": "error", "reason": str(e)}
    
    def check_stop_losses(self) -> List[Dict]:
        results = []
        try:
            positions = self.get_positions()
            if positions["status"] != "success":
                return results
            
            for position in positions["positions"]:
                close_decision = self.risk_manager.should_close_position(position)
                
                if close_decision["should_close"]:
                    if close_decision["reason"] == "stop_loss":
                        result = self._place_market_order(
                            position["symbol"], 
                            "SELL", 
                            int(position["quantity"])
                        )
                        result["reason"] = "stop_loss_triggered"
                        results.append(result)
                    
                    elif close_decision["reason"] == "take_profit":
                        sell_qty = int(position["quantity"] * 0.5)
                        result = self._place_market_order(
                            position["symbol"], 
                            "SELL", 
                            sell_qty
                        )
                        result["reason"] = "profit_taking"
                        results.append(result)
            
            return results
            
        except Exception as e:
            current_app.logger.error(f"손절매 체크 오류: {e}")
            return results