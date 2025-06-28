import datetime
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from flask import current_app
from typing import Dict, Any, List
from .alpaca_service import AlpacaService
from .risk_manager import RiskManager
from .database_service import DatabaseService
from .metrics_service import get_metrics_service

class TradingService:
    
    def __init__(self):
        self.alpaca_service = AlpacaService()
        self.trading_client = self.alpaca_service.trading_client
        self.risk_manager = RiskManager()
        self.db_service = DatabaseService()        
        self.metrics_service = get_metrics_service()
        
    def execute_ai_signal(self, symbol: str, signal: str, confidence: float, ai_signal_result: Dict) -> Dict[str, Any]:
        try:
            self.metrics_service.record_trading_signal(symbol, signal, 'ensemble', confidence)
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
                result = self._execute_buy_order(symbol, confidence, account_info, ai_signal_result)

                if result.get("status") == "success":
                    self.risk_manager.record_trade()
                return result
            
            elif signal == "SELL":
                result = self._execute_sell_order(symbol, confidence, ai_signal_result)
                if result.get("status") == "success":
                    self.risk_manager.record_trade()
                return result
            
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
                if portfolio_risk.get("reason", "").startswith("max_exposure_exceeded"):
                    current_app.logger.info(f"[{symbol}] 포트폴리오 한도 초과. 리밸런싱 가능성 검토.")
                    rebalance_success = self._handle_portfolio_rebalancing(
                        new_buy_symbol=symbol,
                        new_buy_signal_result=ai_signal_result,
                        current_positions=current_positions
                    )
                    
                    if not rebalance_success:
                        return {"status": "blocked", "reason": "리밸런싱 실패 또는 필요 없음"}
                    
                    current_app.logger.info(f"[{symbol}] 리밸런싱 성공. 신규 매수 주문 진행.")

                else:
                    return {"status": "blocked", "reason": portfolio_risk.get("reason")}
            
            return self._place_market_order(symbol, "BUY", shares, ai_signal_result)
        except Exception as e:
            current_app.logger.error(f"매수 주문 실행 오류: {e}", exc_info=True)
            return {"status": "error", "reason": str(e)}
    
    def _execute_sell_order(self, symbol: str, confidence: float, ai_signal_result: Dict | None) -> Dict:
        try:
            current_app.logger.info(f"[{symbol}] 매도 주문 실행 로직 시작. 신뢰도: {confidence:.2%}")

            # 1. 현재 보유 포지션 정보 조회
            positions_data = self.get_positions()
            if positions_data.get("status") != "success":
                current_app.logger.error(f"[{symbol}] 매도 처리 중 포지션 정보 조회 실패.")
                return {"status": "error", "reason": "position_query_failed"}

            # 2. 매도 대상 포지션 탐색
            target_position = next(
                (pos for pos in positions_data["positions"] if pos["symbol"] == symbol and float(pos["quantity"]) > 0),
                None
            )

            # 3. 매도할 포지션이 없는 경우
            if not target_position:
                current_app.logger.warning(f"[{symbol}] 매도할 포지션이 존재하지 않아 주문을 건너뜁니다.")
                return {"status": "skipped", "reason": "no_position_to_sell"}

            # 4. 매도 수량 결정
            #    - 신뢰도에 기반하여 보유 수량의 일부 또는 전체를 매도합니다.
            #    - 최소 1주는 매도하도록 보장합니다.
            holding_qty = float(target_position["quantity"])
            shares_to_sell = int(holding_qty * confidence)
            
            # 계산된 수량이 0이고, 신뢰도가 0보다 크면 최소 1주 매도
            if shares_to_sell == 0 and confidence > 0:
                shares_to_sell = 1
            
            # 계산된 수량이 보유 수량을 초과하지 않도록 보장 (안전장치)
            shares_to_sell = min(shares_to_sell, int(holding_qty))

            if shares_to_sell <= 0:
                current_app.logger.info(f"[{symbol}] 계산된 매도 수량이 0이므로 주문을 건너뜁니다.")
                return {"status": "skipped", "reason": "calculated_sell_quantity_is_zero"}

            current_app.logger.info(f"[{symbol}] 보유수량: {holding_qty}, 매도수량: {shares_to_sell} (신뢰도 {confidence:.2%} 적용)")

            # 5. 최종 주문 실행
            #    - _place_market_order에 AI 신호 정보를 포함하여 전달합니다.
            return self._place_market_order(symbol, "SELL", shares_to_sell, ai_signal_result)

        except Exception as e:
            current_app.logger.error(f"[{symbol}] 매도 주문 실행 중 예외 발생: {e}", exc_info=True)
            return {"status": "error", "reason": f"unexpected_error: {str(e)}"}
    
    def _place_market_order(self, symbol: str, side: str, qty: int, ai_signal_result: Dict | None) -> Dict:
        try:
            order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            
            current_price = self.alpaca_service.get_current_price(symbol)
            if not current_price:
                return {"status": "error", "reason": "price_unavailable"}
            
            order_request = MarketOrderRequest(
                symbol=symbol, qty=qty, side=order_side, time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_request)

            trade_data = {
                'symbol': symbol,
                'side': side,
                'quantity': qty,
                'price': current_price,
                'order_id': str(order.id),
                'executed_at': datetime.datetime.now(),
                'status': 'executed'
            }
            
            if ai_signal_result:
                trade_data.update({
                    'ai_signal': ai_signal_result.get('signal'),
                    'ai_confidence': ai_signal_result.get('confidence'),
                    'ai_predicted_return': ai_signal_result.get('predicted_return')
                })

            self.db_service.record_trade(trade_data)
            self.metrics_service.record_trade_execution(symbol, side, 'success')
            
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
                        result = self._place_market_order(position["symbol"], "SELL", qty_to_sell, None)
                        result["reason"] = close_decision.get("reason")
                        results.append(result)
            return results
        except Exception as e:
            current_app.logger.error(f"손절매 체크 오류: {e}", exc_info=True)
            return results
        
    def _handle_portfolio_rebalancing(self, new_buy_symbol: str, new_buy_signal_result: Dict, current_positions: List[Dict]) -> bool:
        """
        포트폴리오 한도 초과 시, 기존 포지션을 팔고 새 포지션을 매수할지 결정하고 실행합니다.
        :return: 리밸런싱에 성공하여 신규 매수가 가능해지면 True, 아니면 False.
        """
        from .ai_service import AIService # 순환 참조 방지를 위해 여기서 임포트
        ai_service = AIService()

        new_buy_return = new_buy_signal_result.get('predicted_return', 0)
        
        # 현재 보유 포지션들의 기대수익률 평가
        position_evaluations = []
        for pos in current_positions:
            pos_symbol = pos['symbol']
            if pos_symbol == new_buy_symbol: # 신규 매수하려는 종목은 매도 대상에서 제외
                continue

            bars_df = self.alpaca_service.get_stock_bars(pos_symbol, TimeFrame(1, TimeFrameUnit.Day), limit=120)
            if bars_df is None:
                continue
            
            signal_result = ai_service.get_trading_signal(pos_symbol, bars_df)
            position_evaluations.append({
                "symbol": pos_symbol,
                "quantity": float(pos['quantity']),
                "predicted_return": signal_result.get('predicted_return', -1)
            })
        
        if not position_evaluations:
            current_app.logger.warning("리밸런싱 매도 후보 없음. (보유 포지션이 없거나 평가 실패)")
            return False

        # 기대수익률이 가장 낮은 포지션을 매도 후보로 선정
        sell_candidate = min(position_evaluations, key=lambda x: x['predicted_return'])
        
        candidate_return = sell_candidate['predicted_return']
        candidate_symbol = sell_candidate['symbol']
        
        # 교체 결정 로직
        swap_premium = 1.5 # 신규 종목의 기대수익률이 50% 더 높아야 함 (설정값으로 관리 가능)
        
        current_app.logger.info(f"리밸런싱 검토: 신규({new_buy_symbol}, 수익률:{new_buy_return:.4f}) vs "
                                f"매도후보({candidate_symbol}, 수익률:{candidate_return:.4f})")

        is_justified = False
        # 조건 1: 신규 수익률 > 0, 후보 수익률 <= 신규 수익률
        if new_buy_return > 0 and candidate_return <= new_buy_return:
             # 조건 2: 후보 수익률이 음수이거나, 신규 수익률이 후보 수익률보다 '의미있게' 높은 경우
            if candidate_return < 0 or new_buy_return > candidate_return * swap_premium:
                is_justified = True

        if not is_justified:
            current_app.logger.info(f"[{candidate_symbol}] → [{new_buy_symbol}] 교체 기각: 수익률 이점 부족.")
            return False
            
        current_app.logger.info(f"[{candidate_symbol}] → [{new_buy_symbol}] 교체 승인. 매도 실행.")
        
        # 매도 실행
        sell_qty = int(sell_candidate['quantity'])
        # 매도 주문 시 AI 신호 정보는 없으므로 None 전달
        sell_result = self._place_market_order(candidate_symbol, "SELL", sell_qty, None) 
        
        if sell_result.get("status") == "success":
            current_app.logger.info(f"리밸런싱 매도 성공: {candidate_symbol} {sell_qty}주")
            # 매도 성공 시, 후속 매수가 가능함을 알림
            return True
        else:
            current_app.logger.error(f"리밸런싱 매도 실패: {candidate_symbol}. 이유: {sell_result.get('reason')}")
            return False