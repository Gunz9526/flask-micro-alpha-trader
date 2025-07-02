import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from flask import current_app
from .ai_service import AIService
from .alpaca_service import AlpacaService
import copy

class BacktestEngine:
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        
        self.max_position_size = 0.10
        self.max_portfolio_exposure = 0.80
        self.max_positions = 7
        self.stop_loss_pct = -0.05
        self.take_profit_pct = 0.10
        
        self.reset()
    
    def reset(self):
        self.cash = self.initial_capital
        self.positions = {}  # {symbol: {'shares': int, 'avg_cost': float, 'last_price': float}}
        self.portfolio_history = []
        self.trade_history = []
        self.current_date = None
        self.daily_stats = {}
        
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        portfolio_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in prices and position['shares'] > 0:
                portfolio_value += position['shares'] * prices[symbol]
        
        return portfolio_value
    
    def get_position_value(self, symbol: str, price: float) -> float:
        if symbol in self.positions:
            return self.positions[symbol]['shares'] * price
        return 0.0
    
    def get_portfolio_exposure(self, prices: Dict[str, float]) -> float:
        total_position_value = sum(
            self.get_position_value(symbol, prices.get(symbol, 0))
            for symbol in self.positions.keys()
        )
        portfolio_value = self.get_portfolio_value(prices)
        return total_position_value / portfolio_value if portfolio_value > 0 else 0
    
    def check_risk_limits(self, symbol: str, signal: str, confidence: float, 
                         price: float, prices: Dict[str, float]) -> Dict[str, Any]:
        portfolio_value = self.get_portfolio_value(prices)
        
        if signal == "BUY":
            active_positions = len([p for p in self.positions.values() if p['shares'] > 0])
            if active_positions >= self.max_positions:
                return {"approved": False, "reason": f"max_positions_exceeded ({self.max_positions})"}
            
            max_position_value = portfolio_value * self.max_position_size * confidence
            
            current_exposure = self.get_portfolio_exposure(prices)
            new_exposure_ratio = (max_position_value / portfolio_value) + current_exposure
            
            if new_exposure_ratio > self.max_portfolio_exposure:
                return {"approved": False, "reason": f"max_exposure_exceeded ({self.max_portfolio_exposure:.0%})"}
            
            return {"approved": True, "max_position_value": max_position_value}
        
        return {"approved": True}
    
    def check_exit_conditions(self, symbol: str, price: float) -> Optional[Dict[str, Any]]:
        if symbol not in self.positions or self.positions[symbol]['shares'] == 0:
            return None
        
        position = self.positions[symbol]
        avg_cost = position['avg_cost']
        unrealized_pnl_pct = (price - avg_cost) / avg_cost
        
        if unrealized_pnl_pct <= self.stop_loss_pct:
            return {
                "action": "STOP_LOSS",
                "sell_ratio": 1.0,
                "pnl_pct": unrealized_pnl_pct,
                "reason": f"손절매 {unrealized_pnl_pct:.2%}"
            }
        
        if unrealized_pnl_pct >= self.take_profit_pct:
            return {
                "action": "TAKE_PROFIT", 
                "sell_ratio": 0.5,
                "pnl_pct": unrealized_pnl_pct,
                "reason": f"이익실현 {unrealized_pnl_pct:.2%}"
            }
        
        return None
    
    def execute_trade(self, symbol: str, signal: str, confidence: float, 
                     price: float, date: datetime, prices: Dict[str, float]) -> Dict[str, Any]:
        try:
            # 리스크 체크
            risk_check = self.check_risk_limits(symbol, signal, confidence, price, prices)
            if not risk_check["approved"]:
                return {"success": False, "reason": risk_check["reason"]}
            
            if signal == "BUY":
                return self._execute_buy(symbol, confidence, price, date, risk_check["max_position_value"])
            elif signal == "SELL":
                return self._execute_sell(symbol, confidence, price, date, "AI_SIGNAL")
            
            return {"success": False, "reason": "invalid_signal"}
            
        except Exception as e:
            current_app.logger.error(f"거래 실행 오류: {e}")
            return {"success": False, "reason": str(e)}
    
    def _execute_buy(self, symbol: str, confidence: float, price: float, 
                    date: datetime, max_position_value: float) -> Dict[str, Any]:
        shares_to_buy = int(max_position_value / price)
        total_cost = shares_to_buy * price * (1 + self.commission)
        
        if total_cost <= self.cash and shares_to_buy > 0:
            self.cash -= total_cost
            
            if symbol in self.positions:
                current_shares = self.positions[symbol]['shares']
                current_avg_cost = self.positions[symbol]['avg_cost']
                
                new_shares = current_shares + shares_to_buy
                new_avg_cost = ((current_shares * current_avg_cost) + (shares_to_buy * price)) / new_shares
                
                self.positions[symbol] = {
                    'shares': new_shares,
                    'avg_cost': new_avg_cost,
                    'last_price': price
                }
            else:
                self.positions[symbol] = {
                    'shares': shares_to_buy,
                    'avg_cost': price,
                    'last_price': price
                }
            
            trade_record = {
                "date": date,
                "symbol": symbol,
                "action": "BUY",
                "shares": shares_to_buy,
                "price": price,
                "cost": total_cost,
                "confidence": confidence,
                "cash_after": self.cash,
                "position_after": self.positions[symbol]['shares']
            }
            
            self.trade_history.append(trade_record)
            current_app.logger.debug(f"매수 실행: {symbol} {shares_to_buy}주 @ ${price:.2f}")
            
            return {"success": True, "trade": trade_record}
        
        return {"success": False, "reason": "insufficient_cash"}
    
    def _execute_sell(self, symbol: str, confidence: float, price: float, 
                     date: datetime, sell_reason: str) -> Dict[str, Any]:
        if symbol not in self.positions or self.positions[symbol]['shares'] == 0:
            return {"success": False, "reason": "no_position"}
        
        current_shares = self.positions[symbol]['shares']
        avg_cost = self.positions[symbol]['avg_cost']
        
        if sell_reason == "AI_SIGNAL":
            shares_to_sell = max(1, int(current_shares * confidence))
        else:  # STOP_LOSS, TAKE_PROFIT
            shares_to_sell = current_shares
        
        shares_to_sell = min(shares_to_sell, current_shares)
        
        revenue = shares_to_sell * price * (1 - self.commission)
        self.cash += revenue
        
        cost_basis = shares_to_sell * avg_cost
        realized_pnl = revenue - cost_basis
        realized_pnl_pct = realized_pnl / cost_basis if cost_basis > 0 else 0
        
        remaining_shares = current_shares - shares_to_sell
        if remaining_shares > 0:
            self.positions[symbol]['shares'] = remaining_shares
            self.positions[symbol]['last_price'] = price
        else:
            del self.positions[symbol]
        
        trade_record = {
            "date": date,
            "symbol": symbol,
            "action": "SELL",
            "shares": shares_to_sell,
            "price": price,
            "revenue": revenue,
            "realized_pnl": realized_pnl,
            "realized_pnl_pct": realized_pnl_pct,
            "confidence": confidence,
            "sell_reason": sell_reason,
            "cash_after": self.cash,
            "position_after": remaining_shares
        }
        
        self.trade_history.append(trade_record)
        current_app.logger.debug(f"매도 실행: {symbol} {shares_to_sell}주 @ ${price:.2f}, 손익: {realized_pnl_pct:.2%}")
        
        return {"success": True, "trade": trade_record}
    
    def process_risk_management(self, date: datetime, prices: Dict[str, float]):
        symbols_to_process = list(self.positions.keys())
        
        for symbol in symbols_to_process:
            if symbol in prices:
                exit_condition = self.check_exit_conditions(symbol, prices[symbol])
                
                if exit_condition:
                    if exit_condition["sell_ratio"] < 1.0:
                        current_shares = self.positions[symbol]['shares']
                        shares_to_sell = int(current_shares * exit_condition["sell_ratio"])
                        if shares_to_sell > 0:
                            self._execute_sell(symbol, 1.0, prices[symbol], date, exit_condition["action"])
                    else:
                        self._execute_sell(symbol, 1.0, prices[symbol], date, exit_condition["action"])
    
    def record_portfolio_state(self, date: datetime, prices: Dict[str, float]):
        portfolio_value = self.get_portfolio_value(prices)
        
        position_details = {}
        total_unrealized_pnl = 0
        
        for symbol, position in self.positions.items():
            if symbol in prices and position['shares'] > 0:
                current_price = prices[symbol]
                position_value = position['shares'] * current_price
                unrealized_pnl = position['shares'] * (current_price - position['avg_cost'])
                unrealized_pnl_pct = unrealized_pnl / (position['shares'] * position['avg_cost'])
                
                position_details[symbol] = {
                    'shares': position['shares'],
                    'avg_cost': position['avg_cost'],
                    'current_price': current_price,
                    'position_value': position_value,
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_pct': unrealized_pnl_pct
                }
                
                total_unrealized_pnl += unrealized_pnl
        
        portfolio_record = {
            "date": date,
            "cash": self.cash,
            "portfolio_value": portfolio_value,
            "total_return": (portfolio_value - self.initial_capital) / self.initial_capital,
            "positions_count": len([p for p in self.positions.values() if p['shares'] > 0]),
            "portfolio_exposure": self.get_portfolio_exposure(prices),
            "total_unrealized_pnl": total_unrealized_pnl,
            "position_details": position_details
        }
        
        self.portfolio_history.append(portfolio_record)

class BacktestService:
    
    def __init__(self):
        self.alpaca_service = AlpacaService()
        self.ai_service = AIService()
    
    def run_backtest(self, symbols: List[str], start_date: datetime, 
                    end_date: datetime, initial_capital: float = 100000) -> Dict[str, Any]:
        current_app.logger.info(f"향상된 백테스트 시작: {symbols}, {start_date.date()} ~ {end_date.date()}")
        
        try:
            engine = BacktestEngine(initial_capital)
            
            historical_data = self._load_historical_data(symbols, start_date, end_date)
            
            if not historical_data:
                return {"status": "error", "message": "백테스트용 데이터가 없습니다"}
            
            trained_models = self._train_models(symbols, historical_data, start_date)
            
            if not trained_models:
                return {"status": "error", "message": "AI 모델 학습에 실패했습니다"}
            
            backtest_dates = self._get_trading_dates(historical_data, start_date, end_date)
            
            for i, date in enumerate(backtest_dates):
                current_prices = self._get_prices_for_date(historical_data, date)
                
                if not current_prices:
                    continue
                
                engine.process_risk_management(date, current_prices)
                
                for symbol in symbols:
                    if symbol in current_prices and symbol in trained_models:
                        if i % 5 == 0:
                            self._retrain_model(symbol, historical_data, date)
                        
                        signal_result = self._generate_signal(symbol, historical_data, date)
                        
                        if signal_result and signal_result.get("signal") in ["BUY", "SELL"]:
                            confidence = signal_result.get("confidence", 0)
                            
                            if confidence >= 0.65:
                                trade_result = engine.execute_trade(
                                    symbol=symbol,
                                    signal=signal_result["signal"],
                                    confidence=confidence,
                                    price=current_prices[symbol],
                                    date=date,
                                    prices=current_prices
                                )
                                
                                if trade_result.get("success"):
                                    current_app.logger.debug(f"거래 성공: {symbol} {signal_result['signal']}")
                
                engine.record_portfolio_state(date, current_prices)
            
            return self._analyze__results(engine, start_date, end_date, symbols)
            
        except Exception as e:
            current_app.logger.error(f"백테스트 실행 오류: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    def _load_historical_data(self, symbols: List[str], start_date: datetime, 
                            end_date: datetime) -> Dict[str, Dict]:
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        
        # 충분한 학습 데이터를 위해 시작일 이전 200일 추가
        extended_start = start_date - timedelta(days=250)
        
        historical_data = {}
        
        for symbol in symbols:
            try:
                bars_df = self.alpaca_service.get_stock_bars(
                    symbol=symbol,
                    timeframe=TimeFrame(1, TimeFrameUnit.Day),
                    limit=600
                )
                
                if bars_df is not None and not bars_df.empty:
                    # 학습용 전체 데이터
                    training_mask = bars_df.index < start_date
                    training_data = bars_df[training_mask]
                    
                    backtest_mask = (bars_df.index >= start_date) & (bars_df.index <= end_date)
                    backtest_data = bars_df[backtest_mask]
                    
                    if len(training_data) >= 100 and len(backtest_data) >= 1:
                        historical_data[symbol] = {
                            'full_data': bars_df,
                            'training_data': training_data,
                            'backtest_data': backtest_data
                        }
                        current_app.logger.info(f"{symbol} 데이터 로드: 학습 {len(training_data)}일, 백테스트 {len(backtest_data)}일")
                    else:
                        current_app.logger.warning(f"{symbol} 데이터 부족: 학습 {len(training_data)}일, 백테스트 {len(backtest_data)}일")
                        
            except Exception as e:
                current_app.logger.error(f"{symbol} 데이터 로드 실패: {e}")
                continue
        
        current_app.logger.info(f"총 {len(historical_data)}개 종목 데이터 로드 완료")
        return historical_data
    
    def _train_models(self, symbols: List[str], historical_data: Dict, 
                     start_date: datetime) -> Dict[str, bool]:
        trained_models = {}
        
        for symbol in symbols:
            if symbol in historical_data:
                try:
                    training_data = historical_data[symbol]['training_data']
                    
                    if len(training_data) >= 100:
                        result = self.ai_service.train_strategy(symbol, training_data)
                        
                        if result.get("status") == "success":
                            trained_models[symbol] = True
                            current_app.logger.info(f"{symbol} AI 모델 학습 완료")
                        else:
                            current_app.logger.warning(f"❌ {symbol} AI 모델 학습 실패: {result.get('reason', 'unknown')}")
                    else:
                        current_app.logger.warning(f"❌ {symbol} 학습 데이터 부족: {len(training_data)}일")
                        
                except Exception as e:
                    current_app.logger.error(f"{symbol} 모델 학습 오류: {e}")
                    continue
        
        current_app.logger.info(f"총 {len(trained_models)}개 종목 모델 학습 완료")
        return trained_models
    
    def _retrain_model(self, symbol: str, historical_data: Dict, current_date: datetime):
        try:
            symbol_data = historical_data[symbol]['full_data']
            training_data = symbol_data[symbol_data.index < current_date]
            
            if len(training_data) >= 100:
                result = self.ai_service.train_strategy(symbol, training_data)
                if result.get("status") == "success":
                    current_app.logger.debug(f"{symbol} 모델 재학습 완료")
                    
        except Exception as e:
            current_app.logger.error(f"{symbol} 모델 재학습 오류: {e}")
    
    def _generate_signal(self, symbol: str, historical_data: Dict, 
                        current_date: datetime) -> Optional[Dict]:
        try:
            symbol_data = historical_data[symbol]['full_data']
            cutoff_data = symbol_data[symbol_data.index <= current_date]
            
            if len(cutoff_data) >= 50:
                signal_result = self.ai_service.get_trading_signal(symbol, cutoff_data)
                return signal_result
                
        except Exception as e:
            current_app.logger.error(f"{symbol} 신호 생성 오류: {e}")
            
        return None
    
    def _get_trading_dates(self, historical_data: Dict, start_date: datetime, 
                          end_date: datetime) -> List[datetime]:
        all_dates = set()
        
        for symbol_data in historical_data.values():
            backtest_data = symbol_data['backtest_data']
            all_dates.update(backtest_data.index)
        
        trading_dates = [d for d in sorted(all_dates) if start_date <= d <= end_date]
        current_app.logger.info(f"총 {len(trading_dates)}개 거래일 확인")
        
        return trading_dates
    
    def _get_prices_for_date(self, historical_data: Dict, date: datetime) -> Dict[str, float]:
        prices = {}
        
        for symbol, data in historical_data.items():
            backtest_data = data['backtest_data']
            if date in backtest_data.index:
                prices[symbol] = float(backtest_data.loc[date, 'close'])
        
        return prices
    
    def _analyze__results(self, engine: BacktestEngine, 
                                start_date: datetime, end_date: datetime, 
                                symbols: List[str]) -> Dict[str, Any]:
        
        if not engine.portfolio_history:
            return {"status": "error", "message": "백테스트 데이터가 없습니다"}
        
        portfolio_df = pd.DataFrame(engine.portfolio_history)
        portfolio_df.set_index('date', inplace=True)
        
        initial_value = engine.initial_capital
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        daily_returns = portfolio_df['portfolio_value'].pct_change().dropna()
        
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            annualized_return = (1 + total_return) ** (252 / len(portfolio_df)) - 1
            sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
            volatility = daily_returns.std() * np.sqrt(252)
        else:
            annualized_return = total_return
            sharpe_ratio = 0
            volatility = 0
        
        running_max = portfolio_df['portfolio_value'].expanding().max()
        drawdown = (portfolio_df['portfolio_value'] - running_max) / running_max
        max_drawdown = drawdown.min()
        
        buy_trades = [t for t in engine.trade_history if t['action'] == 'BUY']
        sell_trades = [t for t in engine.trade_history if t['action'] == 'SELL']
        
        total_trades = len(buy_trades) + len(sell_trades)
        
        profitable_trades = len([t for t in sell_trades if t.get('realized_pnl', 0) > 0])
        win_rate = profitable_trades / len(sell_trades) if sell_trades else 0
        
        if sell_trades:
            avg_profit = np.mean([t['realized_pnl_pct'] for t in sell_trades])
            avg_profit_winners = np.mean([t['realized_pnl_pct'] for t in sell_trades if t.get('realized_pnl', 0) > 0]) if profitable_trades > 0 else 0
            avg_loss_losers = np.mean([t['realized_pnl_pct'] for t in sell_trades if t.get('realized_pnl', 0) <= 0]) if len(sell_trades) - profitable_trades > 0 else 0
        else:
            avg_profit = avg_profit_winners = avg_loss_losers = 0
        
        stop_loss_trades = len([t for t in sell_trades if t.get('sell_reason') == 'STOP_LOSS'])
        take_profit_trades = len([t for t in sell_trades if t.get('sell_reason') == 'TAKE_PROFIT'])
        
        portfolio_df['year_month'] = portfolio_df.index.to_period('M')
        monthly_returns = portfolio_df.groupby('year_month')['total_return'].last().pct_change().dropna()
        
        symbol_performance = {}
        for symbol in symbols:
            symbol_trades = [t for t in sell_trades if t['symbol'] == symbol]
            if symbol_trades:
                symbol_pnl = sum(t.get('realized_pnl', 0) for t in symbol_trades)
                symbol_trades_count = len(symbol_trades)
                symbol_win_rate = len([t for t in symbol_trades if t.get('realized_pnl', 0) > 0]) / symbol_trades_count
                
                symbol_performance[symbol] = {
                    "trades": symbol_trades_count,
                    "total_pnl": symbol_pnl,
                    "win_rate": symbol_win_rate,
                    "avg_pnl_per_trade": symbol_pnl / symbol_trades_count
                }
        
        result = {
            "status": "success",
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "trading_days": len(portfolio_df),
                "symbols": symbols
            },
            "performance": {
                "initial_capital": initial_value,
                "final_value": final_value,
                "total_return": total_return,
                "annualized_return": annualized_return,
                "sharpe_ratio": sharpe_ratio,
                "volatility": volatility,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "avg_profit_per_trade": avg_profit,
                "avg_profit_winners": avg_profit_winners,
                "avg_loss_losers": avg_loss_losers
            },
            "trading_stats": {
                "total_trades": total_trades,
                "buy_trades": len(buy_trades),
                "sell_trades": len(sell_trades),
                "profitable_trades": profitable_trades,
                "losing_trades": len(sell_trades) - profitable_trades,
                "stop_loss_trades": stop_loss_trades,
                "take_profit_trades": take_profit_trades,
                "avg_holding_period": self._calculate_avg_holding_period(engine.trade_history)
            },
            "risk_management": {
                "max_positions_reached": max(r.get('positions_count', 0) for r in engine.portfolio_history),
                "max_exposure_reached": max(r.get('portfolio_exposure', 0) for r in engine.portfolio_history),
                "stop_loss_effectiveness": stop_loss_trades / len(sell_trades) if sell_trades else 0,
                "take_profit_effectiveness": take_profit_trades / len(sell_trades) if sell_trades else 0
            },
            "symbol_performance": symbol_performance,
            "monthly_returns": monthly_returns.to_dict() if len(monthly_returns) > 0 else {},
            "trades": engine.trade_history,
            "portfolio_history": portfolio_df.to_dict('records'),
            "final_positions": {k: v for k, v in engine.positions.items() if v['shares'] > 0}
        }
        
        current_app.logger.info(f"백테스트 완료: 총 수익률 {total_return:.2%}, 샤프 비율 {sharpe_ratio:.2f}, 승률 {win_rate:.1%}")
        
        return result
    
    def _calculate_avg_holding_period(self, trade_history: List[Dict]) -> float:
        try:
            buy_trades = {(t['symbol'], t['date']): t for t in trade_history if t['action'] == 'BUY'}
            sell_trades = [t for t in trade_history if t['action'] == 'SELL']
            
            holding_periods = []
            
            for sell_trade in sell_trades:
                symbol = sell_trade['symbol']
                sell_date = sell_trade['date']
                
                relevant_buys = [
                    (buy_date, buy_trade) for (buy_symbol, buy_date), buy_trade in buy_trades.items()
                    if buy_symbol == symbol and buy_date <= sell_date
                ]
                
                if relevant_buys:
                    latest_buy_date, _ = max(relevant_buys, key=lambda x: x[0])
                    holding_period = (sell_date - latest_buy_date).days
                    holding_periods.append(holding_period)
            
            return np.mean(holding_periods) if holding_periods else 0
            
        except Exception as e:
            current_app.logger.error(f"평균 보유 기간 계산 오류: {e}")
            return 0