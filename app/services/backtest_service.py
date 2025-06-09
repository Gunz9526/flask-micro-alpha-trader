import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
from flask import current_app
from .ai_service import AIService
from .alpaca_service import AlpacaService

class BacktestEngine:
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.commission = 0.001
        self.reset()
    
    def reset(self):
        self.cash = self.initial_capital
        self.positions = {}  # {symbol: shares}
        self.portfolio_history = []
        self.trade_history = []
        self.current_date = None
        
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        portfolio_value = self.cash
        
        for symbol, shares in self.positions.items():
            if symbol in prices and shares > 0:
                portfolio_value += shares * prices[symbol]
        
        return portfolio_value
    
    def execute_trade(self, symbol: str, signal: str, confidence: float, 
                     price: float, date: datetime) -> bool:
        try:
            portfolio_value = self.get_portfolio_value({symbol: price})
            
            if signal == "BUY":
                max_position_value = portfolio_value * 0.1
                position_value = max_position_value * confidence
                
                shares_to_buy = int(position_value / price)
                cost = shares_to_buy * price * (1 + self.commission)
                
                if cost <= self.cash and shares_to_buy > 0:
                    self.cash -= cost
                    self.positions[symbol] = self.positions.get(symbol, 0) + shares_to_buy
                    
                    self.trade_history.append({
                        "date": date,
                        "symbol": symbol,
                        "action": "BUY",
                        "shares": shares_to_buy,
                        "price": price,
                        "cost": cost,
                        "confidence": confidence
                    })
                    return True
            
            elif signal == "SELL":
                current_shares = self.positions.get(symbol, 0)
                if current_shares > 0:
                    shares_to_sell = int(current_shares * confidence)
                    if shares_to_sell == 0:
                        shares_to_sell = current_shares
                    
                    revenue = shares_to_sell * price * (1 - self.commission)
                    self.cash += revenue
                    self.positions[symbol] = current_shares - shares_to_sell
                    
                    self.trade_history.append({
                        "date": date,
                        "symbol": symbol,
                        "action": "SELL",
                        "shares": shares_to_sell,
                        "price": price,
                        "revenue": revenue,
                        "confidence": confidence
                    })
                    return True
                    
            return False
            
        except Exception as e:
            current_app.logger.error(f"거래 실행 오류: {e}")
            return False
    
    def record_portfolio_state(self, date: datetime, prices: Dict[str, float]):
        portfolio_value = self.get_portfolio_value(prices)
        
        self.portfolio_history.append({
            "date": date,
            "cash": self.cash,
            "portfolio_value": portfolio_value,
            "positions": self.positions.copy(),
            "return": (portfolio_value - self.initial_capital) / self.initial_capital
        })

class BacktestService:
    
    def __init__(self):
        self.alpaca_service = AlpacaService()
        self.ai_service = AIService()
    
    def run_backtest(self, symbols: List[str], start_date: datetime, 
                    end_date: datetime, initial_capital: float = 100000) -> Dict[str, Any]:
        current_app.logger.info(f"백테스트 시작: {symbols}, {start_date} ~ {end_date}")
        
        try:
            engine = BacktestEngine(initial_capital)
            
            historical_data = {}
            for symbol in symbols:
                from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
                
                extended_start = start_date - timedelta(days=200)
                
                bars_df = self.alpaca_service.get_stock_bars(
                    symbol=symbol,
                    timeframe=TimeFrame(1, TimeFrameUnit.Day),
                    limit=500
                )
                
                if bars_df is not None and not bars_df.empty:
                    mask = (bars_df.index >= start_date) & (bars_df.index <= end_date)
                    historical_data[symbol] = {
                        'full_data': bars_df,
                        'backtest_data': bars_df[mask]
                    }
            
            if not historical_data:
                return {"status": "error", "message": "백테스트용 데이터가 없습니다"}
            
            trained_models = {}
            for symbol in symbols:
                if symbol in historical_data:
                    result = self.ai_service.train_strategy(
                        symbol, 
                        historical_data[symbol]['full_data']
                    )
                    if result.get("status") == "success":
                        trained_models[symbol] = True
                        current_app.logger.info(f"{symbol} AI 모델 학습 완료")
            
            all_dates = set()
            for symbol_data in historical_data.values():
                all_dates.update(symbol_data['backtest_data'].index)
            
            sorted_dates = sorted(all_dates)
            
            for date in sorted_dates:
                current_prices = {}
                for symbol, data in historical_data.items():
                    if date in data['backtest_data'].index:
                        current_prices[symbol] = float(data['backtest_data'].loc[date, 'close'])
                
                for symbol in symbols:
                    if symbol in current_prices and symbol in trained_models:
                        symbol_data = historical_data[symbol]['full_data']
                        cutoff_data = symbol_data[symbol_data.index <= date]
                        
                        if len(cutoff_data) >= 50:
                            signal_result = self.ai_service.get_trading_signal(symbol, cutoff_data)
                            
                            if signal_result.get("signal") in ["BUY", "SELL"]:
                                confidence = signal_result.get("confidence", 0)
                                
                                if confidence >= 0.65:
                                    engine.execute_trade(
                                        symbol, 
                                        signal_result["signal"], 
                                        confidence,
                                        current_prices[symbol], 
                                        date
                                    )
                
                engine.record_portfolio_state(date, current_prices)
            
            return self._analyze_results(engine, start_date, end_date)
            
        except Exception as e:
            current_app.logger.error(f"백테스트 실행 오류: {e}")
            return {"status": "error", "message": str(e)}
    
    def _analyze_results(self, engine: BacktestEngine, start_date: datetime, 
                        end_date: datetime) -> Dict[str, Any]:
        
        if not engine.portfolio_history:
            return {"status": "error", "message": "백테스트 데이터가 없습니다"}
        
        portfolio_df = pd.DataFrame(engine.portfolio_history)
        portfolio_df.set_index('date', inplace=True)
        
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - engine.initial_capital) / engine.initial_capital
        
        daily_returns = portfolio_df['portfolio_value'].pct_change().dropna()
        
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0
        
        running_max = portfolio_df['portfolio_value'].expanding().max()
        drawdown = (portfolio_df['portfolio_value'] - running_max) / running_max
        max_drawdown = drawdown.min()
        
        profitable_trades = len([t for t in engine.trade_history 
                               if t['action'] == 'SELL' and 'revenue' in t])
        total_trades = len([t for t in engine.trade_history if t['action'] == 'SELL'])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        benchmark_return = 0.10 
        
        result = {
            "status": "success",
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": len(portfolio_df)
            },
            "performance": {
                "initial_capital": engine.initial_capital,
                "final_value": final_value,
                "total_return": total_return,
                "annualized_return": total_return * (365 / len(portfolio_df)),
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "total_trades": total_trades
            },
            "trades": engine.trade_history,
            "portfolio_history": portfolio_df.to_dict('records'),
            "final_positions": engine.positions
        }
        
        current_app.logger.info(f"백테스트 완료: 총 수익률 {total_return:.2%}, 샤프 비율 {sharpe_ratio:.2f}")
        
        return result