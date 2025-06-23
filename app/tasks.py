
import glob
import json
import os
import requests
import psutil
import pytz
from datetime import datetime

from flask import current_app
from prometheus_client import Gauge, Counter
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from .services.optimizer_service import HyperparameterOptimizer
from . import celery
from .services.alpaca_service import AlpacaService
from .services.ai_service import AIService
from .services.trading_service import TradingService
from .services.metrics_service import get_metrics_service
from .services.risk_manager import RiskManager


WATCHLIST = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'NVDA', 'AMZN', 'META', 'NFLX', 'AMD', 'JNJ', 'UNH', 'JPM', 'PG', 'CAT', 'XOM', 'V', 'MA', 'DIS', 'HD', 'KO', 'PEP', 'INTC', 'CSCO', 'CMCSA', 'VZ', 'T', 'MRK', 'PFE', 'ABT', 'NKE', 'WMT']
PORTFOLIO_VALUE = Gauge('portfolio_value_usd', 'Current portfolio value in USD', multiprocess_mode='max')
ACTIVE_POSITIONS = Gauge('active_positions_count', 'Number of active positions', multiprocess_mode='max')
SYSTEM_CPU = Gauge('system_cpu_usage_percent', 'System CPU usage percentage', multiprocess_mode='max')
SYSTEM_MEMORY = Gauge('system_memory_usage_percent', 'System memory usage percentage', multiprocess_mode='max')
TRADES_EXECUTED = Counter('trades_executed_total', 'Total number of trades executed', ['symbol', 'side', 'status'])


@celery.task(name="app.tasks.smart_trading_pipeline", bind=True)
def smart_trading_pipeline(self):
    current_app.logger.info("자동 트레이딩 파이프라인 시작")
    
    if not is_market_open():
        current_app.logger.info("시장 폐장 - 트레이딩 건너뜀")
        return {"status": "skipped", "reason": "market_closed"}
    
    try:
        trading_service = TradingService()
        ai_service = AIService()
        alpaca_service = trading_service.alpaca_service 
    except Exception as e:
        current_app.logger.error(f"서비스 초기화 실패: {e}", exc_info=True)
        return {"status": "error", "message": "서비스 초기화 실패"}

    trade_results = []
    all_decisions = []
    
    try:
        stop_loss_results = trading_service.check_stop_losses()
        if stop_loss_results:
            current_app.logger.warning(f"손절매 실행: {len(stop_loss_results)}건")
            trade_results.extend(stop_loss_results)
        
        available_symbols = []
        for symbol in WATCHLIST:
            model_files = glob.glob(f"models/{symbol}_*.pkl")
            if model_files:
                available_symbols.append(symbol)
    
        
        if not available_symbols:
            current_app.logger.warning("학습된 모델이 없음")
            return {"status": "no_models", "message": "모델 학습이 필요합니다"}
        
        current_app.logger.info(f"사용 가능한 모델: {available_symbols}")
        
        for symbol in available_symbols:
            try:
                bars_df = alpaca_service.get_stock_bars(
                    symbol, TimeFrame(1, TimeFrameUnit.Day), limit=120
                )
                if bars_df is None or len(bars_df) < 50:
                    current_app.logger.warning(f"[{symbol}] 데이터 부족으로 건너뜀")
                    continue
                
                signal_result = ai_service.get_trading_signal(symbol, bars_df)
                all_decisions.append(signal_result)
                
                current_app.logger.info(
                    f"[{symbol}] AI 신호 결과: "
                    f"Signal={signal_result['signal']}, "
                    f"Confidence={signal_result.get('confidence', 0):.3f}, "
                    f"Predicted Return={signal_result.get('predicted_return', 0):.4f}"
                )
                
                if signal_result["signal"] in ["BUY", "SELL"]:
                    trade_result = trading_service.execute_ai_signal(
                        symbol, signal_result["signal"],
                        signal_result["confidence"], signal_result
                    )
                    trade_result["signal_info"] = signal_result
                    trade_results.append(trade_result)
                    
                    if trade_result["status"] == "success":
                        TRADES_EXECUTED.labels(
                                            symbol=symbol, 
                                            side=signal_result["signal"], 
                                            status='success'
                                        ).inc()
                        update_system_metrics.delay()
                        update_portfolio_metrics.delay()
                        send_notification.delay(
                            f"거래 실행: {symbol} {signal_result['signal']} "
                            f"신뢰도: {signal_result['confidence']:.1%}"
                        )
            except Exception as e:
                current_app.logger.error(f"[{symbol}] 처리 중 오류 발생: {e}", exc_info=True)
                continue
        
        successful_trades = len([r for r in trade_results if r.get("status") == "success"])
        
        return {"status": "completed", "trade_results": trade_results,
                "all_decisions": all_decisions, "successful_trades": successful_trades,
                "processed_symbols": len(available_symbols)}
        
    except Exception as e:
        current_app.logger.error(f"트레이딩 파이프라인 오류: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

@celery.task(name="app.tasks.train_models_batch", bind=True)
def train_models_batch(self):
    current_app.logger.info("배치 모델 학습 시작")
    try:
        alpaca_service = AlpacaService()        
        ai_service = AIService()
        results = []
        
        for symbol in WATCHLIST:
            try:
                best_params_map = {}
                
                for model_name, config in ai_service.model_classes.items():
                    StrategyClass = config['class']
                    
                    class_name = StrategyClass.__name__
                    
                    params_path = os.path.join("best_params", f"{symbol}_{class_name}.json")
                    
                    if os.path.exists(params_path):
                        try:
                            with open(params_path, 'r') as f:
                                loaded_params = json.load(f)
                                
                                temp_strategy = StrategyClass(symbol=symbol, random_state=0)
                                model_type = temp_strategy.model_type
                                best_params_map[model_type] = loaded_params
                                
                                current_app.logger.info(f"[{symbol}] {class_name} → {model_type} 파라미터 로드 성공: {len(loaded_params)}개")
                                current_app.logger.debug(f"[{symbol}] {model_type} 파라미터: {loaded_params}")
                        except Exception as e:
                            current_app.logger.warning(f"[{symbol}] {model_type} 파라미터 파일 로드 실패: {e}", exc_info=True)
                
                current_app.logger.info(f"[{symbol}] 최적 파라미터 완료: {list(best_params_map.keys())}")
                bars_df = alpaca_service.get_stock_bars(
                    symbol,
                    TimeFrame(1, TimeFrameUnit.Day),
                    limit=500
                )
                
                if bars_df is None or len(bars_df) < 200:
                    current_app.logger.warning(f"{symbol} 데이터 부족")
                    continue
                
                result = ai_service.train_strategy(symbol, bars_df, best_params_map)
                results.append({"symbol": symbol, "result": result, "optimized_params": list(best_params_map.keys())})
                
                if result.get("status") == "success":
                    current_app.logger.info(f"{symbol} 모델 학습 성공")
                else:
                    current_app.logger.warning(f"{symbol} 모델 검증 실패")
                
                # import time
                # time.sleep(5)
                    
            except Exception as e:
                current_app.logger.error(f"{symbol} 학습 오류: {e}")
                continue
        
        successful_symbols = 0
        total_successful_models = 0
        optimized_symbols = []
        
        for result in results:
            if result.get("result", {}).get("status") == "success":
                successful_symbols += 1
                total_successful_models += result["result"].get("successful_models", 0)
                if result.get("optimized_params"):
                    optimized_symbols.append(result["symbol"])

        total_attempted = len(WATCHLIST) * len(ai_service.model_classes)
        
        summary_message = (
            f"모델 학습 완료\n"
            f"성공 종목: {successful_symbols}/{len(WATCHLIST)}\n"
            f"성공 모델: {total_successful_models}/{total_attempted}\n"
            f"최적화 활용: {len(optimized_symbols)}개 종목"
        )
        
        send_notification.delay(summary_message)
        current_app.logger.info(f"배치 학습 완료: {summary_message}")
        
        return {
            "status": "completed", 
            "results": results,
            "summary": {
                "successful_symbols": successful_symbols,
                "total_successful_models": total_successful_models,
                "optimized_symbols": optimized_symbols,
                "total_attempted": total_attempted
            }
        }
        
    except Exception as e:
        current_app.logger.error(f"배치 학습 오류: {e}")
        return {"status": "error", "message": str(e)}

@celery.task(name="app.tasks.update_system_metrics")
def update_system_metrics():
    try:
        SYSTEM_CPU.set(psutil.cpu_percent(interval=None))
        SYSTEM_MEMORY.set(psutil.virtual_memory().percent)
                    
    except Exception as e:
        current_app.logger.error(f"메트릭 업데이트 작업 오류: {e}", exc_info=True)
                    

@celery.task(name="app.tasks.send_notification")
def send_notification(message: str):
    webhook_url = current_app.config.get('DISCORD_WEBHOOK_URL')
    
    if not webhook_url:
        return {"status": "skipped"}
    
    try:
        payload = {"content": f"**Alpha Trader**\n{message}"}
        response = requests.post(webhook_url, json=payload, timeout=5)
        return {"status": "success"}
    except Exception as e:
        current_app.logger.error(f"알림 전송 실패: {e}")
        return {"status": "failed"}

@celery.task(name="app.tasks.send_daily_report")
def send_daily_report():
    try:
        trading_service = TradingService()
        account_info = trading_service.alpaca_service.get_account_info()
        positions = trading_service.get_positions()
        
        if account_info:
            report = (
                f"일일 리포트\n"
                f"포트폴리오: ${account_info['portfolio_value']:,.2f}\n"
                f"현금: ${account_info['cash']:,.2f}\n"
                f"포지션 수: {positions.get('total_positions', 0)}개"
            )
            
            send_notification.delay(report)
            
    except Exception as e:
        current_app.logger.error(f"일일 리포트 생성 오류: {e}")

def is_market_open() -> bool:
    try:
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        
        if now.weekday() >= 5:
            return False
        
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    except Exception as e:
        current_app.logger.error(f"시장 시간 체크 오류: {e}")
        return False

@celery.task(name="app.tasks.train_specific_models", bind=True)
def train_specific_models(self, symbols: list):
    current_app.logger.info(f"지정 종목 모델 학습 시작: {symbols}")
    try:
        ai_service = AIService()
        alpaca_service = AlpacaService()
        results = []
        
        for symbol in symbols:
            try:
                bars_df = alpaca_service.get_stock_bars(
                    symbol, TimeFrame(1, TimeFrameUnit.Day), limit=500
                )
                if bars_df is None or len(bars_df) < 200:
                    current_app.logger.warning(f"[{symbol}] 데이터 부족으로 건너뜀")
                    continue
                
                result = ai_service.train_strategy(symbol, bars_df)
                results.append({"symbol": symbol, "result": result})
            except Exception as e:
                current_app.logger.error(f"[{symbol}] 학습 중 오류 발생: {e}", exc_info=True)
                continue
        
        return {"status": "completed", "results": results}
    except Exception as e:
        current_app.logger.error(f"지정 종목 학습 오류: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

@celery.task(name="app.tasks.update_portfolio_metrics")
def update_portfolio_metrics():
    try:
        trading_service = TradingService()
        account_info = trading_service.alpaca_service.get_account_info()
        
        if account_info:
            PORTFOLIO_VALUE.set(float(account_info['portfolio_value']))
            positions = trading_service.get_positions()
            ACTIVE_POSITIONS.set(positions.get('total_positions', 0))
        else:
            PORTFOLIO_VALUE.set(0.0)
            ACTIVE_POSITIONS.set(0.0)
    except Exception as e:
        current_app.logger.error(f"포트폴리오 메트릭 업데이트 오류: {e}", exc_info=True)

@celery.task(name="app.tasks.optimize_hyperparameters_for_watchlist")
def optimize_hyperparameters_for_watchlist():
    
    from .services.ai_service import AIService, LightGBMStrategy, XGBoostStrategy
    current_app.logger.info("하이퍼파라미터 최적화 파이프라인 시작")
    alpaca_service = AlpacaService()
    
    strategies_to_optimize = {
        'lgbm': LightGBMStrategy,
        'xgb': XGBoostStrategy
    }

    for symbol in WATCHLIST:
        try:
            current_app.logger.info(f"[{symbol}] 최적화 데이터 로드 중...")
            bars_df = alpaca_service.get_stock_bars(
                symbol, TimeFrame(1, TimeFrameUnit.Day), limit=500
            )
            if bars_df is None or len(bars_df) < 200:
                current_app.logger.warning(f"[{symbol}] 데이터 부족으로 최적화 건너뜀")
                continue
            
            optimizer = HyperparameterOptimizer(symbol, bars_df)
            
            for model_type, StrategyClass in strategies_to_optimize.items():
                temp_strategy_instance = StrategyClass(symbol=symbol)
                optimizer.optimize(temp_strategy_instance, n_trials=100)

        except Exception as e:
            current_app.logger.error(f"[{symbol}] 최적화 중 심각한 오류 발생: {e}", exc_info=True)
            continue
            
    current_app.logger.info("하이퍼파라미터 최적화 파이프라인 완료")
    return {"status": "completed", "message": "모든 종목에 대한 하이퍼파라미터 최적화 완료"}

@celery.task(name="app.tasks.reset_risk_limits")
def reset_risk_limits():
    risk_manager = RiskManager()
    risk_manager.reset_daily_limits()