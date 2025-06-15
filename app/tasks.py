from . import celery
from .services.alpaca_service import AlpacaService
from .services.ai_service import AIService
from .services.trading_service import TradingService
from .services.metrics_service import get_metrics_service
from flask import current_app
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import os
import requests
import psutil

from prometheus_client import Gauge, Counter
from datetime import datetime
import pytz

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
        
        available_symbols = [
            symbol for symbol in WATCHLIST 
            if os.path.exists(f"models/{symbol}_model_0.pkl")
        ]
        
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
        ai_service = AIService()
        alpaca_service = AlpacaService()
        results = []
        
        for symbol in WATCHLIST:
            try:
                bars_df = alpaca_service.get_stock_bars(
                    symbol,
                    TimeFrame(1, TimeFrameUnit.Day),
                    limit=500
                )
                
                if bars_df is None or len(bars_df) < 100:
                    current_app.logger.warning(f"{symbol} 데이터 부족")
                    continue
                
                result = ai_service.train_strategy(symbol, bars_df)
                results.append({"symbol": symbol, "result": result})
                
                if result.get("validation_passed"):
                    current_app.logger.info(f"{symbol} 모델 학습 성공")
                else:
                    current_app.logger.warning(f"{symbol} 모델 검증 실패")
                
                import time
                # time.sleep(5)
                    
            except Exception as e:
                current_app.logger.error(f"{symbol} 학습 오류: {e}")
                continue
        
        successful_models = len([r for r in results if r["result"].get("validation_passed")])
        
        send_notification.delay(
            f"모델 학습 완료\n"
            f"성공: {successful_models}/{len(WATCHLIST)}개"
        )
        
        return {"status": "completed", "results": results}
        
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
