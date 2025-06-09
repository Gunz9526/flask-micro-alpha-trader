from . import celery
from .services.alpaca_service import AlpacaService
from .services.ai_service import AIService
from .services.trading_service import TradingService
from .services.metrics_service import get_metrics_service
from flask import current_app
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import os
import requests
from datetime import datetime
import pytz

WATCHLIST = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']

@celery.task(name="app.tasks.smart_trading_pipeline", bind=True)
def smart_trading_pipeline(self):
    current_app.logger.info("자동 트레이딩 파이프라인 시작")
    
    if not is_market_open():
        current_app.logger.info("시장 폐장 - 트레이딩 건너뜀")
        return {"status": "skipped", "reason": "market_closed"}
    
    results = []
    
    try:
        trading_service = TradingService()
        ai_service = AIService()
        
        stop_loss_results = trading_service.check_stop_losses()
        if stop_loss_results:
            current_app.logger.warning(f"손절매 실행: {len(stop_loss_results)}건")
            results.extend(stop_loss_results)
        
        available_symbols = []
        for symbol in WATCHLIST:
            model_path = f"models/{symbol}_model.pkl"
            if os.path.exists(model_path):
                available_symbols.append(symbol)
        
        if not available_symbols:
            current_app.logger.warning("학습된 모델이 없음")
            return {"status": "no_models", "message": "모델 학습이 필요합니다"}
        
        current_app.logger.info(f"사용 가능한 모델: {available_symbols}")
        
        # 종목 트레이딩ㅇ
        for symbol in available_symbols:
            try:
                alpaca_service = AlpacaService()
                bars_df = alpaca_service.get_stock_bars(
                    symbol, 
                    TimeFrame(1, TimeFrameUnit.Day), 
                    limit=120
                )
                
                if bars_df is None or len(bars_df) < 50:
                    continue
                
                signal_result = ai_service.get_trading_signal(symbol, bars_df)
                
                if signal_result["signal"] in ["BUY", "SELL"]:
                    trade_result = trading_service.execute_ai_signal(
                        symbol,
                        signal_result["signal"],
                        signal_result["confidence"]
                    )
                    
                    trade_result["signal_info"] = signal_result
                    results.append(trade_result)
                    
                    if trade_result["status"] == "success":
                        send_notification.delay(
                            f"거래 실행: {symbol} {signal_result['signal']} "
                            f"신뢰도: {signal_result['confidence']:.1%}"
                        )
                
            except Exception as e:
                current_app.logger.error(f"{symbol} 처리 오류: {e}")
                continue
        
        successful_trades = len([r for r in results if r.get("status") == "success"])
        
        return {
            "status": "completed",
            "results": results,
            "successful_trades": successful_trades,
            "processed_symbols": len(available_symbols)
        }
        
    except Exception as e:
        current_app.logger.error(f"트레이딩 파이프라인 오류: {e}")
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
                    limit=250 
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
                time.sleep(5)
                    
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
        metrics_service = get_metrics_service()
        metrics_service.update_system_metrics()
        
        trading_service = TradingService()
        account_info = trading_service.alpaca_service.get_account_info()
        
        if account_info:
            portfolio_value = float(account_info['portfolio_value'])
            positions = trading_service.get_positions()
            position_count = positions.get('total_positions', 0)
            
            metrics_service.update_portfolio_metrics(portfolio_value, position_count)
            
    except Exception as e:
        current_app.logger.error(f"메트릭 업데이트 오류: {e}")

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
