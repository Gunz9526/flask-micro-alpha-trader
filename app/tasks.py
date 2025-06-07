from . import celery
from .services.alpaca_service import AlpacaService
from flask import current_app
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import requests
from datetime import datetime

@celery.task(name="app.tasks.get_alpaca_account_info", bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def get_alpaca_account_info_task(self):
    """Alpaca 계정 정보 조회 작업"""
    current_app.logger.info(f"계정 정보 조회 작업 시작 (ID: {self.request.id})")
    
    try:
        service = AlpacaService()
        account_info = service.get_account_info()

        if account_info:
            current_app.logger.info(f"계정 정보 조회 성공 (ID: {self.request.id})")
            
            # Discord 알림 전송
            send_discord_notification.delay(
                f"계정 정보 업데이트\n"
                f"포트폴리오 가치: ${account_info['portfolio_value']:,.2f}\n"
                f"현금: ${account_info['cash']:,.2f}\n"
                f"매수력: ${account_info['buying_power']:,.2f}"
            )
            
            return account_info
        else:
            current_app.logger.warning(f"계정 정보 조회 실패 (ID: {self.request.id})")
            return None
            
    except Exception as e:
        current_app.logger.error(f"계정 정보 조회 작업 예외 (ID: {self.request.id}): {e}", exc_info=True)
        raise

@celery.task(name="app.tasks.fetch_stock_bars", bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 30})
def fetch_stock_bars_task(self, symbol: str, timeframe_unit_str: str, timeframe_value: int, limit: int = 100):
    """주식 바 데이터 조회 작업"""
    current_app.logger.info(f"바 데이터 조회 작업 시작 (ID: {self.request.id}): {symbol}")
    
    try:
        service = AlpacaService()

        try:
            timeframe_unit = TimeFrameUnit[timeframe_unit_str]
        except KeyError:
            error_msg = f"잘못된 TimeFrameUnit: {timeframe_unit_str}"
            current_app.logger.error(error_msg)
            return {"error": error_msg}

        timeframe = TimeFrame(timeframe_value, timeframe_unit)
        bars_df = service.get_stock_bars(symbol=symbol, timeframe=timeframe, limit=limit)

        if bars_df is not None and not bars_df.empty:
            result = {
                "symbol": symbol,
                "count": len(bars_df),
                "latest_price": float(bars_df.iloc[-1]['close']),
                "timestamp": datetime.now().isoformat(),
                "timeframe": f"{timeframe_value}{timeframe_unit_str}"
            }
            
            current_app.logger.info(f"{symbol} 바 데이터 조회 성공 (ID: {self.request.id}): {result['count']}개")
            return result
        else:
            current_app.logger.warning(f"{symbol} 바 데이터 없음 (ID: {self.request.id})")
            return {"symbol": symbol, "count": 0, "error": "No data available"}
            
    except Exception as e:
        current_app.logger.error(f"바 데이터 조회 작업 예외 (ID: {self.request.id}, {symbol}): {e}", exc_info=True)
        raise

@celery.task(name="app.tasks.send_discord_notification", bind=True)
def send_discord_notification(self, message: str):
    """Discord 웹훅 알림 전송"""
    webhook_url = current_app.config.get('DISCORD_WEBHOOK_URL')
    
    if not webhook_url:
        current_app.logger.warning("Discord 웹훅 URL이 설정되지 않음")
        return {"status": "skipped", "reason": "No webhook URL"}
    
    try:
        payload = {
            "content": f"Alpha Trader Bot\n{message}",
            "username": "Alpha Trader"
        }
        
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        
        current_app.logger.info(f"Discord 알림 전송 성공 (ID: {self.request.id})")
        return {"status": "success", "message": "Notification sent"}
        
    except Exception as e:
        current_app.logger.error(f"Discord 알림 전송 실패 (ID: {self.request.id}): {e}")
        return {"status": "failed", "error": str(e)}

# 정기 작업을 위한 스케줄러 (나중에 celery beat에 추가할 예정)
@celery.task(name="app.tasks.market_health_check")
def market_health_check():
    """시장 상태 체크 (정기 작업용)"""
    symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'MSFT']
    
    for symbol in symbols:
        fetch_stock_bars_task.delay(symbol, "Minute", 1, 5)
    
    get_alpaca_account_info_task.delay()
    current_app.logger.info("시장 건강 상태 체크 작업들 예약 완료")