from alpaca.trading.client import TradingClient
from alpaca.common.exceptions import APIError
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
from flask import current_app
import pandas as pd
from typing import Optional, Dict, Any

class AlpacaService:
    def __init__(self):
        self.api_key = current_app.config.get("ALPACA_API_KEY")
        self.api_secret = current_app.config.get("ALPACA_API_SECRET")
        self.base_url = current_app.config.get("APCA_API_BASE_URL")
        self.paper = current_app.config.get("ALPACA_PAPER", True)

        if not self.api_key or not self.api_secret:
            current_app.logger.error("Alpaca API Key 또는 Secret이 설정되지 않았습니다.")
            raise ValueError("Alpaca API Key 또는 Secret이 설정되지 않았습니다.")

        self._init_clients()

    def _init_clients(self):
        """클라이언트 초기화"""
        try:
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper=self.paper,
                url_override=self.base_url if self.paper and self.base_url else None
            )
            current_app.logger.info(f"Alpaca TradingClient 초기화 완료. Paper: {self.paper}")

            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.api_secret
            )
            current_app.logger.info("Alpaca StockHistoricalDataClient 초기화 완료")

        except Exception as e:
            current_app.logger.error(f"Alpaca 클라이언트 초기화 실패: {e}", exc_info=True)
            raise

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """계정 정보 조회"""
        try:
            account = self.trading_client.get_account()
            account_details = {
                "id": str(account.id),
                "account_number": str(account.account_number),
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "status": str(account.status),
                "day_trade_count": account.day_trade_count,
                "last_equity": float(account.last_equity) if account.last_equity else 0.0,
            }
            current_app.logger.debug(f"계정 정보 조회 성공: {account_details['account_number']}")
            return account_details
        except APIError as e:
            current_app.logger.error(f"Alpaca API 오류 (계정 정보): {e}")
            return None
        except Exception as e:
            current_app.logger.error(f"계정 정보 조회 예외: {e}", exc_info=True)
            return None

    def get_stock_bars(self, symbol: str, timeframe: TimeFrame, limit: int = 100) -> Optional[pd.DataFrame]:
        """주식 바 데이터 조회"""
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            current_app.logger.debug(f"바 데이터 요청: {symbol}, {timeframe}, {limit}개")
            bars_data = self.data_client.get_stock_bars(request_params)

            if bars_data and bars_data.df is not None and not bars_data.df.empty:
                current_app.logger.info(f"{symbol} 바 데이터 {len(bars_data.df)}개 조회 성공")
                return bars_data.df
            else:
                current_app.logger.warning(f"{symbol} 바 데이터 없음")
                return None
                
        except APIError as e:
            current_app.logger.error(f"Alpaca API 오류 ({symbol} 바 데이터): {e}")
            return None
        except Exception as e:
            current_app.logger.error(f"{symbol} 바 데이터 조회 예외: {e}", exc_info=True)
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """현재가 조회 (간단한 래퍼)"""
        try:
            bars_df = self.get_stock_bars(symbol, TimeFrame.Minute, limit=1)
            if bars_df is not None and not bars_df.empty:
                return float(bars_df.iloc[-1]['close'])
            return None
        except Exception as e:
            current_app.logger.error(f"{symbol} 현재가 조회 예외: {e}")
            return None

    # 향후 필요한 다른 Alpaca API 연동 메소드 추가
    # 예: place_order, get_orders, get_positions 등