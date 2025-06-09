from alpaca.trading.client import TradingClient
from alpaca.common.exceptions import APIError
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from datetime import datetime, timedelta
from flask import current_app
import pandas as pd
import os
from typing import Optional, Dict, Any

class AlpacaService:
    def __init__(self):
        self.api_key = current_app.config.get("ALPACA_API_KEY") or os.environ.get("ALPACA_API_KEY")
        self.api_secret = current_app.config.get("ALPACA_API_SECRET") or os.environ.get("ALPACA_API_SECRET")
        self.base_url = current_app.config.get("APCA_API_BASE_URL") or os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
        self.paper = str(current_app.config.get("ALPACA_PAPER", "True")).lower() in ('true', '1', 't')

        current_app.logger.info(f"Alpaca 설정 확인:")
        current_app.logger.info(f"- API Key: {self.api_key[:8]}..." if self.api_key else "- API Key: None")
        current_app.logger.info(f"- API Secret: {self.api_secret[:8]}..." if self.api_secret else "- API Secret: None")
        current_app.logger.info(f"- Base URL: {self.base_url}")
        current_app.logger.info(f"- Paper Trading: {self.paper}")

        if not self.api_key or not self.api_secret:
            current_app.logger.error("Alpaca API Key 또는 Secret이 설정되지 않았습니다.")
            raise ValueError("Alpaca API Key 또는 Secret이 설정되지 않았습니다.")

        self._init_clients()

    def _init_clients(self):
        try:
            current_app.logger.info("Alpaca TradingClient 초기화 시작...")
            
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper=self.paper,
                raw_data=False
            )
            current_app.logger.info(f"Alpaca TradingClient 초기화 완료. Paper: {self.paper}")

            current_app.logger.info("Alpaca StockHistoricalDataClient 초기화 시작...")
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
                raw_data=False 
            )
            current_app.logger.info("Alpaca StockHistoricalDataClient 초기화 완료")

        except Exception as e:
            current_app.logger.error(f"Alpaca 클라이언트 초기화 실패: {e}", exc_info=True)
            raise

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        try:
            current_app.logger.info("Alpaca 계정 정보 조회 시작...")
            
            account = self.trading_client.get_account()
            
            account_details = {
                "id": str(account.id),
                "account_number": str(account.account_number),
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "status": str(account.status),
                "daytrade_count": getattr(account, 'daytrade_count', 0),
                "last_equity": float(getattr(account, 'last_equity', account.portfolio_value)),
                "equity": float(getattr(account, 'equity', account.portfolio_value)),
                "long_market_value": float(getattr(account, 'long_market_value', 0)),
                "short_market_value": float(getattr(account, 'short_market_value', 0)),
                "initial_margin": float(getattr(account, 'initial_margin', 0)),
                "maintenance_margin": float(getattr(account, 'maintenance_margin', 0)),
                "pattern_day_trader": getattr(account, 'pattern_day_trader', False),
                "trading_blocked": getattr(account, 'trading_blocked', False),
                "transfers_blocked": getattr(account, 'transfers_blocked', False),
                "regt_buying_power": float(getattr(account, 'regt_buying_power', account.buying_power)),
                "daytrading_buying_power": float(getattr(account, 'daytrading_buying_power', account.buying_power)),
                "non_marginable_buying_power": float(getattr(account, 'non_marginable_buying_power', account.cash)),
                "accrued_fees": float(getattr(account, 'accrued_fees', 0)),
                "pending_transfer_out": float(getattr(account, 'pending_transfer_out' ) or 0),
                "sma": float(getattr(account, 'sma', 0))
            }
            
            current_app.logger.info(f"계정 정보 조회 성공: {account_details['account_number']}")
            return account_details
            
        except APIError as e:
            current_app.logger.error(f"Alpaca API 오류 (계정 정보): {e}")
            current_app.logger.error(f"API 오류 상세: status_code={getattr(e, 'status_code', 'N/A')}, message={getattr(e, 'message', 'N/A')}")
            return None
        except Exception as e:
            current_app.logger.error(f"계정 정보 조회 예외: {e}", exc_info=True)
            return None

    def get_stock_bars(self, symbol: str, timeframe: TimeFrame, limit: int = 100) -> Optional[pd.DataFrame]:
        try:
            current_app.logger.info(f"바 데이터 요청 시작: {symbol}, {timeframe}, limit={limit}")
            
            end_time = datetime.now()
            
            if timeframe.amount == 1 and timeframe.unit == TimeFrameUnit.Day:
                start_time = end_time - timedelta(days=max(limit * 2, 365))
            elif timeframe.unit == TimeFrameUnit.Hour:
                start_time = end_time - timedelta(days=max(limit // 6, 30))
            elif timeframe.unit == TimeFrameUnit.Minute:
                start_time = end_time - timedelta(days=max(limit // 390, 7))
            else:
                start_time = end_time - timedelta(days=limit * 2)
            
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_time,
                end=end_time,
                limit=limit,
                adjustment='raw',
                feed=DataFeed.IEX,
                asof=None,
            )
            
            current_app.logger.debug(f"요청 파라미터: start={start_time}, end={end_time}")
            
            bars_data = self.data_client.get_stock_bars(request_params)

            if bars_data and hasattr(bars_data, 'df') and bars_data.df is not None and not bars_data.df.empty:
                df = bars_data.df
                
                if isinstance(df.index, pd.MultiIndex):
                    if symbol in df.index.get_level_values(0):
                        df = df.xs(symbol, level=0)
                
                df = df.sort_index(ascending=True)
                
                current_app.logger.info(f"{symbol} 바 데이터 {len(df)}개 조회 성공")
                current_app.logger.debug(f"데이터 범위: {df.index[0]} ~ {df.index[-1]}")
                return df
            else:
                current_app.logger.warning(f"{symbol} 바 데이터 없음")
                
                if limit < 1000:
                    current_app.logger.info(f"{symbol} 더 긴 기간으로 재시도: limit={limit*2}")
                    return self.get_stock_bars(symbol, timeframe, min(limit*2, 1000))
                
                return None
                
        except APIError as e:
            current_app.logger.error(f"Alpaca API 오류 ({symbol} 바 데이터): {e}")
            return None
        except Exception as e:
            current_app.logger.error(f"{symbol} 바 데이터 조회 예외: {e}", exc_info=True)
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            
            try:
                quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol,feed=DataFeed.IEX)
                quote_data = self.data_client.get_stock_latest_quote(quote_request)
                
                if quote_data and hasattr(quote_data, 'df') and not quote_data.df.empty:
                    latest_quote = quote_data.df.iloc[0]
                    bid_price = float(latest_quote.get('bid_price', 0))
                    ask_price = float(latest_quote.get('ask_price', 0))
                    
                    if bid_price > 0 and ask_price > 0:
                        return (bid_price + ask_price) / 2
                    
            except Exception as quote_error:
                current_app.logger.debug(f"Quote 데이터 조회 실패, 바 데이터로 대체: {quote_error}")
            
            bars_df = self.get_stock_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), limit=1)
            if bars_df is not None and not bars_df.empty:
                return float(bars_df.iloc[-1]['close'])
            
            bars_df = self.get_stock_bars(symbol, TimeFrame(1, TimeFrameUnit.Day), limit=1)
            if bars_df is not None and not bars_df.empty:
                return float(bars_df.iloc[-1]['close'])
                
            return None
        except Exception as e:
            current_app.logger.error(f"{symbol} 현재가 조회 예외: {e}")
            return None

    def test_connection(self) -> Dict[str, Any]:
        current_app.logger.info("Alpaca 연결 테스트 시작...")
        
        try:
            test_result = {
                "status": "testing",
                "config_check": {
                    "api_key_set": bool(self.api_key),
                    "api_secret_set": bool(self.api_secret),
                    "base_url": self.base_url,
                    "paper_trading": self.paper
                },
                "alpaca_api_version": "0.40.1"
            }
            
            current_app.logger.info(f"설정 확인: {test_result['config_check']}")
            
            account = self.get_account_info()
            
            if account:
                test_result.update({
                    "status": "success",
                    "message": "Alpaca API 연결 성공",
                    "account_status": account.get("status"),
                    "account_number": account.get("account_number"),
                    "portfolio_value": account.get("portfolio_value"),
                    "paper_trading": self.paper,
                    "daytrade_count": account.get("daytrade_count"),
                    "pattern_day_trader": account.get("pattern_day_trader"),
                    "buying_power": account.get("buying_power"),
                    "cash": account.get("cash")
                })
                current_app.logger.info("Alpaca 연결 테스트 성공")
            else:
                test_result.update({
                    "status": "error",
                    "message": "계정 정보 조회 실패 - API 인증 또는 네트워크 문제"
                })
                current_app.logger.error("Alpaca 연결 테스트 실패: 계정 정보 조회 불가")
            
            return test_result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "message": f"연결 테스트 실패: {str(e)}",
                "error_type": type(e).__name__
            }
            current_app.logger.error(f"Alpaca 연결 테스트 예외: {e}", exc_info=True)
            return error_result