import os
from typing import Optional

class Config:
    """기본 설정"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY')
    ALPACA_API_SECRET = os.environ.get('ALPACA_API_SECRET')
    # APCA_API_BASE_URL = os.environ.get('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
    
    # ALPACA_DATA_URL = os.environ.get('ALPACA_DATA_URL', 'https://data.alpaca.markets')
    ALPACA_PAPER = os.environ.get('ALPACA_PAPER', 'True').lower() in ('true', '1', 't')
    
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://redis:6379/0')
    
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
    
    DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL')
    
    AI_MODEL_TYPE = os.environ.get('AI_MODEL_TYPE', 'lightgbm')
    AI_MIN_DATA_POINTS = int(os.environ.get('AI_MIN_DATA_POINTS', '30')) 
    AI_CONFIDENCE_THRESHOLD = float(os.environ.get('AI_CONFIDENCE_THRESHOLD', '0.65'))
    AI_SIGNAL_BUY_THRESHOLD = float(os.environ.get('AI_SIGNAL_BUY_THRESHOLD', '0.005'))
    AI_SIGNAL_SELL_THRESHOLD = float(os.environ.get('AI_SIGNAL_SELL_THRESHOLD', '-0.005'))
    AI_CONFIDENCE_THRESHOLD = float(os.environ.get('AI_CONFIDENCE_THRESHOLD', '0.6'))

    RISK_MAX_PORTFOLIO_EXPOSURE = float(os.environ.get('RISK_MAX_PORTFOLIO_EXPOSURE', '0.80'))
    RISK_MAX_POSITIONS = int(os.environ.get('RISK_MAX_POSITIONS', '7'))
    RISK_MAX_DAILY_LOSS = float(os.environ.get('RISK_MAX_DAILY_LOSS', '-0.02'))

    RISK_MAX_POSITION_SIZE = float(os.environ.get('RISK_MAX_POSITION_SIZE', '0.10'))
    RISK_VOLATILITY_TARGET = float(os.environ.get('RISK_VOLATILITY_TARGET', '0.015'))
    RISK_STOP_LOSS_PCT = float(os.environ.get('RISK_STOP_LOSS_PCT', '-0.05'))
    RISK_TAKE_PROFIT_PCT = float(os.environ.get('RISK_TAKE_PROFIT_PCT', '0.10'))
    

class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = 'INFO'
    AI_CONFIDENCE_THRESHOLD = 0.75

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}