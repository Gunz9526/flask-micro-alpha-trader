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
    
    AI_MODEL_TYPE = os.environ.get('AI_MODEL_TYPE', 'lightgbm')  # lightgbm, randomforest
    AI_MIN_DATA_POINTS = int(os.environ.get('AI_MIN_DATA_POINTS', '30'))  # LightGBM은 더 적은 데이터로 가능
    AI_CONFIDENCE_THRESHOLD = float(os.environ.get('AI_CONFIDENCE_THRESHOLD', '0.65'))  # 65% 이상

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