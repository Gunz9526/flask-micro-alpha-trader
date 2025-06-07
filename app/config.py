import os
from typing import Optional

class Config:
    """기본 설정"""
    
    # Alpaca 설정
    ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY')
    ALPACA_API_SECRET = os.environ.get('ALPACA_API_SECRET')
    APCA_API_BASE_URL = os.environ.get('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets/v2')
    ALPACA_PAPER = os.environ.get('ALPACA_PAPER', 'True').lower() in ('true', '1', 't')
    
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://redis:6379/0')
    
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
    
    DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL')

class DevelopmentConfig(Config):
    """개발 환경 설정"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """운영 환경 설정"""
    DEBUG = False
    LOG_LEVEL = 'INFO'

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}