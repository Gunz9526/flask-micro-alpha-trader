import os
from typing import Optional

class Config:
    """기본 설정"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Alpaca 설정
    ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY')
    ALPACA_API_SECRET = os.environ.get('ALPACA_API_SECRET')
    ALPACA_PAPER = os.environ.get('ALPACA_PAPER', 'True').lower() in ('true', '1', 't')
    
    # Redis 설정
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://redis:6379/0')
    
    # 로깅 설정
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
    
    # Discord 웹훅
    DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL')
    
    # AI 모델 설정
    AI_MIN_DATA_POINTS = int('30') 
    AI_CONFIDENCE_THRESHOLD = float('0.7')
    AI_SIGNAL_BUY_THRESHOLD = float('0.005')
    AI_SIGNAL_SELL_THRESHOLD = float('-0.005')
    
    # 데이터 디렉토리 설정
    DATA_DIR = os.environ.get('DATA_DIR', 'data')
    
    # 최적화 설정 (새로 추가)
    OPTIMIZER_MAX_MEMORY_MB = int('512')
    OPTIMIZER_MAX_TRIALS = int('50')
    OPTIMIZER_BATCH_SIZE = int('1000')
    OPTIMIZER_MAX_SAMPLES = int('5000')
    
    # 리스크 관리 설정
    RISK_MAX_PORTFOLIO_EXPOSURE = float('0.80')
    RISK_MAX_POSITIONS = int('7')
    RISK_MAX_DAILY_LOSS = float('-0.02')
    RISK_MAX_POSITION_SIZE = float('0.10')
    RISK_VOLATILITY_TARGET = float('0.015')
    RISK_STOP_LOSS_PCT = float('-0.05')
    RISK_TAKE_PROFIT_PCT = float('0.10')

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