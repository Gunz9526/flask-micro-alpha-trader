import os
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest, multiprocess
from flask import current_app
import psutil
import time
from typing import Dict, Any
from datetime import datetime

class MetricsService:
    """Prometheus 메트릭 수집 서비스"""
    
    def __init__(self):
        metrics_dir = os.environ.get('PROMETHEUS_MULTIPROC_DIR')
        if metrics_dir and not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
            
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        self.registry = registry
        
        self.trading_signals_total = Counter(
            'trading_signals_total',
            'Total number of trading signals generated',
            ['symbol', 'signal_type'],
            registry=self.registry
        )
        
        self.trades_executed_total = Counter(
            'trades_executed_total',
            'Total number of trades executed',
            ['symbol', 'side', 'status'],
            registry=self.registry
        )
        
        self.ai_model_predictions = Histogram(
            'ai_model_prediction_confidence',
            'AI model prediction confidence scores',
            ['symbol', 'model_type'],
            registry=self.registry
        )
        
        self.portfolio_value = Gauge(
            'portfolio_value_usd',
            'Current portfolio value in USD',
            registry=self.registry
        )
        
        self.position_count = Gauge(
            'active_positions_count',
            'Number of active positions',
            registry=self.registry
        )
        
        self.celery_tasks_total = Counter(
            'celery_tasks_total',
            'Total number of Celery tasks',
            ['task_name', 'status'],
            registry=self.registry
        )
        
        self.api_requests_total = Counter(
            'api_requests_total',
            'Total API requests',
            ['endpoint', 'method', 'status_code'],
            registry=self.registry
        )
        
        self.api_request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration',
            ['endpoint'],
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.app_info = Info(
            'alpha_trader_info',
            'Alpha Trader application information',
            registry=self.registry
        )
        
        self.app_info.info({
            'version': '1.0.0',
            'ai_model': 'LightGBM',
            'paper_trading': str(True)
        })
    
    def record_trading_signal(self, symbol: str, signal_type: str):
        self.trading_signals_total.labels(symbol=symbol, signal_type=signal_type).inc()
        current_app.logger.debug(f"메트릭 기록: 트레이딩 시그널 {symbol} {signal_type}")
    
    def record_trade_execution(self, symbol: str, side: str, status: str):
        self.trades_executed_total.labels(symbol=symbol, side=side, status=status).inc()
        current_app.logger.debug(f"메트릭 기록: 거래 실행 {symbol} {side} {status}")
    
    def record_ai_prediction(self, symbol: str, model_type: str, confidence: float):
        self.ai_model_predictions.labels(symbol=symbol, model_type=model_type).observe(confidence)
        current_app.logger.debug(f"메트릭 기록: AI 예측 {symbol} {confidence:.3f}")
    
    def update_portfolio_metrics(self, portfolio_value: float, position_count: int):
        self.portfolio_value.set(portfolio_value)
        self.position_count.set(position_count)
        current_app.logger.debug(f"메트릭 업데이트: 포트폴리오 ${portfolio_value:.2f}, 포지션 {position_count}개")
    
    def record_celery_task(self, task_name: str, status: str):
        self.celery_tasks_total.labels(task_name=task_name, status=status).inc()
    
    def record_api_request(self, endpoint: str, method: str, status_code: int, duration: float):
        self.api_requests_total.labels(endpoint=endpoint, method=method, status_code=str(status_code)).inc()
        self.api_request_duration.labels(endpoint=endpoint).observe(duration)
    
    def update_system_metrics(self):
        try:
            memory = psutil.virtual_memory()
            self.system_memory_usage.set(memory.percent)
            
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_usage.set(cpu_percent)
            
            current_app.logger.debug(f"시스템 메트릭: Memory {memory.percent:.1f}%, CPU {cpu_percent:.1f}%")
        except Exception as e:
            current_app.logger.error(f"시스템 메트릭 수집 오류: {e}")
    
    def get_metrics(self) -> str:
        return generate_latest(self.registry).decode('utf-8')

metrics_service = None

def get_metrics_service() -> MetricsService:
    global metrics_service
    if metrics_service is None:
        metrics_service = MetricsService()
    return metrics_service