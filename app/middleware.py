from flask import request, g
import time
from .services.metrics_service import get_metrics_service

def setup_metrics_middleware(app):
    """메트릭 수집 미들웨어 설정"""
    
    @app.before_request
    def before_request():
        g.start_time = time.time()
    
    @app.after_request
    def after_request(response):
        # API 요청 메트릭 수집
        if hasattr(g, 'start_time'):
            duration = time.time() - g.start_time
            metrics_service = get_metrics_service()
            
            metrics_service.record_api_request(
                endpoint=request.endpoint or 'unknown',
                method=request.method,
                status_code=response.status_code,
                duration=duration
            )
        
        return response