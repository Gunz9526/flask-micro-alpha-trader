from flask import Flask, jsonify, request
from celery import Celery, Task as CeleryTask
import os
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
from .config import config

load_dotenv()

celery = Celery(__name__, include=['app.tasks'])

config_name = os.environ.get('FLASK_ENV', 'default')
app_config = config[config_name]


celery.conf.update(
    broker_url=app_config.REDIS_URL,
    result_backend=app_config.REDIS_URL,
    task_ignore_result=False,
    task_track_started=True,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

class FlaskTask(CeleryTask):
    _app = None

    def __call__(self, *args: object, **kwargs: object) -> object:
        if FlaskTask._app is None:
            FlaskTask._app = create_app()

        with FlaskTask._app.app_context():
            return super().__call__(*args, **kwargs)
        
celery.Task = FlaskTask

def create_app() -> Flask:
    app = Flask(__name__)
    
    
    flask_config_name = os.environ.get('FLASK_ENV', 'default')
    
    app.config.from_object(config[flask_config_name])
    app.extensions["celery"] = celery

    
    setup_logging(app)

    
    from . import tasks

    
    register_routes(app, tasks)

    return app

def setup_logging(app):
    """로깅 설정"""
    log_level_str = app_config.LOG_LEVEL.upper()
    numeric_level = getattr(logging, log_level_str, logging.INFO)

    if not app.debug and not app.testing:
        if not os.path.exists('logs'):
            os.makedirs('logs', exist_ok=True)
        
        file_handler = RotatingFileHandler('logs/alpha_trader.log', maxBytes=1024*1024, backupCount=10)
        log_format = logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        )
        file_handler.setFormatter(log_format)
        file_handler.setLevel(numeric_level)
        app.logger.addHandler(file_handler)

    app.logger.setLevel(numeric_level)
    app.logger.info(f'Alpha Trader 시작. 로그 레벨: {log_level_str}')

def register_routes(app, tasks):
    """라우트 등록"""
    
    @app.route('/')
    def hello():
        app.logger.info("루트 엔드포인트 호출")
        return jsonify({
            "service": "Alpha Trader",
            "status": "running",
            "paper_trading": app_config.ALPACA_PAPER
        })

    @app.route('/health')
    def health_check():
        """헬스 체크 엔드포인트"""
        return jsonify({
            "status": "healthy",
            "timestamp": "2025-06-06T00:00:00Z"
        })

    @app.route('/api/account/info')
    def get_account_info():
        """계정 정보 조회 API"""
        app.logger.info("계정 정보 조회 API 호출")
        task_result = tasks.get_alpaca_account_info_task.delay()
        
        return jsonify({
            "task_id": task_result.id,
            "status": "submitted",
            "message": "계정 정보 조회 완료"
        })

    @app.route('/api/market/bars/<symbol>')
    def fetch_market_bars(symbol: str):
        """마켓 데이터 조회 API"""
        timeframe_unit = request.args.get('timeframe_unit', 'Day')
        timeframe_value = int(request.args.get('timeframe_value', 1))
        limit = int(request.args.get('limit', 10))
        
        app.logger.info(f"마켓 데이터 조회 API 호출: {symbol}")
        
        task_result = tasks.fetch_stock_bars_task.delay(
            symbol=symbol.upper(),
            timeframe_unit_str=timeframe_unit,
            timeframe_value=timeframe_value,
            limit=limit
        )
        
        return jsonify({
            "task_id": task_result.id,
            "symbol": symbol.upper(),
            "status": "submitted",
            "message": f"{symbol.upper()} 데이터 조회 작업 보냄"
        })

    @app.route('/api/task/<task_id>')
    def get_task_status(task_id: str):
        """작업 상태 조회 API"""
        task_result = celery.AsyncResult(task_id)
        
        if task_result.state == 'PENDING':
            response = {
                'state': task_result.state,
                'status': 'Task 대기중'
            }
        elif task_result.state == 'SUCCESS':
            response = {
                'state': task_result.state,
                'result': task_result.result
            }
        else:
            response = {
                'state': task_result.state,
                'status': str(task_result.info)
            }
        
        return jsonify(response)

    @app.route('/api/market/health')
    def trigger_market_health_check():
        """시장 상태 체크 트리거"""
        app.logger.info("시장 상태 체크 트리거")
        task_result = tasks.market_health_check.delay()
        
        return jsonify({
            "task_id": task_result.id,
            "status": "submitted",
            "message": "시장 상태 체크가 시작되었습니다"
        })