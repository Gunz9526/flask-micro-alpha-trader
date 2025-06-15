from flask import Flask, jsonify, request
from celery import Celery, Task as CeleryTask
import os
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
from .config import config
# from .middleware import setup_metrics_middleware
# from .services.metrics_service import get_metrics_service
from prometheus_flask_exporter.multiprocess import MultiprocessInternalPrometheusMetrics

load_dotenv()

celery = Celery(__name__, include=['app.tasks'])

config_name = os.environ.get('FLASK_ENV', 'default')
app_config = config[config_name]

from .celery_config import beat_schedule

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
    beat_schedule=beat_schedule,
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
    MultiprocessInternalPrometheusMetrics(app)
    
    setup_logging(app)
    # setup_metrics_middleware(app)

    from . import tasks
    register_routes(app, tasks)

    return app

def setup_logging(app):
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
    
    @app.route('/')
    def hello():
        return jsonify({
            "service": "Alpha Trader",
            "status": "running",
            "paper_trading": app_config.ALPACA_PAPER,
            "features": ["market_data", "ai_analysis", "auto_trading"]
        })

    @app.route('/health')
    def health_check():
        return jsonify({"status": "healthy"})

    @app.route('/api/test/connection')
    def test_alpaca_connection():
        try:
            from .services.alpaca_service import AlpacaService
            service = AlpacaService()
            result = service.test_connection()
            return jsonify(result)
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"연결 테스트 실패: {str(e)}"
            }), 500

    @app.route('/api/account/info')
    def get_account_info():
        try:
            from .services.trading_service import TradingService
            trading_service = TradingService()
            account_info = trading_service.alpaca_service.get_account_info()
            
            if account_info:
                return jsonify({"status": "success", "data": account_info})
            else:
                return jsonify({"status": "error", "message": "계정 정보 조회 실패"}), 500
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route('/api/positions')
    def get_positions():
        try:
            from .services.trading_service import TradingService
            trading_service = TradingService()
            positions = trading_service.get_positions()
            return jsonify(positions)
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route('/api/trading/signal/<symbol>')
    def get_trading_signal(symbol: str):
        try:
            from .services.ai_service import AIService
            from .services.alpaca_service import AlpacaService
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            
            alpaca_service = AlpacaService()
            ai_service = AIService()
            
            bars_df = alpaca_service.get_stock_bars(
                symbol.upper(),
                TimeFrame(1, TimeFrameUnit.Day),
                limit=120
            )
            
            if bars_df is None:
                return jsonify({
                    "error": f"{symbol} 데이터 조회 실패"
                }), 404
            
            signal_result = ai_service.get_trading_signal(symbol.upper(), bars_df)
            
            return jsonify(signal_result)
            
        except Exception as e:
            return jsonify({
                "error": str(e)
            }), 500

    @app.route('/api/trading/execute/<symbol>')
    def execute_trading_signal(symbol: str):
        signal = request.args.get('signal', '').upper()
        confidence = float(request.args.get('confidence', 0.8))
        
        if signal not in ['BUY', 'SELL']:
            return jsonify({"error": "Invalid signal. Use BUY or SELL"}), 400
        
        try:
            from .services.trading_service import TradingService
            trading_service = TradingService()
            
            result = trading_service.execute_ai_signal(symbol.upper(), signal, confidence)
            return jsonify(result)
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route('/api/training/start/<symbol>')
    def start_training(symbol: str):
        try:
            from .services.ai_service import AIService
            from .services.alpaca_service import AlpacaService
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            
            alpaca_service = AlpacaService()
            ai_service = AIService()
            
            bars_df = alpaca_service.get_stock_bars(
                symbol.upper(),
                TimeFrame(1, TimeFrameUnit.Day),
                limit=200
            )
            
            if bars_df is None or len(bars_df) < 100:
                return jsonify({
                    "status": "error",
                    "message": f"{symbol} 데이터 부족"
                }), 400
            
            result = ai_service.train_strategy(symbol.upper(), bars_df)
            
            return jsonify({
                "symbol": symbol.upper(),
                "status": "completed",
                "result": result
            })
            
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500

    @app.route('/api/price/<symbol>')
    def get_current_price(symbol: str):
        try:
            from .services.alpaca_service import AlpacaService
            alpaca_service = AlpacaService()
            price = alpaca_service.get_current_price(symbol.upper())
            
            if price:
                return jsonify({"symbol": symbol.upper(), "price": price})
            else:
                return jsonify({"error": "Price not available"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/market/bars/<symbol>')
    def get_market_data(symbol: str):
        try:
            from .services.alpaca_service import AlpacaService
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            
            limit = int(request.args.get('limit', 50))
            alpaca_service = AlpacaService()
            
            bars_df = alpaca_service.get_stock_bars(
                symbol.upper(),
                TimeFrame(1, TimeFrameUnit.Day),
                limit=limit
            )
            
            if bars_df is not None:
                data = bars_df.tail(10).to_dict('records')
                return jsonify({
                    "symbol": symbol.upper(),
                    "data": data,
                    "total_bars": len(bars_df)
                })
            else:
                return jsonify({"error": "No data available"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/task/<task_id>')
    def get_task_status(task_id: str):
        task_result = celery.AsyncResult(task_id)
        
        if task_result.state == 'PENDING':
            response = {'state': task_result.state, 'status': 'Task is waiting'}
        elif task_result.state == 'SUCCESS':
            response = {'state': task_result.state, 'result': task_result.result}
        else:
            response = {'state': task_result.state, 'status': str(task_result.info)}
        
        return jsonify(response)

    @app.route('/api/trading/auto/start')
    def start_auto_trading():
        task_result = tasks.smart_trading_pipeline.delay()
        return jsonify({
            "task_id": task_result.id,
            "status": "auto_trading_started",
            "message": "자동 트레이딩이 시작되었습니다"
        })

    @app.route('/api/trading/auto/stop')
    def stop_auto_trading():
        return jsonify({
            "status": "auto_trading_stopped",
            "message": "자동 트레이딩이 중지되었습니다"
        })

    @app.route('/api/dashboard')
    def get_dashboard():
        try:
            from .services.trading_service import TradingService
            trading_service = TradingService()
            
            account_info = trading_service.alpaca_service.get_account_info()
            positions = trading_service.get_positions()
            
            dashboard_data = {
                "account": {
                    "portfolio_value": account_info.get('portfolio_value', 0) if account_info else 0,
                    "buying_power": account_info.get('buying_power', 0) if account_info else 0,
                    "cash": account_info.get('cash', 0) if account_info else 0
                },
                "positions": positions.get('positions', []),
                "position_count": positions.get('total_positions', 0),
                "system": {
                    "paper_trading": app_config.ALPACA_PAPER,
                    "ai_model": "LightGBM",
                    "status": "running"
                }
            }
            
            return jsonify(dashboard_data)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # @app.route('/metrics')
    # def metrics():
    #     metrics_service = get_metrics_service()     
    #     return metrics_service.get_metrics(), 200, {'Content-Type': 'text/plain; charset=utf-8'}
    

    @app.route('/api/metrics/update')
    def trigger_metrics_update():
        task_result = tasks.update_system_metrics.delay()
        return jsonify({
            "task_id": task_result.id,
            "status": "metrics_update_started"
        })

    @app.route('/api/risk/status')
    def get_risk_status():
        try:
            from .services.risk_manager import RiskManager
            risk_manager = RiskManager()
            
            status = {
                "trading_halted": risk_manager.is_trading_halted,
                "daily_pnl_count": len(risk_manager.daily_pnl),
                "max_position_size": risk_manager.max_position_size,
                "max_portfolio_risk": risk_manager.max_portfolio_risk
            }
            
            return jsonify(status)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/backtest/<symbol>')
    def quick_backtest(symbol: str):
        from datetime import datetime, timedelta
        
        try:
            from .services.backtest_service import BacktestService
            backtest_service = BacktestService()
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            result = backtest_service.run_backtest(
                symbols=[symbol.upper()],
                start_date=start_date,
                end_date=end_date,
                initial_capital=10000
            )
            
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/system/status')
    def get_system_status():
        try:
            import psutil
            
            status = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "redis_status": "connected",
                "celery_status": "running"
            }
            
            return jsonify(status)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
    @app.route('/api/training/start/batch')
    def start_batch_training():
        task_result = tasks.train_models_batch.delay()
        return jsonify({
            "task_id": task_result.id,
            "status": "batch_training_started",
            "message": "전체 모델 학습이 시작되었습니다"
        })

    @app.route('/api/trading/signal/generate')
    def generate_signals_all():
        task_result = tasks.smart_trading_pipeline.delay()
        return jsonify({
            "task_id": task_result.id,
            "status": "signal_generation_started"
        })

    @app.route('/api/training/retrain', methods=['POST'])
    def retrain_models():
        """지정된 종목 리스트에 대해서만 모델을 재학습시킵니다."""
        data = request.get_json()
        if not data or 'symbols' not in data:
            return jsonify({"error": "요청 본문에 'symbols' 리스트가 필요합니다."}), 400
        
        symbols_to_train = data['symbols']
        task_result = tasks.train_specific_models.delay(symbols=symbols_to_train)
        
        return jsonify({
            "task_id": task_result.id,
            "status": "retraining_started",
            "symbols": symbols_to_train
        })