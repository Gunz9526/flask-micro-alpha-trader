from datetime import timedelta, datetime
import time
import json
from flask import Flask, jsonify, request, g
from celery import Celery, Task as CeleryTask
import os
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
from .config import config
from werkzeug.security import check_password_hash
from flask_jwt_extended import create_access_token, jwt_required, JWTManager
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
    app.config['JWT_SECRET_KEY'] = app_config.SECRET_KEY
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)  # 24시간 유효
    app.config['JWT_ALGORITHM'] = 'HS256'
    
    jwt = JWTManager(app)

    app.extensions["celery"] = celery
    MultiprocessInternalPrometheusMetrics(app)
    
    setup_logging(app)
    # setup_metrics_middleware(app)    


    @app.before_request
    def log_request_info():
        g.start_time = time.time()
        g.client_ip = get_client_ip()
        
        if request.path.startswith('/api/'):
            app.logger.info(
                f"API 요청 - IP: {g.client_ip}, "
                f"Endpoint: {request.endpoint}, "
                f"Method: {request.method}, "
                f"Path: {request.path}, "
                f"Args: {dict(request.args)}, "
                f"User-Agent: {request.headers.get('User-Agent', 'Unknown')}, "
                f"Content-Length: {request.content_length or 0}"
            )
    
    @app.after_request
    def log_response_info(response):
        if hasattr(g, 'start_time') and request.path.startswith('/api/'):
            duration = time.time() - g.start_time
            app.logger.info(
                f"API 응답 - IP: {g.client_ip}, "
                f"Status: {response.status_code}, "
                f"Duration: {duration:.3f}s, "
                f"Content-Length: {response.content_length or 0}, "
                f"Content-Type: {response.content_type}"
            )
        return response

    # JWT 에러 핸들러
    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return jsonify({"error": "Token has expired", "code": "token_expired"}), 401

    @jwt.invalid_token_loader
    def invalid_token_callback(error):
        return jsonify({"error": "Invalid token", "code": "invalid_token"}), 401

    @jwt.unauthorized_loader
    def missing_token_callback(error):
        return jsonify({"error": "Authorization token is required", "code": "authorization_required"}), 401
    
    from . import tasks
    register_routes(app, tasks)

    return app

def get_client_ip():
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    elif request.headers.get('X-Real-IP'):
        return request.headers.get('X-Real-IP')
    else:
        return request.remote_addr

def validate_master_credentials(username, password):
    master_username = os.environ.get('MASTER_USERNAME')
    master_password_hash = os.environ.get('MASTER_PASSWORD_HASH')
    
    if not master_username or not master_password_hash:
        return False

    if username == master_username and check_password_hash(master_password_hash, password):
        return True
    
    return False


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

    @app.route('/api/auth/login', methods=['POST'])
    def login():
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "Request body is required"}), 400
            
            username = data.get("username")
            password = data.get("password")
            
            if not username or not password:
                return jsonify({"error": "Username and password are required"}), 400
            
            if not validate_master_credentials(username, password):
                app.logger.warning(f"Failed login attempt from IP: {get_client_ip()}, Username: {username}")
                return jsonify({"error": "Invalid credentials"}), 401
            
            additional_claims = {
                "user_type": "master",
                "login_ip": get_client_ip(),
                "login_time": datetime.now().isoformat()
            }
            
            access_token = create_access_token(
                identity=username,
                additional_claims=additional_claims
            )
            
            app.logger.info(f"Successful login - IP: {get_client_ip()}, Username: {username}")
            
            return jsonify({
                "access_token": access_token,
                "token_type": "Bearer",
                "expires_in": 86400,
                "user_type": "master"
            })
            
        except Exception as e:
            app.logger.error(f"Login error: {e}", exc_info=True)
            return jsonify({"error": "Internal server error"}), 500

    @app.route('/api/test/connection')
    @jwt_required()
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
    @jwt_required()
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
    @jwt_required()
    def get_positions():
        try:
            from .services.trading_service import TradingService
            trading_service = TradingService()
            positions = trading_service.get_positions()
            return jsonify(positions)
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route('/api/trading/signal/<symbol>')
    @jwt_required()
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
    @jwt_required()
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
    @jwt_required()
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
    @jwt_required()
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
    @jwt_required()
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
    @jwt_required()
    def get_task_status(task_id: str):
        task_result = celery.AsyncResult(task_id)
        
        if task_result.state == 'PENDING':
            response = {'state': task_result.state, 'status': 'Task is waiting'}
        elif task_result.state == 'SUCCESS':
            response = {'state': task_result.state, 'result': task_result.result}
        else:
            response = {'state': task_result.state, 'status': str(task_result.info)}
        
        return jsonify(response)

    # @app.route('/metrics')
    # def metrics():
    #     metrics_service = get_metrics_service()     
    #     return metrics_service.get_metrics(), 200, {'Content-Type': 'text/plain; charset=utf-8'}
    

    @app.route('/api/metrics/update')
    @jwt_required()
    def trigger_metrics_update():
        task_result = tasks.update_system_metrics.delay()
        return jsonify({
            "task_id_1": task_result.id,
            "task_id_2": tasks.update_portfolio_metrics.delay().id,
            "status": "metrics_update_started"
        })

    @app.route('/api/risk/status')
    @jwt_required()
    def get_risk_status():
        try:
            from .services.risk_manager import RiskManager
            from .services.trading_service import TradingService
            
            risk_manager = RiskManager()
            trading_service = TradingService()
            
            basic_status = risk_manager.get_status()
            
            account_info = trading_service.alpaca_service.get_account_info()
            positions = trading_service.get_positions()
            
            if account_info:
                portfolio_value = float(account_info['portfolio_value'])
                
                total_position_value = sum(
                    float(pos.get('market_value', 0)) 
                    for pos in positions.get('positions', [])
                )
                current_exposure = total_position_value / portfolio_value if portfolio_value > 0 else 0
                
                risk_manager.update_daily_pnl(portfolio_value)
                
                comprehensive_status = {
                    **basic_status,
                    "portfolio": {
                        "value": portfolio_value,
                        "cash": float(account_info.get('cash', 0)),
                        "buying_power": float(account_info.get('buying_power', 0)),
                        "current_exposure": f"{current_exposure:.1%}",
                        "exposure_limit": f"{risk_manager.max_portfolio_exposure:.1%}",
                        "exposure_remaining": f"{max(0, risk_manager.max_portfolio_exposure - current_exposure):.1%}"
                    },
                    "positions": {
                        "count": len(positions.get('positions', [])),
                        "limit": risk_manager.max_positions,
                        "remaining": max(0, risk_manager.max_positions - len(positions.get('positions', [])))
                    },
                    "risk_alerts": []
                }
                
                alerts = []
                
                if risk_manager.is_trading_halted:
                    alerts.append({
                        "level": "critical",
                        "message": "거래 중단됨 - 일일 손실 한도 초과",
                        "pnl": f"{basic_status['daily_pnl']:.2%}"
                    })
                
                if current_exposure > 0.7:
                    alerts.append({
                        "level": "warning", 
                        "message": f"높은 포트폴리오 노출도: {current_exposure:.1%}",
                        "limit": f"{risk_manager.max_portfolio_exposure:.1%}"
                    })
                
                if basic_status['trade_count_today'] > basic_status['max_daily_trades'] * 0.8:
                    alerts.append({
                        "level": "info",
                        "message": f"일일 거래 한도 80% 도달: {basic_status['trade_count_today']}/{basic_status['max_daily_trades']}"
                    })
                
                if basic_status['daily_pnl'] < -0.01:
                    alerts.append({
                        "level": "warning",
                        "message": f"일일 손실 발생: {basic_status['daily_pnl']:.2%}"
                    })
                
                comprehensive_status["risk_alerts"] = alerts
                
                
                return jsonify(comprehensive_status)
            else:
                return jsonify({
                    **basic_status,
                    "error": "계정 정보 조회 실패",
                    "system_info": {
                        "risk_manager_type": "in_memory",
                        "redis_dependency": False
                    }
                }), 500
                
        except Exception as e:
            app.logger.error(f"리스크 상태 조회 오류: {e}", exc_info=True)
            return jsonify({
                "error": str(e),
                "system_info": {
                    "risk_manager_type": "in_memory",
                    "redis_dependency": False
                }
            }), 500

    @app.route('/api/backtest/<symbol>')
    @jwt_required()
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
    @jwt_required()
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
    @jwt_required()
    def start_batch_training():
        task_result = tasks.train_models_batch.delay()
        return jsonify({
            "task_id": task_result.id,
            "status": "batch_training_started",
            "message": "전체 모델 학습이 시작되었습니다"
        })

    @app.route('/api/trading/signal/generate')
    @jwt_required()
    def generate_signals_all():
        task_result = tasks.smart_trading_pipeline.delay()
        return jsonify({
            "task_id": task_result.id,
            "status": "signal_generation_started"
        })

    @app.route('/api/training/retrain', methods=['POST'])
    @jwt_required()
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
    
    @app.route('/api/hyperparameter/optimize/')
    @jwt_required()
    def optimize_hyperparameters():
        task_result = tasks.optimize_hyperparameters_for_watchlist.delay()
        return jsonify({
            "task_id": task_result.id,
            "status": task_result['status'],
            "message": task_result['message']
        })
    
    @app.route('/api/optimization/status')
    @jwt_required()
    def get_optimization_status():
        """최적화 파라미터 사용 현황 확인"""
        try:
            from .services.ai_service import AIService
            
            ai_service = AIService()
            status = {
                "optimization_files": {},
                "model_files": {},
                "summary": {
                    "symbols_with_optimization": 0,
                    "symbols_with_models": 0,
                    "total_optimization_files": 0,
                    "total_model_files": 0
                }
            }
            
            import glob
            optimization_files = glob.glob("best_params/*.json")
            for file_path in optimization_files:
                filename = os.path.basename(file_path)
                parts = filename.replace('.json', '').split('_', 1)
                if len(parts) == 2:
                    symbol, model_class = parts
                    if symbol not in status["optimization_files"]:
                        status["optimization_files"][symbol] = []
                    status["optimization_files"][symbol].append(model_class)
            
            model_files = glob.glob("models/*.pkl")
            for file_path in model_files:
                filename = os.path.basename(file_path)
                if not filename.endswith('_scaler.pkl'):
                    parts = filename.replace('.pkl', '').split('_', 1)
                    if len(parts) == 2:
                        symbol, model_name = parts
                        if symbol not in status["model_files"]:
                            status["model_files"][symbol] = []
                        status["model_files"][symbol].append(model_name)
            
            status["summary"]["symbols_with_optimization"] = len(status["optimization_files"])
            status["summary"]["symbols_with_models"] = len(status["model_files"])
            status["summary"]["total_optimization_files"] = len(optimization_files)
            status["summary"]["total_model_files"] = len([f for f in model_files if not f.endswith('_scaler.pkl')])
            
            return jsonify(status)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/optimization/validate/<symbol>')
    @jwt_required()
    def validate_optimization_usage(symbol: str):
        """특정 종목의 최적화 파라미터 사용 검증"""
        try:
            from .services.ai_service import AIService, LightGBMStrategy, XGBoostStrategy
            
            symbol = symbol.upper()
            validation_result = {
                "symbol": symbol,
                "optimization_files": {},
                "load_test": {},
                "recommendations": []
            }
            
            for strategy_class in [LightGBMStrategy, XGBoostStrategy]:
                class_name = strategy_class.__name__
                params_path = os.path.join("best_params", f"{symbol}_{class_name}.json")
                
                validation_result["optimization_files"][class_name] = {
                    "exists": os.path.exists(params_path),
                    "path": params_path
                }
                
                if os.path.exists(params_path):
                    try:
                        with open(params_path, 'r') as f:
                            params = json.load(f)
                            validation_result["optimization_files"][class_name]["params_count"] = len(params)
                            validation_result["optimization_files"][class_name]["params"] = params
                    except Exception as e:
                        validation_result["optimization_files"][class_name]["error"] = str(e)
            
            ai_service = AIService()
            for model_name, config in ai_service.model_classes.items():
                StrategyClass = config['class']
                class_name = StrategyClass.__name__
                
                strategy = StrategyClass(symbol=symbol, random_state=42)
                model_type = strategy.model_type
                
                validation_result["load_test"][model_name] = {
                    "class_name": class_name,
                    "model_type": model_type,
                    "optimization_available": validation_result["optimization_files"][class_name]["exists"]
                }
            
            missing_optimizations = [k for k, v in validation_result["optimization_files"].items() if not v["exists"]]
            if missing_optimizations:
                validation_result["recommendations"].append(f"최적화 누락: {', '.join(missing_optimizations)}")
            else:
                validation_result["recommendations"].append("모든 모델에 대한 최적화 파라미터 사용 가능")
            
            return jsonify(validation_result)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        


    @app.route('/dashboard')
    @jwt_required()
    def dashboard_page():
        """대시보드 HTML 페이지를 렌더링합니다."""
        from flask import render_template
        return render_template('index.html')

    @app.route('/api/dashboard/all_data')
    @jwt_required()
    def get_all_dashboard_data():
        """대시보드에 필요한 모든 데이터를 한 번에 제공합니다."""
        try:
            from .services.database_service import get_trade_database
            from .services.risk_manager import RiskManager
            from .services.alpaca_service import AlpacaService
            
            db_service = get_trade_database()
            risk_manager = RiskManager()
            alpaca_service = AlpacaService()

            # 1. 계좌 및 포지션 정보
            account_info = alpaca_service.get_account_info() or {}
            positions_info = alpaca_service.trading_client.get_all_positions()
            positions_list = [{
                "symbol": p.symbol, "qty": float(p.qty), "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl), "unrealized_plpc": float(p.unrealized_plpc)
            } for p in positions_info]

            # 2. 리스크 관리 상태
            risk_status = risk_manager.get_status()

            # 3. 거래 요약 (DB)
            summary_30d = db_service.get_trading_summary(days=30)
            summary_today = db_service.get_trading_summary(days=1)

            # 4. 일별 성과 차트 데이터 (DB)
            daily_perf_data = db_service.get_daily_performance(days=30)
            daily_perf_data.reverse() # 날짜 오름차순 정렬
            
            daily_chart = {
                "labels": [item['date'] for item in daily_perf_data],
                "datasets": [
                    {
                        "label": "Daily Realized PnL (USD)",
                        "data": [item['realized_pnl'] for item in daily_perf_data],
                        "borderColor": '#0d6efd', "backgroundColor": 'rgba(13, 110, 253, 0.2)',
                        "type": 'line', "yAxisID": 'y_pnl'
                    },
                    {
                        "label": "Daily Trade Volume (USD)",
                        "data": [item['volume'] for item in daily_perf_data],
                        "borderColor": '#6c757d', "backgroundColor": 'rgba(108, 117, 125, 0.5)',
                        "type": 'bar', "yAxisID": 'y_volume'
                    }
                ]
            }

            # 5. 종목별 성과 차트 데이터 (DB)
            symbol_perf_data = db_service.get_symbol_performance()
            top_10_symbols = symbol_perf_data[:10]
            symbol_chart = {
                "labels": [item['symbol'] for item in top_10_symbols],
                "datasets": [{
                    "label": "Trade Volume by Symbol",
                    "data": [item['total_volume'] for item in top_10_symbols],
                    "backgroundColor": ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#C9CBCF', '#7BC225', '#FF6347', '#4682B4']
                }]
            }

            # 최종 응답 데이터 구성
            response_data = {
                "account": account_info,
                "positions": positions_list,
                "risk": risk_status,
                "summary_30d": summary_30d,
                "summary_today": summary_today,
                "charts": {
                    "daily_performance": daily_chart,
                    "symbol_performance": symbol_chart
                }
            }
            return jsonify(response_data)

        except Exception as e:
            app.logger.error(f"통합 대시보드 API 오류: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500