from celery.schedules import crontab

beat_schedule = {
    'update-system-metrics': {
        'task': 'app.tasks.update_system_metrics',
        'schedule': 120.0,
    },
    'update-portfolio-metrics': {
        'task': 'app.tasks.update_portfolio_metrics',
        'schedule': 120.0,
    },
    
    'smart-trading-pipeline': {
        'task': 'app.tasks.smart_trading_pipeline',
        'schedule': crontab(minute='*/10', hour='13-21', day_of_week='mon-fri'),
    },
    
    'daily-report': {
        'task': 'app.tasks.send_daily_report',
        'schedule': crontab(hour=21, minute=5, day_of_week='mon-fri'),
    },

    'daily-model-finetuning': {
        'task': 'app.tasks.train_models_batch',
        'schedule': crontab(hour=21, minute=15, day_of_week='mon-fri'),
    },
    
    'weekend-full-retraining': {
        'task': 'app.tasks.train_models_batch',
        'schedule': crontab(hour=8, minute=0, day_of_week='sat'), # 토요일로 변경
    },
}