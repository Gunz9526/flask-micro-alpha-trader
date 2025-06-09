from celery.schedules import crontab

beat_schedule = {
    'update-system-metrics': {
        'task': 'app.tasks.update_system_metrics',
        'schedule': 120.0,
    },
    
    'smart-trading-pipeline': {
        'task': 'app.tasks.smart_trading_pipeline',
        'schedule': crontab(minute='*/10', hour='9-16', day_of_week='1-5'),
    },
    
    'weekend-model-training': {
        'task': 'app.tasks.train_models_batch',
        'schedule': crontab(hour=3, minute=0, day_of_week=0),
    },
    
    'daily-report': {
        'task': 'app.tasks.send_daily_report',
        'schedule': crontab(hour=17, minute=0, day_of_week='1-5'),
    }
}