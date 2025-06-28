from celery.schedules import crontab

beat_schedule = {
    'update-system-metrics': {
        'task': 'app.tasks.update_system_metrics',
        'schedule': 300.0,
    },
    'update-portfolio-metrics': {
        'task': 'app.tasks.update_portfolio_metrics',
        'schedule': 300.0,
    },
    
    'smart-trading-pipeline': {
        'task': 'app.tasks.smart_trading_pipeline',
        'schedule': crontab(minute='*/10', hour='11-23', day_of_week='mon-fri'),  
    },
    
    'daily-report': {
        'task': 'app.tasks.send_daily_report',
        'schedule': crontab(hour=21, minute=5, day_of_week='mon-fri'),
    },
    'parameter-optimization': {
        'task': 'app.tasks.optimize_hyperparameters_for_watchlist',
        'schedule': crontab(hour=22, minute=30, day_of_week='mon-fri'),
    },

    'daily-model-finetuning': {
        'task': 'app.tasks.train_models_batch',
        'schedule': crontab(hour=23, minute=0, day_of_week='mon-fri'),
    },
    
    
    'weekend-full-retraining': {
        'task': 'app.tasks.train_models_batch',
        'schedule': crontab(hour=8, minute=0, day_of_week='sun'),
    },
    'reset-daily-limits': {
    'task': 'app.tasks.reset_risk_limits',
    'schedule': crontab(hour=5, minute=0),
},
}