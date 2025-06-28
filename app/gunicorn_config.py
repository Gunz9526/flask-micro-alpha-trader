import glob
import os
import shutil
import logging.handlers
from prometheus_client import multiprocess

prometheus_metrics_dir = '/prometheus_metrics'
os.environ['PROMETHEUS_MULTIPROC_DIR'] = prometheus_metrics_dir

max_requests = 1000
max_requests_jitter = 100
timeout = 60
workers = 1
keepalive = 5

accesslog = "logs/gunicorn_access.log"
errorlog = "logs/gunicorn_error.log" 
loglevel = "info"                  
capture_output = True
enable_stdio_inheritance = True

access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

def setup_log_rotation():
    access_handler = logging.handlers.RotatingFileHandler(
        'logs/gunicorn_access.log',
        maxBytes=10*1024*1024,
        backupCount=1
    )
    
    error_handler = logging.handlers.RotatingFileHandler(
        'logs/gunicorn_error.log',
        maxBytes=10*1024*1024,
        backupCount=1
    )


def on_starting(server):
    os.makedirs('logs', exist_ok=True)    
    setup_log_rotation()

    if os.path.isdir(prometheus_metrics_dir):
        for f in glob.glob(os.path.join(prometheus_metrics_dir, '*.db')):
            os.remove(f)
    else:
        os.makedirs(prometheus_metrics_dir)

def worker_exit(server, worker):
    multiprocess.mark_process_dead(worker.pid)