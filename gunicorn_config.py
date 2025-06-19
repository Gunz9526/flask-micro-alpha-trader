import glob
import os
import shutil
from prometheus_client import multiprocess

prometheus_metrics_dir = '/prometheus_metrics'
os.environ['PROMETHEUS_MULTIPROC_DIR'] = prometheus_metrics_dir

def on_starting(server):
    if os.path.isdir(prometheus_metrics_dir):
        for f in glob.glob(os.path.join(prometheus_metrics_dir, '*.db')):
            os.remove(f)
    else:
        os.makedirs(prometheus_metrics_dir)

def worker_exit(server, worker):
    multiprocess.mark_process_dead(worker.pid)