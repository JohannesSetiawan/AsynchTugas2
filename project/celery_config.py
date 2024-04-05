import os
import time

from celery import Celery

broker_url = os.environ.get("CELERY_BROKER_URL", 'redis://localhost:6379')
result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379")
celery = Celery(__name__, broker=broker_url, backend=result_backend)
celery.conf.update(result_persistent=True)


@celery.task
def divide(x, y):
    import time
    time.sleep(5)
    return x / y