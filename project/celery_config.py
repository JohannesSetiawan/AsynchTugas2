import os
from analyzer.analyzer import *

from celery import Celery

broker_url = os.environ.get("CELERY_BROKER_URL", 'redis://localhost:6379')
result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379")
celery = Celery(__name__, broker=broker_url, backend=result_backend)
celery.conf.update(result_persistent=True)

def checkIsEmpty(input):
    return len(input) == 0

@celery.task
def analyze_sentiment(input_text):
    if checkIsEmpty(input_text):
        return {'Error': 'Empty input!'}
    else:
        result = sentiment_analysis(input_text)
        return result

@celery.task
def analyze_spam(input_text):
    if checkIsEmpty(input_text):
        return {'Error': 'Empty input!'}
    else:
        result = spam_analysis(input_text)
        return result

@celery.task
def analyze_hate_speech(input_text):
    if checkIsEmpty(input_text):
        return {'Error': 'Empty input!'}
    else:
        result = hate_speech_offensive_languange_analysis(input_text)
        return result

@celery.task
def analyze(input_text, analyze_sent, analyze_spam, analyze_speech):
    if checkIsEmpty(input_text):
        return {'Error': 'Empty input!'}
    elif not (analyze_sent or analyze_spam or analyze_speech):
        return {'Error': 'Nothing to analyze!'}
    else:
        result = perform_analysis(input_text, analyze_sent, analyze_spam, analyze_speech)
        return result
