from celery.result import AsyncResult
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from celery_config import analyze, analyze_hate_speech, analyze_sentiment, analyze_spam
from model import RequestBody, RequestBodyAnalyzeMoreThanOne

app = FastAPI()

@app.get("/")
def mainpage():
    return JSONResponse({"Nama Webservice":"TextAnalyzer!!!",
                         "Kegunaan": "Web service ini bisa digunakan untuk menganalisis sentiment, spam, atau hate speech dari minimal 1 teks masukan",
                         "Bahasa":"Web service ini bisa digunakan untuk semua bahasa"})


@app.get("/tasks/{task_id}")
def get_status(task_id):
    task_result = AsyncResult(task_id)
    result = {
        "task_id": task_id,
        "task_status": task_result.status,
        "task_result": task_result.result
    }
    return JSONResponse(result)

@app.post("/analyze-sentiment")
def run_analyze_sentiment_task(input: RequestBody):
    task = analyze_sentiment.delay(input.text)
    return JSONResponse({"task_id": task.id}, status_code = 202)

@app.post("/analyze-spam")
def run_analyze_spam_task(input: RequestBody):
    task = analyze_spam.delay(input.text)
    return JSONResponse({"task_id": task.id}, status_code = 202)

@app.post("/analyze-hate-speech")
def run_analyze_hate_speech_task(input: RequestBody):
    task = analyze_hate_speech.delay(input.text)
    return JSONResponse({"task_id": task.id}, status_code = 202)

@app.post("/analyze")
def run_analyze_more_than_one_task(input: RequestBodyAnalyzeMoreThanOne):
    task = analyze.delay(input.text, input.analyze_sentiment, input.analyze_spam, input.analyze_hate_speech)
    return JSONResponse({"task_id": task.id}, status_code = 202)
