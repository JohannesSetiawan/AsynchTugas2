# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")
# async def root()
from celery.result import AsyncResult
from fastapi import Body, FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from celery_config import divide


app = FastAPI()



@app.get("/")
def home(request: Request):
    return JSONResponse({"jalan": "jalan"})


@app.post("/tasks", status_code=201)
def run_task():
    task = divide.delay(1, 2)
    return JSONResponse({"task_id": task.id})


@app.get("/tasks/{task_id}")
def get_status(task_id):
    task_result = AsyncResult(task_id)
    result = {
        "task_id": task_id,
        "task_status": task_result.status,
        "task_result": task_result.result
    }
    return JSONResponse(result)
