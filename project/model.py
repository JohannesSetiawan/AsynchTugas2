from pydantic import BaseModel
from typing import List

class RequestBody(BaseModel):
    text: List[str] = []

class RequestBodyAnalyzeMoreThanOne(BaseModel):
    text: List[str] = []
    analyze_sentiment: bool
    analyze_spam: bool
    analyze_hate_speech: bool