from pydantic import BaseModel
from typing import List

class RequestBody(BaseModel):
    text: List[str] = []

class RequestBodyAnalyzeMoreThanOne(BaseModel):
    text: List[str] = []
    analyze_sentiment: bool = False
    analyze_spam: bool = False
    analyze_hate_speech: bool = False