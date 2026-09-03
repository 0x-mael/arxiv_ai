from datetime import datetime
from pydantic import BaseModel
from typing import Dict, Literal,Optional


class QueryFormat(BaseModel):
    query: str = "all:paper"
    criteria: Literal["submitteddate", "relevance", "lastUpdateddate"] = "relevance"


class EvaluationFormat(BaseModel):
    score : int
    feedback: str

class ResponseFormat(BaseModel):
    title: str
    authors: list[str]
    published_date: str
    summary: Optional[str] = None
    pdf_url: str