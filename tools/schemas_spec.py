from datetime import datetime
from pydantic import BaseModel
from typing import Dict, Literal,Optional


class QueryFormat(BaseModel):
    query:str
    query_prefix:Optional[Literal["ti:","au:"]]
    criteria : Literal["submitteddate", "relevance", "lastUpdateddate"]


class EvaluationFormat(BaseModel):
    evaluator_name : str
    score : int
    review: str

class ResponseFormat(BaseModel):
    title:str
    authors : list[str]
    published_date: datetime
    pdf_url : str