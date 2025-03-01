from pydantic import BaseModel
from typing import List, Dict, Optional, Literal, Union

class Article(BaseModel):
    title: str
    content: str
    source: Optional[str] = None

class Argument(BaseModel):
    content: str
    position: Literal["pro", "con"]
    number: int
    verified: bool = False

class DebateState(BaseModel):
    article: Article
    summary: Optional[str] = None
    arguments: List[Argument] = []
    current_turn: Literal["pro", "con"] = "pro"
    pro_count: int = 0
    con_count: int = 0
    user_inputs: List[str] = []
    is_active: bool = True
    iteration_count: int = 0 