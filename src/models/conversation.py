from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class Message(BaseModel):
    speaker: str = Field(..., description="The speaker of the message")
    text: str = Field(..., description="The content of the message")

class ConversationOptimizeRequest(BaseModel):
    conversation: List[Message] = Field(..., description="List of conversation messages")
    goal: str = Field(..., description="The conversation goal to optimize for")

class MoveCandidate(BaseModel):
    move: str = Field(..., description="The suggested next move")
    avg_score: float = Field(..., description="Average score for this move")
    visits: int = Field(..., description="Number of simulations for this move")

class ConversationOptimizeResponse(BaseModel):
    candidates: List[MoveCandidate] = Field(..., description="List of candidate moves")

class EvaluationDetails(BaseModel):
    goal_alignment: float = Field(..., ge=0, le=100, description="Goal alignment score")
    coherence: float = Field(..., ge=0, le=100, description="Coherence score")
    engagement: float = Field(..., ge=0, le=100, description="Engagement score")

class ConversationEvaluateResponse(BaseModel):
    evaluation: EvaluationDetails
    aggregated_score: float = Field(..., description="Overall aggregated score") 