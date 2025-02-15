from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict

from ..models.conversation import (
    ConversationOptimizeRequest,
    ConversationOptimizeResponse,
    ConversationEvaluateResponse
)
from ..core.config import Settings, get_settings
from ..services.mcts import run_mcts
from ..services.evaluation import evaluate_conversation

router = APIRouter()

@router.post("/optimize", response_model=ConversationOptimizeResponse)
async def optimize_conversation(
    request: ConversationOptimizeRequest,
    settings: Settings = Depends(get_settings)
) -> ConversationOptimizeResponse:
    """
    Optimize a conversation using Monte Carlo Tree Search.
    Returns a list of candidate next moves ranked by their potential effectiveness.
    """
    try:
        # Convert Pydantic models to dict format expected by MCTS
        conversation = [msg.dict() for msg in request.conversation]
        
        # Run MCTS optimization
        candidates = await run_mcts(conversation, request.goal)
        
        return ConversationOptimizeResponse(candidates=candidates)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to optimize conversation: {str(e)}"
        )

@router.post("/evaluate", response_model=ConversationEvaluateResponse)
async def evaluate_conversation_endpoint(
    request: ConversationOptimizeRequest,
    settings: Settings = Depends(get_settings)
) -> ConversationEvaluateResponse:
    """
    Evaluate a conversation based on goal alignment, coherence, and engagement.
    Returns detailed evaluation scores and an aggregated score.
    """
    try:
        # Convert Pydantic models to dict format
        conversation = [msg.dict() for msg in request.conversation]
        
        # Get evaluation
        evaluation, aggregated_score = await evaluate_conversation(
            conversation,
            request.goal
        )
        
        return ConversationEvaluateResponse(
            evaluation=evaluation,
            aggregated_score=aggregated_score
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to evaluate conversation: {str(e)}"
        ) 