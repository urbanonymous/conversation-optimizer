import json
from typing import Dict, List, Tuple

from .llm import async_call_llm, LLMError
from ..core.config import settings

async def async_evaluate_conversation_multi(
    conversation: List[Dict[str, str]],
    goal: str
) -> Tuple[Dict[str, float], float]:
    """
    Evaluate a conversation on multiple dimensions:
    - goal_alignment: How well the conversation addresses the goal
    - coherence: How coherent the conversation is
    - engagement: How engaging the conversation is
    
    Returns:
        Tuple of (evaluation_details, aggregated_score)
    """
    system_message = {
        "role": "system",
        "content": (
            "You are an expert conversation analyst. Evaluate the following conversation on three dimensions: "
            "goal_alignment (how well the conversation addresses the goal), coherence (the logical flow of the conversation), "
            "and engagement (how interesting and lively the conversation is). "
            "Output your evaluation as a JSON object with keys 'goal_alignment', 'coherence', and 'engagement', "
            "each a number between 0 and 100."
        )
    }
    
    transcript = "\n".join([f"{msg['speaker']}: {msg['text']}" for msg in conversation])
    user_instruction = {
        "role": "user",
        "content": (
            f"Goal: '{goal}'\n\n"
            "Conversation:\n" + transcript + "\n\n"
            "Please provide only the JSON output."
        )
    }
    
    messages = [system_message, user_instruction]
    
    try:
        evaluation = await async_call_llm(messages, max_tokens=100)
        eval_dict = json.loads(evaluation)
        
        # Ensure all required dimensions are present
        for key in ["goal_alignment", "coherence", "engagement"]:
            if key not in eval_dict:
                raise ValueError(f"Missing evaluation dimension: {key}")
            
            # Ensure scores are within valid range
            score = float(eval_dict[key])
            if not 0 <= score <= 100:
                raise ValueError(f"Invalid score for {key}: {score}")
            eval_dict[key] = score
        
        # Calculate aggregated score
        aggregated_score = (
            settings.GOAL_ALIGNMENT_WEIGHT * eval_dict["goal_alignment"] +
            settings.COHERENCE_WEIGHT * eval_dict["coherence"] +
            settings.ENGAGEMENT_WEIGHT * eval_dict["engagement"]
        )
        
        return eval_dict, aggregated_score
        
    except json.JSONDecodeError as e:
        raise LLMError(f"Failed to parse evaluation response: {str(e)}")
    except ValueError as e:
        raise LLMError(f"Invalid evaluation response: {str(e)}")
    except Exception as e:
        raise LLMError(f"Evaluation failed: {str(e)}")

async def evaluate_conversation(
    conversation: List[Dict[str, str]],
    goal: str
) -> Tuple[Dict[str, float], float]:
    """
    Public interface for conversation evaluation.
    Returns evaluation details and aggregated score.
    """
    return await async_evaluate_conversation_multi(conversation, goal) 