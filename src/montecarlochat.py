import asyncio
import openai
import json
import hashlib
import logging
import random
import re
import copy
import time
import math
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class MCTSConfig:
    simulation_breadth: int = 5
    simulation_depth: int = 4
    top_n: int = 3
    early_stop_threshold: int = 30
    mcts_iterations: int = 20
    exploration_constant: float = 1.414

@dataclass
class LLMConfig:
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 150
    retries: int = 3
    initial_delay: float = 1.0

@dataclass
class ScoringWeights:
    goal_alignment: float = 0.5
    coherence: float = 0.3
    engagement: float = 0.2

class Config:
    def __init__(self):
        self.mcts = MCTSConfig()
        self.llm = LLMConfig()
        self.weights = ScoringWeights()
        
        # Override with environment variables if present
        if os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OPENAI_API_KEY environment variable is required")
            
        if os.getenv("LLM_MODEL"):
            self.llm.model = os.getenv("LLM_MODEL")
        if os.getenv("SIMULATION_DEPTH"):
            self.mcts.simulation_depth = int(os.getenv("SIMULATION_DEPTH"))
        if os.getenv("SIMULATION_BREADTH"):
            self.mcts.simulation_breadth = int(os.getenv("SIMULATION_BREADTH"))

# Initialize global config
config = Config()

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize LLM cache with TTL
class LLMCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl_seconds:
                return entry["content"]
            else:
                del self.cache[key]
        return None

    def set(self, key: str, content: str) -> None:
        self.cache[key] = {
            "content": content,
            "timestamp": time.time()
        }

    def cleanup(self) -> None:
        current_time = time.time()
        expired_keys = [
            k for k, v in self.cache.items()
            if current_time - v["timestamp"] > self.ttl_seconds
        ]
        for k in expired_keys:
            del self.cache[k]

llm_cache = LLMCache()

def hash_messages(messages):
    """Create a hash key from the messages list."""
    messages_str = json.dumps(messages, sort_keys=True)
    return hashlib.sha256(messages_str.encode("utf-8")).hexdigest()

class RateLimiter:
    def __init__(self, max_requests: int = 50, time_window: float = 60.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: List[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request can be made without exceeding the rate limit."""
        async with self._lock:
            now = time.time()
            # Remove old requests outside the time window
            self.requests = [t for t in self.requests if now - t <= self.time_window]
            
            if len(self.requests) >= self.max_requests:
                # Wait until the oldest request expires
                sleep_time = self.requests[0] + self.time_window - now
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                self.requests = self.requests[1:]
            
            self.requests.append(now)

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass

class TokenLimitError(LLMError):
    """Raised when the token limit is exceeded."""
    pass

class APIError(LLMError):
    """Raised when the API call fails."""
    pass

class RateLimitError(LLMError):
    """Raised when rate limit is hit."""
    pass

# Initialize rate limiter
rate_limiter = RateLimiter()

async def async_call_llm(
    messages: List[Dict[str, str]],
    model: str = config.llm.model,
    temperature: float = config.llm.temperature,
    max_tokens: int = config.llm.max_tokens,
    retries: int = config.llm.retries,
    delay: float = config.llm.initial_delay
) -> str:
    """
    Asynchronously calls the LLM with retries, caching, and rate limiting.
    
    Args:
        messages: List of message dictionaries to send to the LLM
        model: The model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens in the response
        retries: Number of retries on failure
        delay: Initial delay between retries (doubles with each retry)
    
    Returns:
        The LLM's response text
    
    Raises:
        TokenLimitError: If the token limit is exceeded
        RateLimitError: If the rate limit is hit
        APIError: If the API call fails after all retries
    """
    key = hash_messages(messages)
    cached_response = llm_cache.get(key)
    if cached_response:
        logger.debug("Cache hit for prompt.")
        return cached_response

    for attempt in range(retries):
        try:
            # Wait for rate limit
            await rate_limiter.acquire()
            
            # Make API call
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            content = response.choices[0].message['content'].strip()
            llm_cache.set(key, content)
            return content
            
        except openai.error.RateLimitError as e:
            if attempt == retries - 1:
                raise RateLimitError("Rate limit exceeded") from e
            wait_time = delay * (2 ** attempt)
            logger.warning(f"Rate limit hit (attempt {attempt + 1}/{retries}). Waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
            
        except openai.error.InvalidRequestError as e:
            if "maximum context length" in str(e):
                raise TokenLimitError("Token limit exceeded") from e
            raise APIError(f"Invalid request: {str(e)}") from e
            
        except Exception as e:
            if attempt == retries - 1:
                raise APIError(f"API call failed after {retries} attempts: {str(e)}") from e
            wait_time = delay * (2 ** attempt)
            logger.warning(f"API call failed (attempt {attempt + 1}/{retries}): {str(e)}. Retrying in {wait_time:.1f}s")
            await asyncio.sleep(wait_time)

async def async_generate_possible_move(conversation: List[Dict[str, str]], speaker: str) -> str:
    """
    Uses an LLM to generate a conversational move for the given speaker.
    Handles various error cases gracefully.
    """
    system_message = {
        "role": "system", 
        "content": (
            f"You are acting as a conversation agent with a distinct personality for '{speaker}'. "
            "Answer naturally and succinctly. Include your chain-of-thought only if it helps, but output only the final reply."
        )
    }
    
    user_instruction = {
        "role": "user",
        "content": (
            "Given the conversation history below, generate a single, concise reply that is natural, context-aware, "
            "and reflects your personality. Do not include any explanations—just provide the message text.\n\n"
            "Conversation History:\n" +
            "\n".join([f"{msg['speaker']}: {msg['text']}" for msg in conversation])
        )
    }
    
    messages = [system_message, user_instruction]
    
    try:
        move = await async_call_llm(messages)
        logger.debug(f"Generated move for '{speaker}': {move}")
        return move
    except TokenLimitError:
        logger.warning("Token limit exceeded, generating shorter response")
        # Try again with a more concise prompt
        user_instruction["content"] = (
            "Based on this conversation, give a very short reply (max 50 words):\n" +
            "\n".join([f"{msg['speaker']}: {msg['text']}" for msg in conversation[-3:]])
        )
        return await async_call_llm([system_message, user_instruction])
    except (RateLimitError, APIError) as e:
        logger.error(f"Error generating move: {str(e)}")
        return f"[Error: {str(e)}]"

async def async_evaluate_conversation_multi(conversation, goal):
    """
    Uses an LLM to evaluate the conversation on multiple dimensions:
      - goal_alignment: How well the conversation addresses the goal.
      - coherence: How coherent the conversation is.
      - engagement: How engaging the conversation is.
    
    The LLM is instructed to output a JSON object with keys "goal_alignment", "coherence", and "engagement",
    each between 0 and 100.
    Returns the parsed dictionary.
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
        for key in ["goal_alignment", "coherence", "engagement"]:
            if key not in eval_dict:
                eval_dict[key] = random.uniform(0, 100)
        logger.debug(f"Multi-dimensional evaluation: {eval_dict}")
        return eval_dict
    except Exception as e:
        logger.error(f"Error during multi-dimensional evaluation: {e}")
        return {
            "goal_alignment": random.uniform(0, 100),
            "coherence": random.uniform(0, 100),
            "engagement": random.uniform(0, 100)
        }

def aggregate_score(score_details):
    """
    Compute an aggregated score based on the weighted sum of multi-dimensional scores.
    """
    weights = config.weights
    total = (weights.goal_alignment * score_details.get("goal_alignment", 0) +
             weights.coherence * score_details.get("coherence", 0) +
             weights.engagement * score_details.get("engagement", 0))
    return total

async def async_simulate_conversation_multi(conversation, goal, depth):
    """
    Starting from the given conversation, simulate further moves for up to `depth` steps.
    After each move, evaluate the conversation using multi-dimensional scoring.
    If the aggregated score falls below EARLY_STOP_THRESHOLD, stop early.
    
    Returns a tuple: (simulated_conversation, evaluation_details, aggregated_score)
    """
    simulated_convo = copy.deepcopy(conversation)
    current_speaker = "you"  # Assume next move is from "you".
    
    for i in range(depth):
        move_text = await async_generate_possible_move(simulated_convo, current_speaker)
        simulated_convo.append({"speaker": current_speaker, "text": move_text})
        current_speaker = "opponent" if current_speaker == "you" else "you"
        await asyncio.sleep(0.5)
        
        eval_details = await async_evaluate_conversation_multi(simulated_convo, goal)
        agg_score = aggregate_score(eval_details)
        if agg_score < config.early_stop_threshold:
            logger.info(f"Early stopping simulation: aggregated score {agg_score:.2f} below threshold {config.early_stop_threshold}.")
            return simulated_convo, eval_details, agg_score

    final_eval = await async_evaluate_conversation_multi(simulated_convo, goal)
    final_agg = aggregate_score(final_eval)
    return simulated_convo, final_eval, final_agg

# ------------------ MCTS Components ------------------ #

class MCTSError(Exception):
    """Base exception for MCTS-related errors."""
    pass

class SimulationError(MCTSError):
    """Raised when a simulation fails."""
    pass

class EvaluationError(MCTSError):
    """Raised when conversation evaluation fails."""
    pass

@dataclass
class ConversationMove:
    speaker: str
    text: str
    score: Optional[float] = None

@dataclass
class SimulationResult:
    conversation: List[Dict[str, str]]
    evaluation: Dict[str, float]
    aggregated_score: float

class MCTSNode:
    def __init__(self, conversation: List[Dict[str, str]], parent: Optional['MCTSNode'], move: Optional[str]):
        self.conversation = conversation
        self.parent = parent
        self.move = move
        self.children: List[MCTSNode] = []
        self.visits: int = 0
        self.total_score: float = 0.0
        self.depth: int = 0 if parent is None else parent.depth + 1
        self.ucb_score: float = float('inf')  # Upper Confidence Bound score

    def add_child(self, move: str, conversation: List[Dict[str, str]]) -> 'MCTSNode':
        child = MCTSNode(conversation, self, move)
        self.children.append(child)
        return child

    def update(self, simulation_score: float) -> None:
        self.visits += 1
        self.total_score += simulation_score
        self.update_ucb_score()

    def update_ucb_score(self) -> None:
        if self.visits == 0:
            self.ucb_score = float('inf')
        elif self.parent is None:
            self.ucb_score = float('inf')
        else:
            exploitation = self.total_score / self.visits
            exploration = config.mcts.exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
            self.ucb_score = exploitation + exploration

async def select_node(node: MCTSNode) -> MCTSNode:
    """
    Traverse the tree from this node using the UCT formula until a leaf node is reached.
    Returns the selected leaf node.
    """
    current = node
    while current.children:
        current.update_ucb_score()
        current = max(current.children, key=lambda n: n.ucb_score)
    return current

async def expand_node(node: MCTSNode, goal: str) -> MCTSNode:
    """
    Expand a leaf node by generating a new child node.
    Raises SimulationError if expansion fails.
    """
    try:
        move = await async_generate_possible_move(node.conversation, "you")
        new_convo = copy.deepcopy(node.conversation)
        new_convo.append({"speaker": "you", "text": move})
        return node.add_child(move, new_convo)
    except Exception as e:
        raise SimulationError(f"Failed to expand node: {str(e)}") from e

async def simulate_and_evaluate(node: MCTSNode, goal: str, remaining_depth: int) -> SimulationResult:
    """
    Run a simulation from the given node and evaluate the result.
    Raises SimulationError if simulation fails.
    """
    try:
        simulated_convo, eval_details, agg_score = await async_simulate_conversation_multi(
            node.conversation,
            goal,
            remaining_depth
        )
        return SimulationResult(simulated_convo, eval_details, agg_score)
    except Exception as e:
        raise SimulationError(f"Simulation failed: {str(e)}") from e

async def backpropagate(node: MCTSNode, score: float) -> None:
    """
    Backpropagate the simulation result up the tree.
    """
    current = node
    while current is not None:
        current.update(score)
        current = current.parent

async def run_mcts(conversation: List[Dict[str, str]], goal: str) -> List[Dict[str, Any]]:
    """
    Run the complete MCTS algorithm and return the best moves.
    """
    root = MCTSNode(conversation, None, None)
    
    try:
        for _ in range(config.mcts.mcts_iterations):
            # Selection
            leaf = await select_node(root)
            
            # Expansion
            if leaf.depth < config.mcts.simulation_depth:
                leaf = await expand_node(leaf, goal)
            
            # Simulation
            remaining_depth = config.mcts.simulation_depth - leaf.depth
            result = await simulate_and_evaluate(leaf, goal, remaining_depth)
            
            # Backpropagation
            await backpropagate(leaf, result.aggregated_score)
            
            # Cleanup cache periodically
            if _ % 5 == 0:
                llm_cache.cleanup()
                
    except MCTSError as e:
        logger.error(f"MCTS error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in MCTS: {str(e)}")
        raise MCTSError(f"MCTS failed: {str(e)}") from e
    
    # Sort children by average score
    moves = []
    for child in root.children:
        avg_score = child.total_score / child.visits if child.visits > 0 else 0
        moves.append({
            "move": child.move,
            "avg_score": avg_score,
            "visits": child.visits
        })
    
    moves.sort(key=lambda x: x["avg_score"], reverse=True)
    return moves[:config.mcts.top_n]

# ------------------ Main Usage ------------------ #

async def main():
    conversation = [
        {"speaker": "you", "text": "Hi, I wanted to discuss our upcoming project strategy."},
        {"speaker": "opponent", "text": "I'm concerned about the risks involved with our current approach."}
    ]
    
    goal = "Discuss the merits and details of our project strategy while addressing risks."
    
    logger.info("Running hybrid beam search with MCTS for conversation planning...")
    candidates = await run_mcts(
        conversation,
        goal
    )
    
    if candidates:
        print("\nTop candidate moves:")
        for idx, candidate in enumerate(candidates, start=1):
            print(f"\nOption {idx}:")
            print(f"Speaker: you")
            print(f"Text: {candidate['move']}")
            print(f"Average aggregated score: {candidate['avg_score']:.2f} (based on {candidate['visits']} simulations)")
    else:
        print("No moves generated.")

if __name__ == "__main__":
    asyncio.run(main())
