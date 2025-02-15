import asyncio
import copy
import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..core.config import settings
from .llm import async_call_llm, LLMError
from .evaluation import async_evaluate_conversation_multi

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
        self.ucb_score: float = float('inf')

    def add_child(self, move: str, conversation: List[Dict[str, str]]) -> 'MCTSNode':
        child = MCTSNode(conversation, self, move)
        self.children.append(child)
        return child

    def update(self, simulation_score: float) -> None:
        self.visits += 1
        self.total_score += simulation_score
        self.update_ucb_score()

    def update_ucb_score(self) -> None:
        if self.visits == 0 or self.parent is None:
            self.ucb_score = float('inf')
        else:
            exploitation = self.total_score / self.visits
            exploration = settings.EXPLORATION_CONSTANT * math.sqrt(math.log(self.parent.visits) / self.visits)
            self.ucb_score = exploitation + exploration

async def select_node(node: MCTSNode) -> MCTSNode:
    """Traverse the tree using UCT formula until reaching a leaf node."""
    current = node
    while current.children:
        current.update_ucb_score()
        current = max(current.children, key=lambda n: n.ucb_score)
    return current

async def expand_node(node: MCTSNode, goal: str) -> MCTSNode:
    """Expand a leaf node by generating a new child node."""
    try:
        move = await async_generate_possible_move(node.conversation, "you")
        new_convo = copy.deepcopy(node.conversation)
        new_convo.append({"speaker": "you", "text": move})
        return node.add_child(move, new_convo)
    except Exception as e:
        raise LLMError(f"Failed to expand node: {str(e)}") from e

async def simulate_and_evaluate(node: MCTSNode, goal: str, remaining_depth: int) -> SimulationResult:
    """Run a simulation from the given node and evaluate the result."""
    try:
        simulated_convo = copy.deepcopy(node.conversation)
        current_speaker = "opponent"  # Start with opponent since node is "you"
        
        for _ in range(remaining_depth):
            move = await async_generate_possible_move(simulated_convo, current_speaker)
            simulated_convo.append({"speaker": current_speaker, "text": move})
            current_speaker = "you" if current_speaker == "opponent" else "opponent"
            
            eval_details = await async_evaluate_conversation_multi(simulated_convo, goal)
            agg_score = aggregate_score(eval_details)
            
            if agg_score < settings.EARLY_STOP_THRESHOLD:
                return SimulationResult(simulated_convo, eval_details, agg_score)
        
        final_eval = await async_evaluate_conversation_multi(simulated_convo, goal)
        final_agg = aggregate_score(final_eval)
        return SimulationResult(simulated_convo, final_eval, final_agg)
        
    except Exception as e:
        raise LLMError(f"Simulation failed: {str(e)}") from e

async def backpropagate(node: MCTSNode, score: float) -> None:
    """Backpropagate the simulation result up the tree."""
    current = node
    while current is not None:
        current.update(score)
        current = current.parent

async def run_mcts(conversation: List[Dict[str, str]], goal: str) -> List[Dict[str, Any]]:
    """Run the complete MCTS algorithm and return the best moves."""
    root = MCTSNode(conversation, None, None)
    
    try:
        for _ in range(settings.MCTS_ITERATIONS):
            # Selection
            leaf = await select_node(root)
            
            # Expansion
            if leaf.depth < settings.SIMULATION_DEPTH:
                leaf = await expand_node(leaf, goal)
            
            # Simulation
            remaining_depth = settings.SIMULATION_DEPTH - leaf.depth
            result = await simulate_and_evaluate(leaf, goal, remaining_depth)
            
            # Backpropagation
            await backpropagate(leaf, result.aggregated_score)
            
    except Exception as e:
        raise LLMError(f"MCTS failed: {str(e)}") from e
    
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
    return moves[:settings.TOP_N]

def aggregate_score(score_details: Dict[str, float]) -> float:
    """Compute an aggregated score based on the weighted sum of multi-dimensional scores."""
    return (
        settings.GOAL_ALIGNMENT_WEIGHT * score_details.get("goal_alignment", 0) +
        settings.COHERENCE_WEIGHT * score_details.get("coherence", 0) +
        settings.ENGAGEMENT_WEIGHT * score_details.get("engagement", 0)
    ) 