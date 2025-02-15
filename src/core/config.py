from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Conversation Optimizer"
    
    # OpenAI Settings
    OPENAI_API_KEY: str
    LLM_MODEL: str = "gpt-3.5-turbo"
    
    # MCTS Settings
    SIMULATION_DEPTH: int = 4
    SIMULATION_BREADTH: int = 5
    TOP_N: int = 3
    EARLY_STOP_THRESHOLD: int = 30
    MCTS_ITERATIONS: int = 20
    EXPLORATION_CONSTANT: float = 1.414
    
    # LLM Settings
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 150
    RETRIES: int = 3
    INITIAL_DELAY: float = 1.0
    
    # Scoring Weights
    GOAL_ALIGNMENT_WEIGHT: float = 0.5
    COHERENCE_WEIGHT: float = 0.3
    ENGAGEMENT_WEIGHT: float = 0.2

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings() 