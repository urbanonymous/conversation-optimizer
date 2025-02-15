import asyncio
import openai
import hashlib
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..core.config import settings

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass

class TokenLimitError(LLMError):
    """Raised when the token limit is exceeded."""
    pass

class RateLimitError(LLMError):
    """Raised when rate limit is hit."""
    pass

class APIError(LLMError):
    """Raised when the API call fails."""
    pass

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

class RateLimiter:
    def __init__(self, max_requests: int = 50, time_window: float = 60.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: List[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.time()
            self.requests = [t for t in self.requests if now - t <= self.time_window]
            
            if len(self.requests) >= self.max_requests:
                sleep_time = self.requests[0] + self.time_window - now
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                self.requests = self.requests[1:]
            
            self.requests.append(now)

# Initialize global instances
llm_cache = LLMCache()
rate_limiter = RateLimiter()

def hash_messages(messages: List[Dict[str, str]]) -> str:
    """Create a hash key from the messages list."""
    messages_str = json.dumps(messages, sort_keys=True)
    return hashlib.sha256(messages_str.encode("utf-8")).hexdigest()

async def async_call_llm(
    messages: List[Dict[str, str]],
    model: str = settings.LLM_MODEL,
    temperature: float = settings.TEMPERATURE,
    max_tokens: int = settings.MAX_TOKENS,
    retries: int = settings.RETRIES,
    delay: float = settings.INITIAL_DELAY
) -> str:
    """
    Asynchronously calls the LLM with retries, caching, and rate limiting.
    """
    key = hash_messages(messages)
    cached_response = llm_cache.get(key)
    if cached_response:
        return cached_response

    for attempt in range(retries):
        try:
            await rate_limiter.acquire()
            
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
            await asyncio.sleep(wait_time)
            
        except openai.error.InvalidRequestError as e:
            if "maximum context length" in str(e):
                raise TokenLimitError("Token limit exceeded") from e
            raise APIError(f"Invalid request: {str(e)}") from e
            
        except Exception as e:
            if attempt == retries - 1:
                raise APIError(f"API call failed after {retries} attempts: {str(e)}") from e
            wait_time = delay * (2 ** attempt)
            await asyncio.sleep(wait_time)

async def async_generate_possible_move(conversation: List[Dict[str, str]], speaker: str) -> str:
    """Generate a conversational move for the given speaker."""
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
        return move
    except TokenLimitError:
        # Try again with a more concise prompt
        user_instruction["content"] = (
            "Based on this conversation, give a very short reply (max 50 words):\n" +
            "\n".join([f"{msg['speaker']}: {msg['text']}" for msg in conversation[-3:]])
        )
        return await async_call_llm([system_message, user_instruction])
    except (RateLimitError, APIError) as e:
        raise LLMError(f"Error generating move: {str(e)}") 