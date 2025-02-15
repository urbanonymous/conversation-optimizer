# Conversation Optimizer

An AI-powered conversation optimization service that uses Monte Carlo Tree Search (MCTS) and Large Language Models to suggest optimal conversational moves and evaluate conversation quality.

## Features

- Conversation optimization using MCTS
- Multi-dimensional conversation evaluation
- FastAPI-based REST API
- Caching and rate limiting for LLM calls
- Async support for improved performance

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/conversation-optimizer.git
cd conversation-optimizer
```

2. Install uv (if not already installed):
```bash
pip install uv
```

3. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

4. Create a `.env` file in the project root with your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
LLM_MODEL=gpt-3.5-turbo  # Optional: Change the model
```

## Usage

1. Start the API server:
```bash
uvicorn src.api.main:app --reload
```

2. The API will be available at `http://localhost:8000`

3. Access the API documentation at `http://localhost:8000/docs`

## API Endpoints

### POST /api/v1/optimize
Optimize a conversation by suggesting the best next moves.

Request:
```json
{
    "conversation": [
        {"speaker": "you", "text": "Hi, I wanted to discuss our project strategy."},
        {"speaker": "opponent", "text": "I'm concerned about the risks involved."}
    ],
    "goal": "Discuss project strategy while addressing risks"
}
```

### POST /api/v1/evaluate
Evaluate a conversation based on multiple dimensions.

Request:
```json
{
    "conversation": [
        {"speaker": "you", "text": "Hi, I wanted to discuss our project strategy."},
        {"speaker": "opponent", "text": "I'm concerned about the risks involved."}
    ],
    "goal": "Discuss project strategy while addressing risks"
}
```

## Development

1. Install development dependencies:
```bash
uv pip install -e ".[dev]"
```

2. Run tests:
```bash
pytest
```

3. Format code:
```bash
black src/
isort src/
```

4. Run linting:
```bash
ruff check src/
mypy src/
```

## Configuration

The application can be configured through environment variables or a `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `LLM_MODEL`: The LLM model to use (default: gpt-3.5-turbo)
- `SIMULATION_DEPTH`: MCTS simulation depth (default: 4)
- `SIMULATION_BREADTH`: MCTS simulation breadth (default: 5)
- `MCTS_ITERATIONS`: Number of MCTS iterations (default: 20)

See `src/api/core/config.py` for all available settings.

## License

MIT License
