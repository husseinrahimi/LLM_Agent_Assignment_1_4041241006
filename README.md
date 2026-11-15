# LLM_Agent_Assignment_1_4041241006

A multi-step local LLM reasoning agent built with Python and Ollama for solving mathematical, logical, scheduling, and constraint-satisfaction problems.

## Requirements

- Python 3.9 or newer
- Ollama installed and running locally
- The evaluation model downloaded in Ollama (`qwen3:8b` / `qwen3.8:27b`)

## Installation

Install the Python dependency:

    pip install -r requirements.txt

Download the model specified in the assignment:

    ollama pull qwen3:8b

Make sure Ollama is running before executing the program.

## Input format

Create an `input.json` file (supports both a single JSON object and a list of JSON objects):

    {
        "id": "test_001",
        "problem": "What is 17 multiplied by 23?"
    }

## Run

    python main.py --input input.json --output output.json

## Output format

The program writes the result to the specified output file:

    {
        "id": "test_001",
        "answer": "391"
    }

The `answer` field contains only the final normalized answer without explanations, thinking tags, or Markdown formatting.

## Agent architecture

The reasoning pipeline is implemented in `agent.py` (`ReasoningAgent`) and consists of three sequential stages:

1. **Planner & Solver:** Analyzes the problem statement, identifies variables, domains, and explicit/implicit constraints, and derives a step-by-step initial solution.
2. **Verifier & Critic:** Checks the proposed solution against every constraint in the original problem (verifying truth tables, relative orderings, grid boundaries, and arithmetic). If any contradiction is found, it re-solves the problem from scratch.
3. **Strict JSON Formatter & Normalizer:** Extracts only the verified final answer in structured JSON mode, strips `<think>` tags and Markdown artifacts, and converts Persian/Arabic digits into standard English digits (`0-9`).

All model inference is performed locally through Ollama (`temperature=0.0`) with automatic model resolution and a 600-second timeout per request.

## Repository files

- `main.py`: Standard CLI entry point (`--input` and `--output`).
- `agent.py`: Multi-step `ReasoningAgent` implementation.
- `requirements.txt`: Required Python dependencies.
- `README.md`: Project documentation and usage guide.
