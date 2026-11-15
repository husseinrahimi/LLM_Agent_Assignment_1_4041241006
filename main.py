import argparse
import json
import sys
from agent import ReasoningAgent


def process_single_item(agent: ReasoningAgent, item: dict) -> dict:
    problem_id = item.get("id", "unknown_id")
    problem_text = item.get("problem", "")

    if not problem_text or not str(problem_text).strip():
        return {"id": problem_id, "answer": ""}

    try:
        final_answer = agent.solve(str(problem_text))
    except Exception:
        final_answer = ""

    return {
        "id": problem_id,
        "answer": str(final_answer).strip(),
    }


def main():
    parser = argparse.ArgumentParser(description="Local LLM Reasoning Agent")
    parser.add_argument("--input", required=True, help="Path to input.json")
    parser.add_argument("--output", required=True, help="Path to output.json")
    parser.add_argument("--model", default="qwen3:8b", help="Ollama model name")
    args = parser.parse_args()

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            input_data = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)

    agent = ReasoningAgent(model_name=args.model)

    # پشتیبانی همزمان از ورودی تک‌مسئله‌ای (مطابق PDF) و چندمسئله‌ای (لیست)
    if isinstance(input_data, list):
        output_data = [process_single_item(agent, item) for item in input_data]
    elif isinstance(input_data, dict):
        output_data = process_single_item(agent, input_data)
    else:
        output_data = {"id": "error", "answer": "Invalid input format"}

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
