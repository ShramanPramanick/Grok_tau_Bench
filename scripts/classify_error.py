#!/usr/bin/env python3
import os
import json
import argparse
import time
import requests

XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_CHAT_URL = "https://api.x.ai/v1/chat/completions"
MODEL_NAME = "grok-4-fast-non-reasoning"  # or "grok-4-1-fast-reasoning" if enabled on your account


ERROR_CATEGORIES = [
    "partial_plan_execution",
    "policy_constraint_violation",
    "intent_misinterpretation",
    "overgeneralized_action_pattern",
]


def build_classification_prompt(task_obj: dict) -> str:
    """
    Build a compact but informative prompt for Grok given a single τ-Bench task entry.
    """
    task_id = task_obj.get("task_id")
    reward = task_obj.get("reward")
    info = task_obj.get("info", {})
    task = info.get("task", {})
    traj = task_obj.get("traj", [])
    reward_info = info.get("reward_info", {})

    # Keep the JSON we send trimmed to useful pieces
    compact = {
        "task_id": task_id,
        "reward": reward,
        "instruction": task.get("instruction"),
        "actions_ground_truth": reward_info.get("actions"),
        "outputs_ground_truth": reward_info.get("info", {}).get("outputs"),
        "model_actions": task.get("actions"),
        "model_outputs": task.get("outputs"),
        "trajectory_snippet": traj[:8],  # limit for length
    }

    prompt = f"""
You are auditing an agent's failure on a tool-using benchmark (τ-Bench).

You must classify the **primary** reason for failure into exactly ONE of these four categories:

1) partial_plan_execution
   - The model does some steps correctly but fails to fully complete the required workflow
   - Examples: returns only some requested items; changes only one leg of a round-trip; doesn't compute or apply refund correctly; stops early.

2) policy_constraint_violation
   - The model understands the task but violates hard domain rules or procedures
   - Examples: uses two certificates when only one is allowed; ignores an explicit business-class request; performs multiple irreversible tool calls when only one is allowed.

3) intent_misinterpretation
   - The model misreads or drops parts of the user’s stated intent or conditional preferences
   - Examples: ignores a fallback condition; treats "book later" as "book now"; misses that both directions must be updated; forgets requested bags.

4) overgeneralized_action_pattern
   - The model applies a memorized workflow pattern that doesn't fit this specific instruction
   - Examples: automatically cancels and rebooks when the user only asked to cancel; modifies reservations or orders just because they exist, not because the user asked.

Given the task object below (JSON), identify which single category best explains why this task failed (reward < 1).
Then briefly justify your choice based on the mismatch between intended behavior and the model's actions.

TASK_JSON:
{json.dumps(compact, indent=2)}

Return your answer as a JSON object with exactly these keys:
- "category": one of {ERROR_CATEGORIES}
- "rationale": 2–4 sentences explaining why.
"""
    return prompt


def call_grok(prompt: str) -> dict:
    """
    Call xAI Grok chat completions API with the given prompt.
    Returns the parsed JSON content from the model (or raises).
    """
    if not XAI_API_KEY:
        raise RuntimeError("XAI_API_KEY env var not set.")

    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": MODEL_NAME,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict evaluation assistant. "
                    "Always output valid JSON with keys 'category' and 'rationale'. "
                    f"Allowed categories: {ERROR_CATEGORIES}."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        # keep temperature low for consistency
        "temperature": 0.1,
    }

    resp = requests.post(XAI_CHAT_URL, headers=headers, json=data, timeout=120)
    resp.raise_for_status()
    result = resp.json()

    # OpenAI/xAI-style response structure
    content = result["choices"][0]["message"]["content"]
    # Try to parse the content as JSON (model is instructed to output JSON)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: wrap raw content
        parsed = {"category": "unknown", "rationale": content}

    return parsed


def classify_file(input_path: str, output_path: str, sleep_sec: float = 0.5):
    with open(input_path, "r") as f:
        data = json.load(f)

    with open(output_path, "w") as out_f:
        for task_obj in data:
            task_id = task_obj.get("task_id")
            reward = task_obj.get("reward", 0.0)

            # Only classify failures
            if reward >= 1.0:
                continue

            prompt = build_classification_prompt(task_obj)
            try:
                classification = call_grok(prompt)
            except Exception as e:
                classification = {
                    "category": "api_error",
                    "rationale": f"API call failed: {e}",
                }

            record = {
                "task_id": task_id,
                "reward": reward,
                "category": classification.get("category"),
                "rationale": classification.get("rationale"),
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            print(f"[task {task_id}] -> {record['category']}")
            time.sleep(sleep_sec)  # gentle rate limiting


def main():
    parser = argparse.ArgumentParser(
        description="Classify τ-Bench Grok failures into four error categories using Grok itself."
    )
    parser.add_argument(
        "input_file",
        help="Path to τ-Bench result JSON file (e.g., retail or airline results).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="error_classification.jsonl",
        help="Output JSONL file with classifications (default: error_classification.jsonl).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep seconds between API calls (for rate limiting).",
    )

    args = parser.parse_args()
    classify_file(args.input_file, args.output, sleep_sec=args.sleep)


if __name__ == "__main__":
    main()
