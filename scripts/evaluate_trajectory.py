#!/usr/bin/env python3
import os
import json
import argparse
from openai import OpenAI

SYSTEM_PROMPT = """
You are a strict airline agent policy evaluator.

Given:
(1) the user's goal,
(2) the domain rules implicitly encoded in the actions,
(3) the executed tool actions,

evaluate the quality of the tool-use trajectory.

For each tool call, decide whether it is:
- correct (necessary and appropriate for the goal),
- unnecessary (not needed but does not break correctness),
- incorrect (violates the user intent or domain constraints).

Then provide an overall score in the range 1–5, where:
- 5 means all tool calls are correct and necessary with a near-optimal trajectory,
- lower scores penalize incorrect or unnecessary calls and overly long trajectories.

Return your answer as a short, well-structured explanation plus the final numeric score.
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Score τ-Bench airline trajectories with Grok as a judge."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to τ-Bench result JSON file for the airline domain.",
    )
    parser.add_argument(
        "--output_dir",
        default="judged",
        help="Directory to write per-episode judge outputs (default: judged).",
    )
    return parser.parse_args()


def make_client() -> OpenAI:
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the XAI_API_KEY environment variable.")
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
    )
    return client


def score_episode(client: OpenAI, ep: dict) -> str:
    # User goal / instruction
    user_goal = ep["info"]["task"]["instruction"]

    model_actions = ep["info"]["task"].get("actions", [])
    traj = ep.get("traj", [])

    content = (
        "USER GOAL:\n"
        f"{user_goal}\n\n"
        "EXECUTED TOOL ACTIONS (model):\n"
        f"{json.dumps(model_actions, indent=2)}\n\n"
        "FULL TRAJECTORY (if available):\n"
        f"{json.dumps(traj, indent=2)}"
    )

    resp = client.chat.completions.create(
        model="grok-4-fast-non-reasoning", 
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content


def main():
    args = parse_args()
    client = make_client()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load τ-Bench episodes
    with open(args.input_file, "r") as f:
        episodes = json.load(f)

    for ep in episodes:
        task_id = ep.get("task_id", "unknown")
        print(f"Scoring episode {task_id}...")
        result = score_episode(client, ep)

        out_path = os.path.join(args.output_dir, f"{task_id}.txt")
        with open(out_path, "w") as out_f:
            out_f.write(result)


if __name__ == "__main__":
    main()
