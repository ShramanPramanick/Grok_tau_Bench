# Evaluating Grok on τ-bench

## Setup

1. Clone this repository:

```bash
git clone https://github.com/ShramanPramanick/Grok_tau_Bench.git && cd ./Grok_tau_Bench
```

2. Create a conda environment and install from source (which also installs required packages):

```bash
conda create --name tau_bench python=3.11
conda activate tau_bench
pip install -e .
```

3. Set up your XAI API key as environment variables.

```bash
XAI_API_KEY=...
```

## Run

Run a tool-calling agent on the τ-retail environment:

```bash
python run.py --agent-strategy tool-calling --env airline --model grok-4-fast-non-reasoning --model-provider xai --user-model grok-4-fast-non-reasoning --user-model-provider xai --user-strategy llm --max-concurrency 5 --num-trials 1
```

#### How to run different evaluation settings?
1. Change `--model` to evaluate different Grok variants, and vary `--num-trials` to ablate the value of `k` in the `Pass^k` metric.
2. Change `--agent-strategy` to `react`, `act` or `few-shot` for correcponding experiments. Use `--few-shot-displays-path few_shot_data/MockAirlineDomainEnv-few_shot.jsonl` as an additional argument for few-shot evaluation strategy.


