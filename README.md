# Intent-as-a-Tool

This repository contains the code and instructions for reproducing the experiments in our paper, *Intent-as-a-Tool Makes it Easy to Track Agentic Misalignment*. We study agentic misalignment during LLM reasoning and use intent-tool call probabilities to track behavior-specific intent.

## Overview

The repository implements a two-stage decoding pipeline, action classifiers, and two defense methods:

- **Stage 1 — Full inference.** Run a complete inference and capture the full reasoning trace (token IDs) together with the final tool call.
- **Stage 2 — Truncated decoding.** Replay the saved reasoning trace at token, sentence, or paragraph boundaries and score the first-token probability of each candidate tool name.
- **Classification.** Label the collected actions with two classifiers:
  - `behavioral_judge.py` — LLM-as-judge harmfulness over the rendered transcript.
  - `decision_taxonomy.py` — hierarchical A/B/C/D/E labels of the decision process.
- **Defense.** Run online intervention during decoding or system-prompt prompting.

## Scenarios and Tool Modes

Three agentic-misalignment scenarios — `blackmail`, `leaking`, and `murder` — are built from the prompts in `data/agentic_misalignment/` (one folder per condition). Tool definitions live in the `tools/` package, and each scenario can be run in one of three tool modes:

| Mode | Tools exposed |
|---|---|
| `neutral` | Neutral action tools only — `blackmail`: `email`; `leaking`: `email`, `forward`; `murder`: `email`, `cancel_alert` |
| `signal` | Neutral tools **plus** a per-scenario intent-declaration tool (`declare_blackmail_intent`, `declare_leaking_intent`, `declare_deny_rescue`) |
| `all` | Every scenario's neutral and signal tools at once |

Signal tools do **not** perform the harmful action themselves; they ask the model to declare that it has formed the harmful intent and is about to act on it. The harmful action is still carried out through a neutral tool. (`all` and `clean` are meta-scenarios that expose every scenario's tools and reuse the same prompts.)

## Requirements

- Python 3.10–3.14
- A CUDA-capable GPU to serve the behavior model with [vLLM](https://github.com/vllm-project/vllm)
- An OpenAI-compatible API key for the LLM-as-judge classifiers

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your credentials
```

`.env` holds the OpenAI-compatible credentials for the classifiers and the vLLM endpoint for the behavior model. It is gitignored — never commit it.

## Usage

### 1. Start a vLLM server for the behavior model

```bash
bash vllm_setups/run_vllm_qwen3_8b_tool.sh   # serves "qwen3-8b" on port 8000
```

The launchers use vLLM's default Hugging Face cache and port `8000`. They do not force offline mode, a mirror, or a GPU selection. Deployment-specific settings can be supplied without editing the scripts:

```bash
HF_HOME=/path/to/cache CUDA_VISIBLE_DEVICES=0,1 \
  bash vllm_setups/run_vllm_qwen3_8b_tool.sh \
  --tensor-parallel-size 2 --port 8104
```

When changing the port, set `VLLM_SERVER` in `.env` (or in the command environment) to the same endpoint. The default in `.env.example` is `http://localhost:8000/v1`.

### 2. Stage 1 — collect complete reasoning traces

```bash
bash scripts/collect_action_decode.sh
```

### 3. Stage 2 — truncated decoding

```bash
STAGE=2 bash scripts/collect_action_decode.sh
```

### 4. Classification

```bash
# LLM-as-judge harmfulness
bash scripts/behavioral_judge.sh

# Decision-process taxonomy
bash scripts/decision_taxonomy.sh
TOOL_MODE=neutral bash scripts/decision_taxonomy.sh
```

### 5. Defense (optional)

```bash
bash scripts/defense.sh
```

## Configuration

All driver scripts are configured through environment variables:

| Variable | Default | Description |
|---|---|---|
| `STAGE` | `1` | Pipeline stage: `1` (full trace) or `2` (truncated) |
| `MODEL` / `BEHAVIOR_MODEL` | `qwen3-8b` | Behavior model under study |
| `TOOL_MODE` | `signal` | `neutral` / `signal` / `all` (`defense.sh` supports `neutral` / `signal`) |
| `TRUNCATION_GRANULARITY` | `token` | `token` / `sentence` / `paragraph` |
| `SCENARIOS` | `blackmail leaking murder` | Scenarios to run |
| `NUM_SAMPLES` | `1` | Reasoning samples per prompt |
| `MAX_WORKERS` | `32` | Client-side parallelism |
| `BATCH_SIZE` | `32` | Prompts per batched vLLM call (stage-2 token only) |
| `DEFENSE_METHOD` | `prompting` | `intervention` / `prompting` |
| `MODE` | `stage1` | `behavioral_judge.sh`: `stage1` / `stage2_token` / `stage2_sentence` / `intervention` / `prompting` |
| `CLASSIFIER_MODEL` | `gpt-4.1` | OpenAI-compatible judge model |
| `VLLM_SERVER` | `http://localhost:8000/v1` | vLLM OpenAI-compatible endpoint |

## Project Structure

```
pipeline/                       # Core two-stage decoding pipeline
  collect_action_decode.py      #   Entry point: stage 1 / 2
  truncation_engine.py          #   Truncation + forced tool-call engine
  predict_model.py              #   Prefix-continuation first-token tool-name probabilities
  reasoning_splitter.py         #   Token / sentence / paragraph splitting
classifiers/                    # Post-hoc labeling of collected actions
  behavioral_judge.py           #   LLM-as-judge harmfulness classifier
  decision_taxonomy.py          #   Hierarchical decision-process taxonomy
defense/                        # Online intervention + prompting defenses
  collect_action_decode_defense.py
  defense_engine.py
tools/                          # Scenario tool definitions
  neutral.py                    #   Neutral action tools per scenario
  signal.py                     #   Intent-declaration (signal) tools per scenario
scripts/                        # Shell driver scripts
  collect_action_decode.sh      #   Stage 1 / 2 driver
  behavioral_judge.sh           #   LLM-as-judge driver
  decision_taxonomy.sh          #   Taxonomy driver (auto-selects variant)
  defense.sh                    #   Defense driver
vllm_setups/                    # Per-model vLLM launch scripts
data/
  agentic_misalignment/         # Scenario prompts and upstream dataset license
```

## Outputs

```
results/agentic_misalignment_decode_{tool_mode}/{model}/
├── stage1/                                    # Complete traces (one .jsonl per scenario)
├── stage2_{granularity}/                      # Truncated tool calls (e.g. stage2_token)
├── {mode}_final_judge/                        # behavioral_judge labels (stage1 / stage2_*)
├── decision_process_taxonomy_final_{model}/   # Taxonomy labels
└── defense/
    ├── {intervention,prompting}/              # Defense runs
    └── {mode}_final_judge/                    # Judge labels for defense runs
```
