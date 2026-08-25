"""
Unified truncation engine for agentic-misalignment scenarios.
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .predict_model import ToolCallPredictor
from .reasoning_splitter import get_reasoning_prefix_from_indices

from tools import get_tools_for_scenario

MISALIGNMENT_SCENARIOS = {"blackmail", "leaking", "murder", "all", "clean"}


def get_reasoning_token_ids(token_ids: List[int], tokenizer: AutoTokenizer, model: str) -> List[int]:
    """Extract reasoning token IDs from a generated token sequence."""
    model_lower = model.lower()

    if "qwen" in model_lower:
        begin_token_id = tokenizer.convert_tokens_to_ids('<think>')
        end_token_id = tokenizer.convert_tokens_to_ids('</think>')
        start_index = token_ids.index(begin_token_id)+2
        end_index = token_ids.index(end_token_id)
    elif "gemma4" in model_lower:
        begin_token_id = tokenizer.convert_tokens_to_ids("<|channel>")
        end_token_id = tokenizer.convert_tokens_to_ids('<channel|>')
        start_index = token_ids.index(begin_token_id)+3
        end_index = token_ids.index(end_token_id)
    else:
        raise ValueError(f"Unsupported model: {model}")

    return token_ids[start_index:end_index]


def _resolve_tools(
    scenario: str,
    tool_mode: str = "neutral",
    signal_override: Optional[Dict[str, Any]] = None,
) -> list:
    """Return the tool list for a scenario."""
    if scenario not in MISALIGNMENT_SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Supported: {sorted(MISALIGNMENT_SCENARIOS)}"
        )
    if signal_override is not None:
        neutral = get_tools_for_scenario(scenario, mode="neutral")
        override_tool = {
            "type": "function",
            "function": {
                "name": signal_override["name"],
                "description": signal_override["description"],
                "parameters": signal_override.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            },
        }
        return neutral + [override_tool]
    return get_tools_for_scenario(scenario, mode=tool_mode)


class TruncationEngine:
    """Run full-trace and reasoning-truncation inference."""
    def __init__(
        self,
        model: str,
        scenario: str = "blackmail",
        inference_type: str = "vllm",
        temperature: float = 0.6,
        tool_mode: str = "neutral",
        verbose: bool = False,
        load_tokenizer: bool = True,
        signal_override: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the truncation engine."""
        from openai import OpenAI

        self.model = model
        self.scenario = scenario
        self.inference_type = inference_type
        self.temperature = temperature
        self.tool_mode = tool_mode
        self.verbose = verbose
        self.load_tokenizer = load_tokenizer
        self.signal_override = signal_override

        self.tools = _resolve_tools(scenario, tool_mode, signal_override=signal_override)
        self.tokenizer = None

        if inference_type == "vllm":
            self.client = OpenAI(
                api_key=os.getenv('VLLM_API_KEY', 'EMPTY'),
                base_url=os.getenv('VLLM_SERVER', 'http://localhost:8000/v1'),
            )

            models = self.client.models.list()
            model_path = models.data[0].root

            if load_tokenizer:
                if verbose:
                    print(f"[INFO] Loading tokenizer from {model_path}...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            else:
                if verbose:
                    print(f"[INFO] Skipping tokenizer loading (load_tokenizer=False)")

        else:
            from dotenv import load_dotenv
            load_dotenv()

            self.client = OpenAI(
                api_key=os.getenv('OPENAI_API_KEY'),
                base_url=os.getenv('OPENAI_BASE_URL'),
            )

        if load_tokenizer:
            self.predictor = ToolCallPredictor(
                model_name=model,
                tools=self.tools,
                tokenizer=self.tokenizer,
                client=self.client,
            )
        else:
            self.predictor = None

        if verbose:
            print(f"[INFO] Initialized TruncationEngine")
            print(f"[INFO] Scenario: {scenario}, Tool Mode: {tool_mode}")
            print(f"[INFO] Tools: {[t['function']['name'] for t in self.tools]}")


    def set_scenario(self, scenario: str):
        """Switch to a different scenario."""
        self.scenario = scenario
        self.tools = _resolve_tools(scenario, self.tool_mode, signal_override=self.signal_override)

        if self.verbose:
            print(f"[INFO] Switched to scenario: {scenario}")
            print(f"[INFO] Tools: {[t['function']['name'] for t in self.tools]}")



    def run_complete_inference(
        self,
        full_user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run one inference through an OpenAI-compatible chat API."""
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_user_prompt}
            ]

            # Request token IDs only when Stage 2 data is needed.
            extra_body = {}
            if self.load_tokenizer:
                extra_body["return_token_ids"] = True
            if "gemma4" in self.model.lower():
                extra_body["chat_template_kwargs"] = {"enable_thinking": True}
            else:
                extra_body["thinking"] = {"type": "enabled"}

            chat_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                temperature=self.temperature,
                extra_body=extra_body
            )

            message = chat_response.choices[0].message
            content = message.content or ""
            tool_calls = message.tool_calls

            # Token IDs are returned by the vLLM extension fields.
            output_token_ids = getattr(chat_response.choices[0], "token_ids", None) or []
            input_token_ids = getattr(chat_response, "prompt_token_ids", None) or []

            # Extract reasoning text
            reasoning_text = ""
            reasoning_token_ids = []

            if hasattr(message, "reasoning") and message.reasoning:
                reasoning_text = message.reasoning
            elif hasattr(message, "reasoning_content") and message.reasoning_content:
                reasoning_text = message.reasoning_content
            else:
                think_match =re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                if think_match:
                    reasoning_text = think_match.group(1).strip()
            # Extract reasoning token IDs when available.
            if self.load_tokenizer and self.tokenizer is not None and output_token_ids is not None:
                try:
                    reasoning_token_ids = get_reasoning_token_ids(output_token_ids, self.tokenizer, self.model)
                except ValueError:
                    reasoning_token_ids = self.tokenizer.encode(reasoning_text, add_special_tokens=False)

            # Format tool calls
            formatted_tool_calls = []
            if tool_calls:
                for tc in tool_calls:
                    formatted_tool_calls.append({
                        "name": tc.function.name,
                        "arguments": (
                            json.loads(tc.function.arguments)
                            if tc.function.arguments else {}
                        )
                    })
            return {
                "content": content,
                "reasoning": reasoning_text,
                "input_token_ids": input_token_ids if self.load_tokenizer else [],
                "reasoning_ids": reasoning_token_ids if self.load_tokenizer else [],
                "tool_calls": formatted_tool_calls,
                "termination": "tool_call" if tool_calls else "content"
            }

        except Exception as e:
            import traceback
            return {
                "error": str(e),
                "traceback": traceback.format_exc()
            }


    def run_truncated_trace_inference_by_indices(
        self,
        input_token_ids: List[int],
        reasoning_ids: List[int],
        split_indices: List[int],
        batch_size: int = 32,
        progress_bar=None
    ) -> List[Dict[str, Any]]:
        """At each truncation point in split_indices, score the probability of every tool name.
        Process positions in batches for efficiency.
        """
        if self.predictor is None:
            raise RuntimeError("Stage 2 is not available when tokenizer is disabled.")

        tool_names = [t["function"]["name"] for t in self.tools]

        # Collect all reasoning prefixes
        all_prefix_ids = [
            get_reasoning_prefix_from_indices(reasoning_ids, split_indices, step_idx)
            for step_idx in range(len(split_indices))
        ]

        # Process in batches
        all_tool_probs = []
        for i in range(0, len(all_prefix_ids), batch_size):
            batch_prefix_ids = all_prefix_ids[i:i + batch_size]
            batch_results = self.predictor.calculate_tool_name_probs_batch(
                input_token_ids=input_token_ids,
                reasoning_prefix_token_ids_list=batch_prefix_ids,
                tool_names=tool_names,
            )
            all_tool_probs.extend(batch_results)

            if progress_bar is not None:
                progress_bar.update(len(batch_prefix_ids))

        # Build results
        preds = []
        for step_idx, step_pos in enumerate(split_indices):
            reasoning_prefix_token_ids = all_prefix_ids[step_idx]
            tool_probs = all_tool_probs[step_idx]

            preds.append({
                "step_index": step_idx,
                "token_position": step_pos,
                "reasoning_content": self.tokenizer.decode(reasoning_prefix_token_ids),
                "tool_name_probs": tool_probs,
            })

        return preds
