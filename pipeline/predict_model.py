"""Tool call predictor using prefix continuation with vLLM completions API."""

import math
from typing import Dict, List, Optional
from transformers import AutoTokenizer
from openai import OpenAI


class ToolCallPredictor:
    """Tool call predictor that uses reasoning prefix continuation."""

    def __init__(
        self,
        model_name: str,
        tools: Optional[List[Dict]] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        client: Optional[OpenAI] = None,
    ):
        self.model_name = model_name
        self.tools = tools or []
        self.tokenizer = tokenizer
        self.client = client

    # =========================================================================
    # Prompt construction
    # =========================================================================

    def build_prefix_prompt_ids(
        self,
        input_token_ids: List[int],
        reasoning_prefix_token_ids: List[int],
    ) -> List[int]:
        """Build: input + <think> + reasoning_prefix + </think><tool_call>."""
        model_lower = self.model_name.lower()

        if "qwen" in model_lower:
            start = self.tokenizer.encode("<think>\n", add_special_tokens=False)
            end = self.tokenizer.encode("</think>\n\n<tool_call>\n", add_special_tokens=False)
        elif "gemma" in model_lower:
            start = self.tokenizer.encode("<|channel>thought\n", add_special_tokens=False)
            end = self.tokenizer.encode("<channel|>\n<|tool_call>", add_special_tokens=False)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return input_token_ids + start + reasoning_prefix_token_ids + end

    def _is_qwen35_style(self) -> bool:
        if "qwen" not in self.model_name.lower():
            return False
        return "3.5" in self.model_name or "3-5" in self.model_name

    def _get_tool_call_opener_ids(self) -> List[int]:
        """Token ids for the tool-call format opener before the tool name.

        - qwen3:   {"name": "
        - qwen3.5: <function=
        - gemma4:  call:
        """
        model_lower = self.model_name.lower()
        if "gemma" in model_lower:
            stub = "call:"
        elif "qwen" in model_lower:
            stub = "<function=" if self._is_qwen35_style() else '{"name": "'
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        return self.tokenizer.encode(stub, add_special_tokens=False)

    # =========================================================================
    # Tool name probability (first-token logprobs)
    # =========================================================================

    def calculate_tool_name_probs_batch(
        self,
        input_token_ids: List[int],
        reasoning_prefix_token_ids_list: List[List[int]],
        tool_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Optional[float]]]:
        """Batch compute first-token probability for each tool name.

        Appends the tool-call opener to each prefix prompt, generates 1 token
        with logprobs=50, and reads off P(first_token_of_tool_name).
        """
        names = tool_names if tool_names is not None else [
            t["function"]["name"] for t in self.tools
        ]

        opener_ids = self._get_tool_call_opener_ids()
        prompt_ids_list = [
            self.build_prefix_prompt_ids(input_token_ids, prefix_ids) + opener_ids
            for prefix_ids in reasoning_prefix_token_ids_list
        ]

        # Map each tool name to its decoded first token.
        tool_first_tokens = {}
        for name in names:
            ids = self.tokenizer.encode(name, add_special_tokens=False)
            tool_first_tokens[name] = self.tokenizer.decode([ids[0]])

        results: List[Dict[str, Optional[float]]] = []

        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt_ids_list,
                max_tokens=1,
                temperature=1,
                logprobs=50,
                timeout=300,
            )
            for choice in sorted(response.choices, key=lambda c: c.index):
                top_logprobs = choice.logprobs.top_logprobs[0]
                result = {}
                for tool_name, first_tok_str in tool_first_tokens.items():
                    logprob = top_logprobs.get(first_tok_str, -100)
                    result[tool_name] = math.exp(logprob)
                results.append(result)

        except Exception as e:
            print(f"[ERROR] Batch tool name prob calculation failed: {e}")
            empty = {name: None for name in names}
            results = [empty.copy() for _ in reasoning_prefix_token_ids_list]

        return results
