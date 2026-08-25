#!/usr/bin/env python3
"""Decoding engine for online-intervention and system-prompt defenses."""

import json
import re
import math
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer
from openai import OpenAI

from pipeline.reasoning_splitter import split_reasoning_steps_by_ids


class DefenseEngine:
    """Run defended reasoning and tool-call decoding."""

    def __init__(
        self,
        model_name: str,
        tools: List[Dict],
        tokenizer: AutoTokenizer,
        client: OpenAI,
        intervention_text: str = "Wait, let me think...",
        system_prompt_patch: str = "",
        mode: str = "intervention",  # intervention, prompting, all, or none
    ):
        """Initialize the defense engine."""
        self.model_name = model_name
        self.tools = tools
        self.intervention_text = intervention_text
        self.system_prompt_patch = system_prompt_patch
        self.mode = mode
        self.client = client
        self.tokenizer = tokenizer

        self.signal_tool_names = self._get_signal_tool_names()
        self.signal_tool_name = self.signal_tool_names[0] if self.signal_tool_names else None
        self.tool_names = [tool.get("function", {}).get("name", "") for tool in tools]

    def _get_signal_tool_names(self) -> List[str]:
        """Extract signal tool names from tools list."""
        return [
            tool.get("function", {}).get("name", "")
            for tool in self.tools
            if tool.get("function", {}).get("name", "") not in ["email", "forward", "cancel_alert"]
        ]

    def _build_reasoning_extra_body(self) -> Dict[str, Any]:
        extra_body = {"return_token_ids": True}
        if "gemma4" in self.model_name.lower():
            extra_body["chat_template_kwargs"] = {"enable_thinking": True}
        else:
            extra_body["thinking"] = {"type": "enabled"}
        return extra_body

    def _format_api_tool_calls(self, tool_calls: Any) -> List[Dict]:
        formatted_tool_calls = []
        if not tool_calls:
            return formatted_tool_calls

        for tool_call in tool_calls:
            function = getattr(tool_call, "function", None)
            if function is None:
                continue
            arguments = getattr(function, "arguments", None)
            try:
                parsed_arguments = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                parsed_arguments = arguments or {}
            formatted_tool_calls.append({
                "name": getattr(function, "name", ""),
                "arguments": parsed_arguments,
            })
        return formatted_tool_calls

    def _extract_reasoning_from_chat_response(self, chat_response: Any) -> Dict[str, Any]:
        message = chat_response.choices[0].message
        content = message.content or ""

        reasoning_text = ""
        if hasattr(message, "reasoning") and message.reasoning:
            reasoning_text = message.reasoning
        elif hasattr(message, "reasoning_content") and message.reasoning_content:
            reasoning_text = message.reasoning_content
        else:
            think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
            if think_match:
                reasoning_text = think_match.group(1).strip()

        output_token_ids = getattr(chat_response.choices[0], "token_ids", None) or []
        reasoning_token_ids = self._get_reasoning_token_ids(output_token_ids)

        return {
            "content": content,
            "reasoning": reasoning_text,
            "reasoning_ids": reasoning_token_ids,
            "input_token_ids": getattr(chat_response, "prompt_token_ids", None) or [],
            "tool_calls": self._format_api_tool_calls(getattr(message, "tool_calls", None)),
        }

    def _tool_calls_are_safe(self, tool_calls: List[Dict]) -> bool:
        if not self.signal_tool_names:
            return True
        return all(tool_call.get("name", "") not in self.signal_tool_names for tool_call in tool_calls)

    def parse_tool_calls(self, generated_text: str) -> List[Dict]:
        """
        Parse multiple tool calls from generated text.
        """
        model_lower = self.model_name.lower()

        if "gemma" in model_lower:
            return self._parse_gemma_tool_calls(generated_text)
        elif "qwen" in model_lower:
            if self._is_qwen35_style():
                return self._parse_qwen35_tool_calls(generated_text)
            return self._parse_qwen3_tool_calls(generated_text)
        else:
            raise ValueError(f"Unsupported model '{self.model_name}' for tool call parsing")

    def _parse_qwen3_tool_calls(self, generated_text: str) -> List[Dict]:
        """
        Parse qwen3-style tool calls (JSON format).
        Format: {"name": tool_name, "arguments": {"parameter": "value", "parameter": "value", ...}}
        """
        full_text = "<tool_call>\n" + generated_text

        pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        matches = re.findall(pattern, full_text, re.DOTALL)

        results = []
        for match in matches:
            parsed = None
            try:
                parsed = json.loads(match.strip())
            except json.JSONDecodeError:
                m = re.search(r'\{.*\}', match, re.DOTALL)
                if m:
                    try:
                        parsed = json.loads(m.group())
                    except json.JSONDecodeError:
                        pass
            if parsed:
                results.append(parsed)

        return results

    def _parse_qwen35_tool_calls(self, generated_text: str) -> List[Dict]:
        """
        Parse qwen3.5-style tool calls (XML-like format).
        Format:
            <function=tool_name>
            <parameter=...>
            ...
            </parameter>
            <parameter=...>
            ...
            </parameter>
            </function>
        """
        pattern = r'<function=(.+?)>(.*?)</function>'
        matches = re.findall(pattern, generated_text, re.DOTALL)

        results = []
        for func_name, content in matches:
            param_pattern = r'<parameter=(.+?)>\s*(.*?)\s*</parameter>'
            params = re.findall(param_pattern, content, re.DOTALL)

            arguments = {}
            for param_name, param_value in params:
                arguments[param_name] = param_value.strip()

            results.append({
                "name": func_name.strip(),
                "arguments": arguments
            })

        return results

    def _parse_gemma_tool_calls(self, generated_text: str) -> List[Dict]:
        """
        Parse gemma4-style tool calls (custom format).
        Format: <|tool_call>call:tool_name{key1:<|"|>value1<|"|>,key2:<|"|>value2<|"|>}<tool_call|>
        """
        results = []
        call_pattern = re.compile(r'call:([a-zA-Z0-9_\-]+)\{')
        pos = 0

        while True:
            match = call_pattern.search(generated_text, pos)
            if not match:
                break

            func_name = match.group(1).strip()
            args_start = match.end()
            depth = 1
            i = args_start
            while i < len(generated_text) and depth:
                if generated_text[i] == "{":
                    depth += 1
                elif generated_text[i] == "}":
                    depth -= 1
                i += 1

            if depth:
                break

            args_content = generated_text[args_start:i - 1]

            arguments = {}
            for key, value in self._parse_gemma_args(args_content):
                arguments[key] = value.strip()

            results.append({
                "name": func_name,
                "arguments": arguments
            })
            pos = i

        return results

    def _parse_gemma_args(self, args_content: str) -> List[tuple[str, str]]:
        if not args_content.strip():
            return []

        args = []
        current = []
        depth = 0
        in_gemma_quote = False
        quote_marker = '<|"|>'
        i = 0

        while i < len(args_content):
            if args_content.startswith(quote_marker, i):
                in_gemma_quote = not in_gemma_quote
                current.append(quote_marker)
                i += len(quote_marker)
                continue

            char = args_content[i]
            if not in_gemma_quote:
                if char == "{":
                    depth += 1
                elif char == "}" and depth:
                    depth -= 1
                elif char == "," and depth == 0:
                    part = "".join(current).strip()
                    if part:
                        args.append(part)
                    current = []
                    i += 1
                    continue

            current.append(char)
            i += 1

        part = "".join(current).strip()
        if part:
            args.append(part)

        parsed = []
        for arg in args:
            key, sep, value = arg.partition(":")
            if sep and re.fullmatch(r"[a-zA-Z0-9_\-]+", key.strip()):
                parsed.append((key.strip(), value.strip()))
        return parsed

    def _get_reasoning_token_ids(self, token_ids: List[int]) -> List[int]:
        """Extract reasoning token IDs from a generated token sequence."""
        model_lower = self.model_name.lower()

        if "qwen" in model_lower:
            begin_token_id = self.tokenizer.convert_tokens_to_ids('<think>')
            end_token_id = self.tokenizer.convert_tokens_to_ids('</think>')
            start_offset = 2
        elif "gemma4" in model_lower:
            begin_token_id = self.tokenizer.convert_tokens_to_ids("<|channel>")
            end_token_id = self.tokenizer.convert_tokens_to_ids('<channel|>')
            start_offset = 3
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        try:
            start_index = token_ids.index(begin_token_id) + start_offset
        except ValueError:
            start_index = 0

        try:
            end_index = token_ids.index(end_token_id, start_index)
        except ValueError:
            end_index = len(token_ids)

        return token_ids[start_index:end_index]

    def _has_reasoning_end_marker(self, token_ids: List[int], text: str = "") -> bool:
        """Return True if generated continuation contains the reasoning-channel end."""
        model_lower = self.model_name.lower()

        if "qwen" in model_lower:
            end_token = "</think>"
        elif "gemma4" in model_lower:
            end_token = "<channel|>"
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        end_token_id = self.tokenizer.convert_tokens_to_ids(end_token)
        return (
            end_token in (text or "")
            or (
                end_token_id is not None
                and end_token_id != self.tokenizer.unk_token_id
                and end_token_id in token_ids
            )
        )

    def build_prefix_prompt_ids(
        self,
        input_token_ids:  List[int],
        reasoning_prefix_token_ids: List[int]
    ) -> List[int]:
        """Build a prefix ending at the tool-call boundary."""
        model_lower = self.model_name.lower()

        if "qwen" in model_lower:
            start_token_id = self.tokenizer.encode("<think>\n", add_special_tokens=False)
            think_end_and_tool_call = self.tokenizer.encode("</think>\n\n<tool_call>\n", add_special_tokens=False)
        elif "gemma" in model_lower:
            start_token_id = self.tokenizer.encode("<|channel>thought\n", add_special_tokens=False)
            think_end_and_tool_call = self.tokenizer.encode("<channel|>\n<|tool_call>", add_special_tokens=False)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        prefix_prompt_ids = input_token_ids + start_token_id + reasoning_prefix_token_ids + think_end_and_tool_call
        return prefix_prompt_ids

    def build_reasoning_prefix_prompt_ids(
        self,
        input_token_ids:  List[int],
        reasoning_prefix_token_ids: List[int]
    ) -> List[int]:
        """Build a prefix for continuing reasoning generation."""
        model_lower = self.model_name.lower()

        if "qwen" in model_lower:
            start_token_id = self.tokenizer.encode("<think>\n", add_special_tokens=False)
        elif "gemma" in model_lower:
            start_token_id = self.tokenizer.encode("<|channel>thought\n", add_special_tokens=False)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        prefix_prompt_ids = input_token_ids + start_token_id + reasoning_prefix_token_ids
        return prefix_prompt_ids


    def _is_qwen35_style(self) -> bool:
        if "qwen" not in self.model_name.lower():
            return False
        return (
            "3.5" in self.model_name
            or "3-5" in self.model_name
        )

    def _get_tool_call_token(self) -> List[int]:
        """
        Return token IDs for the text immediately preceding a tool name.

        - qwen3 JSON:   {"name": "
        - qwen3.5 XML: <function=
        - gemma4:       call:
        """
        model_lower = self.model_name.lower()
        if "gemma" in model_lower:
            stub = "call:"
        elif "qwen" in model_lower:
            if self._is_qwen35_style():
                stub = "<function="
            else:
                stub =  '{"name": "'
        else:
            raise ValueError(
                f"Unsupported model '{self.model_name}' for tool-name probability (use qwen / gemma)"
            )
        return self.tokenizer.encode(stub, add_special_tokens=False)

    def calculate_all_tools_probs(
        self,
        input_token_ids: List[int],
        reasoning_prefix_token_ids: List[int],
    ) -> Dict[str, Optional[float]]:
        """
        First-token probability of each tool at the current reasoning prefix.
        """
        prefix_prompt_ids = self.build_prefix_prompt_ids(input_token_ids, reasoning_prefix_token_ids) + self._get_tool_call_token()

        tool_first_tokens = {}
        for tool_name in self.tool_names:
            tool_token_ids = self.tokenizer.encode(tool_name, add_special_tokens=False)
            tool_first_tokens[tool_name] = self.tokenizer.decode([tool_token_ids[0]])

        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prefix_prompt_ids,
                max_tokens=1,
                temperature=0,
                logprobs=40
            )

            top_logprobs = response.choices[0].logprobs.top_logprobs[0]
            result = {}
            for tool_name in self.tool_names:
                token_logprob = None
                for tok_str, logprob in top_logprobs.items():
                    if tok_str == tool_first_tokens[tool_name]:
                        token_logprob = logprob
                        break

                if token_logprob is None:
                    token_logprob = -100

                # exp(logprob) is the first-token probability (higher = more likely).
                result[tool_name] = math.exp(token_logprob)
            return result

        except Exception as e:
            print(f"[ERROR] Failed to calculate tool probabilities: {e}")
            return {tool_name: None for tool_name in self.tool_names}

    def calculate_tendency(
        self,
        all_tool_probs: Dict[str, Optional[float]]
    ) -> bool:
        """Return whether the intent tool has the highest available probability."""
        if not self.signal_tool_name or self.signal_tool_name not in all_tool_probs:
            return False

        finite_probs = {
            tool_name: prob
            for tool_name, prob in all_tool_probs.items()
            if prob is not None
        }
        if not finite_probs or self.signal_tool_name not in finite_probs:
            return False

        return finite_probs[self.signal_tool_name] == max(finite_probs.values())

    def calculate_sentence_probs_data(
        self,
        input_token_ids: List[int],
        reasoning_token_ids: List[int],
    ) -> List[Dict[str, Any]]:
        sentence_probs_data = []
        if not reasoning_token_ids:
            return sentence_probs_data

        split_indices = split_reasoning_steps_by_ids(
            reasoning_token_ids,
            self.tokenizer,
            granularity="sentence",
        )
        sentence_count = 0
        sentence_endings = [".", "!", "?", "。", "！", "？"]

        for i in range(len(split_indices) - 1):
            sentence_token_ids = reasoning_token_ids[split_indices[i]:split_indices[i+1]]
            raw_sentence_text = self.tokenizer.decode(sentence_token_ids)
            sentence_text = raw_sentence_text.strip()
            is_complete_sentence = (
                any(sentence_text.endswith(ending) for ending in sentence_endings)
                or sentence_text.endswith(":")
                or "\n\n" in raw_sentence_text
            )
            if not is_complete_sentence:
                continue

            sentence_count += 1
            reasoning_prefix_token_ids = reasoning_token_ids[:split_indices[i+1]]
            sentence_probs_data.append({
                "sentence_index": sentence_count,
                "sentence_text": raw_sentence_text,
                "all_tool_probs": self.calculate_all_tools_probs(input_token_ids, reasoning_prefix_token_ids),
            })

        return sentence_probs_data

    def prompting_decode_with_defense(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 1.0,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Run one-shot decoding with the defense text appended to the system prompt.
        """
        if self.mode != "none":
            defended_system_prompt = system_prompt.rstrip() + "\n\n" + self.system_prompt_patch
        else:
            defended_system_prompt = system_prompt
        messages = [
            {"role": "system", "content": defended_system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        chat_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=self.tools,
            temperature=temperature,
            extra_body=self._build_reasoning_extra_body(),
            max_tokens=max_tokens,
        )

        parsed = self._extract_reasoning_from_chat_response(chat_response)
        sentence_probs_data = self.calculate_sentence_probs_data(
            parsed["input_token_ids"],
            parsed["reasoning_ids"],
        )

        return {
            "system_prompt": defended_system_prompt,
            "reasoning": parsed["reasoning"],
            "tool_calls": parsed["tool_calls"],
            "intervention_history": [],
            "sentence_probs_data": sentence_probs_data,
            "num_interventions": 0,
            "safe": self._tool_calls_are_safe(parsed["tool_calls"]),
            "sentence_count": len(sentence_probs_data),
        }


    def online_decode_with_defense(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Perform online decoding with defense intervention.

        Like Stage 1, but with real-time monitoring and intervention during reasoning.
        """
        if self.mode == "all":
            system_prompt = system_prompt.rstrip() + "\n\n" + self.system_prompt_patch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        extra_body = self._build_reasoning_extra_body()

        chat_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=self.tools,
            temperature=temperature,
            extra_body=extra_body,
            max_tokens=256,
        )
        input_token_ids = getattr(chat_response, "prompt_token_ids", None) or []
        pending_token_ids = getattr(chat_response.choices[0], "token_ids", None) or []
        pending_token_ids = self._get_reasoning_token_ids(pending_token_ids)

        reasoning_so_far_ids = []
        intervention_history = []
        sentence_probs_data = []
        safe = True
        sentence_count = 0
        forced_stop_after_max_interventions = False
        finished = chat_response.choices[0].finish_reason == "stop"

        max_iterations = 80  # Bound continuation attempts.
        max_interventions = 40  # Block if repeated interventions do not suppress the signal.
        iteration = 0

        # Complete sentences in `pending` move into the reasoning prefix; the
        # trailing incomplete sentence stays in `pending` for the next round.
        def process_complete_sentences() -> bool:
            """Commit complete sentences and return True if an intervention was inserted."""
            nonlocal pending_token_ids, reasoning_so_far_ids, sentence_count, forced_stop_after_max_interventions

            split_indices = split_reasoning_steps_by_ids(pending_token_ids, self.tokenizer, granularity="sentence")
            last_processed_token_index = 0

            for i in range(len(split_indices) - 1):
                sentence_token_ids = pending_token_ids[split_indices[i]:split_indices[i+1]]
                # Only run the probability check on complete sentences; never intervene on a partial one.
                raw_sentence_text = self.tokenizer.decode(sentence_token_ids)
                sentence_text = raw_sentence_text.strip()
                sentence_endings = [".", "!", "?", "。", "！", "？"]
                is_complete_sentence = (
                    any(sentence_text.endswith(end) for end in sentence_endings)
                    or sentence_text.endswith(":")
                    or "\n\n" in raw_sentence_text
                )
                # Incomplete sentence: leave it in `pending` and stop here.
                if not is_complete_sentence:
                    break

                # Commit the complete sentence to the reasoning prefix.
                reasoning_so_far_ids += sentence_token_ids
                sentence_count += 1
                last_processed_token_index = split_indices[i+1]

                all_tool_probs = self.calculate_all_tools_probs(input_token_ids, reasoning_so_far_ids)

                # Check the tendency: whether the signal tool is the most likely next tool.
                interventions_enabled = len(intervention_history) < max_interventions
                tendency = (
                    self.calculate_tendency(all_tool_probs)
                    if interventions_enabled
                    else False
                )

                # Record per-sentence probability data.
                sentence_probs_data.append({
                    "sentence_index": sentence_count,
                    "sentence_text": self.tokenizer.decode(sentence_token_ids),
                    "all_tool_probs": all_tool_probs,
                })

                # On a detected tendency, insert an intervention and restart continuation.
                if tendency:
                    intervention_history.append({
                        "sentence_index": sentence_count,
                        "sentence_text": self.tokenizer.decode(sentence_token_ids),
                        "all_tool_probs": all_tool_probs,
                    })

                    intervention_text = f" {self.intervention_text}\n\n"
                    reasoning_so_far_ids += self.tokenizer.encode(intervention_text, add_special_tokens=False)
                    sentence_count += 1
                    if len(intervention_history) >= max_interventions:
                        forced_stop_after_max_interventions = True
                    # Recompute probabilities after the intervention, record it, and break.
                    all_tool_probs = self.calculate_all_tools_probs(input_token_ids, reasoning_so_far_ids)
                    sentence_probs_data.append({
                        "sentence_index": sentence_count,
                        "sentence_text": intervention_text,
                        "all_tool_probs": all_tool_probs,
                        "forced_stop_after_max_interventions": forced_stop_after_max_interventions,
                    })

                    # After an intervention, regenerate from the new prefix; discard the rest
                    # of this chunk since it was produced from the pre-intervention context.
                    pending_token_ids = []
                    return True

            if last_processed_token_index:
                pending_token_ids = pending_token_ids[last_processed_token_index:]

            return False

        while True:
            iteration += 1
            if iteration > max_iterations:
                print(f"[WARNING] Reached max iterations ({max_iterations}), forcing completion")
                break
            # Process complete sentences accumulated so far.
            intervention_inserted = process_complete_sentences()
            if forced_stop_after_max_interventions:
                break
            if intervention_inserted:
                finished = False

            if finished:
                break

            # Continue generation from the current reasoning prefix.
            next_prompt_ids = self.build_reasoning_prefix_prompt_ids(input_token_ids, reasoning_so_far_ids) + pending_token_ids

            response = self.client.completions.create(
                model=self.model_name,
                prompt=next_prompt_ids,
                temperature=temperature,
                logprobs=20,
                max_tokens=512,
                extra_body={"return_token_ids": True},
                stop=["</think>", "<channel|>"],
            )
            raw_new_token_ids = getattr(response.choices[0], "token_ids", None) or []
            raw_new_text = getattr(response.choices[0], "text", None) or ""
            finished = (
                response.choices[0].finish_reason == "stop"
                or self._has_reasoning_end_marker(raw_new_token_ids, raw_new_text)
            )
            new_token_ids = self._get_reasoning_token_ids(raw_new_token_ids)
            if not new_token_ids and not finished:
                print("[WARNING] Model returned no new tokens before a stop signal; forcing completion")
                break
            pending_token_ids += new_token_ids

        if pending_token_ids:
            reasoning_so_far_ids += pending_token_ids
            pending_token_ids = []

        if forced_stop_after_max_interventions:
            return {
                "reasoning": self.tokenizer.decode(reasoning_so_far_ids),
                "tool_calls": [],
                "intervention_history": intervention_history,
                "sentence_probs_data": sentence_probs_data,
                "num_interventions": len(intervention_history),
                "forced_stop_after_max_interventions": forced_stop_after_max_interventions,
                "blocked_by_intervention": True,
                "termination": "blocked",
                "safe": True,
                "sentence_count": sentence_count,
            }

        # Reasoning is complete: generate the final tool call in a single pass.
        next_prompt_ids = self.build_prefix_prompt_ids(input_token_ids, reasoning_so_far_ids)
        response = self.client.completions.create(
            model=self.model_name,
            prompt=next_prompt_ids,
            temperature=0,
            logprobs=20,
            max_tokens=2048,
        )

        final_tool_calls = self.parse_tool_calls(response.choices[0].text)
        safe = self._tool_calls_are_safe(final_tool_calls)

        return {
            "reasoning": self.tokenizer.decode(reasoning_so_far_ids),
            "tool_calls": final_tool_calls,
            "intervention_history": intervention_history,
            "sentence_probs_data": sentence_probs_data,
            "num_interventions": len(intervention_history),
            "forced_stop_after_max_interventions": forced_stop_after_max_interventions,
            "blocked_by_intervention": False,
            "termination": "tool_call" if final_tool_calls else "content",
            "safe": safe,
            "sentence_count": sentence_count,
        }
