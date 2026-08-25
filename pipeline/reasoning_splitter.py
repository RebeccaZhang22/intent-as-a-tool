"""Split reasoning traces into truncation points using token IDs."""

import re
from typing import List, Tuple
from transformers import PreTrainedTokenizer


def split_reasoning_steps_by_ids(
    reasoning_ids: List[int],
    tokenizer: PreTrainedTokenizer,
    granularity: str = "token",
) -> List[int]:
    """
    Split reasoning into truncation points based on granularity.
    """
    if not reasoning_ids:
        return []

    if granularity == "token":
        # Token-level: split at every position
        return list(range(0, len(reasoning_ids) + 1))

    elif granularity == "paragraph":
        # Paragraph-level: split at blank lines.
        split_indices = [0]  # Start with 0 for empty prefix

        for i, token_id in enumerate(reasoning_ids):
            token_text = tokenizer.decode([token_id])
            if '\n\n' in token_text:
                split_indices.append(i + 1)

        # Always add the final position
        if split_indices[-1] != len(reasoning_ids):
            split_indices.append(len(reasoning_ids))

        return split_indices

    elif granularity == "sentence":
        return _split_by_sentence(reasoning_ids, tokenizer)

    else:
        raise ValueError(f"Unknown granularity: {granularity}. Use 'token', 'paragraph' or 'sentence'.")


def _split_by_sentence(
    reasoning_ids: List[int],
    tokenizer: PreTrainedTokenizer,
) -> List[int]:
    """
    Split by sentence boundaries.

    A boundary is detected after:
      - ". " / "! " / "? " (end-of-sentence punctuation followed by whitespace)
      - "\\n" (line break)

    Whitespace tokens (like newlines) are included with the preceding sentence,
    not at the start of the new sentence.
    """
    # Build per-token character spans and decode each token
    token_end_char: List[int] = []
    token_texts: List[str] = []
    cumulative = ""
    for tid in reasoning_ids:
        token_text = tokenizer.decode([tid])
        token_texts.append(token_text)
        cumulative += token_text
        token_end_char.append(len(cumulative))

    # Sentence boundary: after . ! ? + whitespace, or after one or more line breaks.
    # Do not split numbered-list markers like "1. " as standalone sentences.
    boundary_re = re.compile(r'(?<!\d)(?<=[.!?])[ \t]+|\n+')

    split_indices = [0]
    for m in boundary_re.finditer(cumulative):
        boundary_char = m.end()

        # Find the first token at or after the boundary
        for i, end_pos in enumerate(token_end_char):
            if end_pos >= boundary_char:
                idx = i
                # Keep newline and whitespace tokens with the preceding sentence.
                while idx < len(reasoning_ids):
                    token_text = token_texts[idx]
                    stripped = token_text.strip()
                    if not stripped or '\n' in token_text:
                        idx += 1
                    else:
                        break

                if idx > split_indices[-1] and idx <= len(reasoning_ids):
                    split_indices.append(idx)
                break

    if split_indices[-1] != len(reasoning_ids):
        split_indices.append(len(reasoning_ids))

    return split_indices


def get_reasoning_prefix_from_indices(
    reasoning_ids: List[int],
    split_indices: List[int],
    step_idx: int
) -> List[int]:
    """
    Get reasoning prefix up to step_idx (for truncated inference).

    Example:
        split_indices=[0,3,7,10], step_idx=0 -> []
        split_indices=[0,3,7,10], step_idx=1 -> [1,2,3]
        split_indices=[0,3,7,10], step_idx=2 -> [1,2,3,4,5,6,7]
    """
    if not split_indices:
        return []

    if step_idx < 0 or step_idx >= len(split_indices):
        return []

    return reasoning_ids[:split_indices[step_idx]]
