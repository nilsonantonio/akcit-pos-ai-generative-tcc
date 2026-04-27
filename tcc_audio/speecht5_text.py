"""Text normalization and token-audit helpers for SpeechT5."""

from __future__ import annotations

import re
import unicodedata


_WHITESPACE_RE = re.compile(r"\s+")
_TYPOGRAPHIC_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00ab": '"',
        "\u00bb": '"',
    }
)


def normalize_text_for_speecht5(text: object) -> str:
    value = str(text or "")
    value = value.translate(_TYPOGRAPHIC_TRANSLATION)
    value = unicodedata.normalize("NFKD", value)
    value = "".join(character for character in value if not unicodedata.combining(character))
    value = _WHITESPACE_RE.sub(" ", value).strip()
    return value


def _flatten_input_ids(input_ids: object) -> list[int]:
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if isinstance(input_ids, list):
        return [int(token_id) for token_id in input_ids]
    return [int(input_ids)]


def count_unk_tokens(text: object, tokenizer: object) -> int:
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if unk_token_id is None:
        raise ValueError("Tokenizer does not define unk_token_id")
    encoded = tokenizer(str(text or ""), return_attention_mask=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return sum(token_id == int(unk_token_id) for token_id in _flatten_input_ids(input_ids))


def has_unk_tokens(text: object, tokenizer: object) -> bool:
    return count_unk_tokens(text, tokenizer) > 0
