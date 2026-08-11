"""
Extraction pipeline, ported from OpenSPG/OneKE's "quick mode":

    schema_agent.get_retrieved_schema -> extraction_agent.extract_information_direct
                                       -> extraction_agent.summarize_answer

(see ``src/pipeline.py`` and ``src/modules/extraction_agent.py`` in OneKE).

This module keeps OneKE's chunk -> per-chunk extract -> summarize structure
and its JSON-recovery regex (``utils/process.py:extract_json_dict``), but
replaces the nltk sentence tokenizer with a small regex splitter so the app
has no nltk-data download step, and always talks to an OpenAI-compatible
small LLM (``src/llm.py``) instead of a local torch model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from json_repair import repair_json

from typing import Protocol

from .prompts import TASK_LABELS, build_extract_prompt, build_summarize_prompt

CHUNK_TOKEN_LIMIT = 900  # OneKE default is 1024; slightly lower for small models' context.

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\u00C0-\u017F])")


def sent_tokenize(text: str) -> list[str]:
    """Lightweight stand-in for nltk's sent_tokenize (OneKE uses nltk;
    dropped here to avoid an nltk-data download step on Render)."""
    text = text.strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_str(text: str, token_limit: int = CHUNK_TOKEN_LIMIT) -> list[str]:
    """Ported from OneKE's utils/process.py:chunk_str."""
    sentences = sent_tokenize(text)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for sentence in sentences:
        token_count = len(sentence.split())
        if current_length + token_count <= token_limit:
            current_chunk.append(sentence)
            current_length += token_count
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = token_count
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks or ([text] if text.strip() else [])


def _process_single_quotes(text: str) -> str:
    return re.sub(r"(?<!\w)'|'(?!\w)", '"', text)


def _remove_empty_values(obj):
    if isinstance(obj, dict):
        return {k: _remove_empty_values(v) for k, v in obj.items() if v not in ("", None, [], {})}
    if isinstance(obj, list):
        return [_remove_empty_values(v) for v in obj if v not in ("", None, [], {})]
    return obj


def extract_json_dict(text):
    """Ported from OneKE's utils/process.py:extract_json_dict -- recovers the
    last well-formed JSON object out of an LLM's free-form response.

    Extended with a ``json_repair`` fallback: small LLMs frequently emit
    entity names with embedded, unescaped quotes (e.g. nicknames like
    ``"Shurandy "Tyson" Q."``), which breaks strict ``json.loads``. OneKE's
    original version just returned the broken string in that case; here we
    try to repair it, and only fall back to a plain-text result (never a
    dangling JSON-looking string) if repair also fails -- passing a
    not-quite-JSON string to the frontend is what caused the "Json Parse
    Error" the UI used to show.
    """
    if isinstance(text, dict):
        return text
    pattern = r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\})*)*\}"
    matches = re.findall(pattern, text or "")
    if not matches:
        return {"raw_response": text} if text else "No valid information found."

    json_string = _process_single_quotes(matches[-1])
    try:
        json_dict = json.loads(json_string)
    except json.JSONDecodeError:
        try:
            json_dict = json.loads(repair_json(json_string))
        except Exception:
            # Give up on parsing, but still hand back a dict so the UI
            # never receives a malformed JSON *string* to render.
            return {"raw_response": text}

    json_dict = _remove_empty_values(json_dict)
    return json_dict if json_dict is not None else "No valid information found."


class ChatEngine(Protocol):
    """Structural type: anything with get_chat_response() works here --
    covers both ApiLLMEngine and LocalHFEngine from src/llm.py without
    hard-importing either concrete class name."""

    model: str

    def get_chat_response(self, prompt: str) -> str: ...


@dataclass
class ExtractionResult:
    task: str
    label: str
    result: dict | str
    chunks_used: int
    raw_responses: list[str] = field(default_factory=list)
    error: str | None = None


def run_extraction(llm: ChatEngine, task: str, text: str) -> ExtractionResult:
    """Faithful port of ExtractionAgent.extract_information_direct +
    ExtractionAgent.summarize_answer for a single task on a single document."""
    label = TASK_LABELS[task]
    try:
        chunks = chunk_str(text)
        if not chunks:
            return ExtractionResult(task, label, {}, 0, [], error="No text to extract from.")

        result_list = []
        raw_responses = []
        for chunk in chunks:
            prompt = build_extract_prompt(task, chunk)
            response = llm.get_chat_response(prompt)
            raw_responses.append(response)
            result_list.append(extract_json_dict(response))

        if len(result_list) == 1:
            final = result_list[0]
        else:
            summarize_prompt = build_summarize_prompt(task, result_list)
            summarize_response = llm.get_chat_response(summarize_prompt)
            raw_responses.append(summarize_response)
            final = extract_json_dict(summarize_response)

        return ExtractionResult(task, label, final, len(chunks), raw_responses)
    except Exception as exc:  # surface API/network errors to the UI instead of crashing
        return ExtractionResult(task, label, {}, 0, [], error=str(exc))
