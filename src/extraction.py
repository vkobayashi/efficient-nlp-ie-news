"""
Extraction pipeline:

    schema_agent.get_retrieved_schema -> extraction_agent.extract_information_direct
                                       -> extraction_agent.summarize_answer

"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from json_repair import repair_json

from typing import Protocol

from .prompts import TASK_LABELS, build_extract_prompt, build_summarize_prompt

CHUNK_TOKEN_LIMIT = 900  # can be adjusted, slightly lower for small models' context.

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\u00C0-\u017F])")


def sent_tokenize(text: str) -> list[str]:
    """Lightweight stand-in for nltk's sent_tokenize."""
    text = text.strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_str(text: str, token_limit: int = CHUNK_TOKEN_LIMIT) -> list[str]:
    
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
    """ recovers the
    last well-formed JSON object out of an LLM's free-form response.

    Extended with a ``json_repair`` fallback: small LLMs frequently emit
    entity names with embedded, unescaped quotes (e.g. nicknames like
    ``"Shurandy "Tyson" Q."``), which breaks strict ``json.loads``. 
    original version just returned the broken string in that case; here we 
     repair it, and only fall back to a plain-text result (never a
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
    covers both ApiLLMEngine and LocalHFEngine from src/llm.py """

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
    """ single task on a single document."""
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
