"""
Small-scale LLM engine wrappers.

Ships two kinds of engine there: OpenAI-compatible API clients (``ChatGPT``,
``DeepSeek``, ``LocalServer``) and local ``transformers``-based engines
(``LLaMA``, ``Qwen``, ``MiniCPM``, ``ChatGLM``, a fine-tuned 
model itself). This module keeps both flavours, both exposing the same
``get_chat_response(prompt) -> str`` contract ``BaseEngine`` uses,
so the rest of the extraction pipeline (``src/extraction.py``) doesn't care
which one it's talking to:

- ``ApiLLMEngine``   -- OpenAI-compatible hosted APIs (Groq/OpenAI/DeepSeek/
                         Together/custom). No GPU needed.
- ``LocalHFEngine``  -- runs a small open-weight model in-process on CPU
                         with ``transformers``, no external API or key
                         needed. just swapped to genuinely small (<=1.5B)
                         checkpoints so plain-CPU inference is feasible.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class ProviderPreset:
    key: str
    label: str
    base_url: str
    default_model: str
    api_key_env: str
    notes: str


# Curated list of OpenAI-compatible providers that serve genuinely
# small-scale models (a handful of billion parameters) rather than
# frontier-scale ones. "custom" lets you point at any OpenAI-compatible
# endpoint, e.g. a self-hosted vLLM/Ollama server 
# ``LocalServer`` engine.
PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "groq": ProviderPreset(
        key="groq",
        label="Groq — gpt oss 120b (hosted, small, fast, free tier)",
        base_url="https://api.groq.com/openai/v1",
        default_model="openai/gpt-oss-120b",
        api_key_env="GROQ_API_KEY",
        notes="Recommended default: an 8B-parameter open-weight model.",
    ),
    "deepseek": ProviderPreset(
        key="deepseek",
        label="DeepSeek Chat (hosted)",
        base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        notes="Same provider ships a preset for.",
    ),
    "openai": ProviderPreset(
        key="openai",
        label="OpenAI — GPT-4o mini (hosted)",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        notes="'ChatGPT' engine preset.",
    ),
    "together": ProviderPreset(
        key="together",
        label="Together AI — Llama 3.3 70B(hosted)",
        base_url="https://api.together.xyz/v1",
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        api_key_env="TOGETHER_API_KEY",
        notes="Another small open-weight option.",
    ),
    "custom": ProviderPreset(
        key="custom",
        label="Custom OpenAI-compatible endpoint (e.g. your own vLLM/Ollama)",
        base_url="http://localhost:8000/v1",
        default_model="your-model-name",
        api_key_env="CUSTOM_LLM_API_KEY",
        notes="LocalServer engine for self-hosted servers reachable over HTTP.",
    ),
}

# Small open-weight checkpoints that are realistic to run in-process on a
# CPU-only Render instance. Ordered roughly fastest/lightest first. Users
# can also type in any other HF model id.
LOCAL_MODEL_PRESETS: dict[str, str] = {
    "Qwen2.5-0.5B-Instruct (fastest, ~1GB RAM)": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-Instruct (better quality, ~3GB RAM)": "Qwen/Qwen2.5-1.5B-Instruct",
    "SmolLM2-1.7B-Instruct (~3.5GB RAM)": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}


class ApiLLMEngine:
    """OpenAI-compatible chat engine, so the rest of the extraction pipeline is a
    drop-in port of OneKE's ``ExtractionAgent`` / ``SchemaAgent`` logic."""

    kind = "api"

    def __init__(self, provider: str, model_name: str, api_key: str, base_url: str | None = None):
        preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["custom"])
        self.provider = provider
        self.name = preset.label
        self.model = model_name or preset.default_model
        self.base_url = base_url or preset.base_url
        self.temperature = 0.2
        self.top_p = 0.9
        self.max_tokens = 1536
        self.api_key = api_key or os.environ.get(preset.api_key_env, "")
        if not self.api_key:
            raise ValueError(
                f"No API key provided for provider '{provider}'. "
                f"Set it in the sidebar or the {preset.api_key_env} environment variable."
            )
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_chat_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=None,
        )
        return response.choices[0].message.content or ""


class LocalHFEngine:
    """Runs a small open-weight instruct model in-process on CPU via
    ``transformers``, mirroring OneKE's local ``Qwen``/``LLaMA`` engine
    classes but restricted to genuinely small (<=~1.7B parameter)
    checkpoints so it's feasible without a GPU.

    ``torch``/``transformers`` are only imported here, lazily, so a
    deployment that never selects "Local model" doesn't need those (large)
    packages installed at all .
    """

    kind = "local"
    _lock = threading.Lock()  # transformers .generate() isn't thread-safe across reruns

    def __init__(self, model_name: str, max_new_tokens: int = 768):
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ValueError(
                "Local model support needs the 'torch' and 'transformers' packages, "
                "which aren't installed. Install them with "
                "`pip install -r requirements-local.txt` (see README) and redeploy, "
                "or use one of the hosted API providers instead."
            ) from exc

        self.model_name = model_name
        self.name = f"Local (CPU) — {model_name}"
        self.model = model_name
        self.max_new_tokens = max_new_tokens

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.hf_model.eval()

    def get_chat_response(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        except Exception:
            # Model has no chat template -- fall back to plain completion.
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        with self._lock, self._torch.no_grad():
            output_ids = self.hf_model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def create_llm_engine(
    mode: str,
    provider: str = "",
    model_name: str = "",
    api_key: str = "",
    base_url: str | None = None,
):
    """Factory returning either an ApiLLMEngine or a LocalHFEngine, both
    exposing the same ``get_chat_response`` contract."""
    if mode == "local":
        return LocalHFEngine(model_name=model_name)
    return ApiLLMEngine(provider=provider, model_name=model_name, api_key=api_key, base_url=base_url)
