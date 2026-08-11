# Dutch News → Information Extraction (Prototype - because possible is real)

A Streamlit web app that fetches news articles from major Netherlands news
outlets (NOS, NU.nl, Telegraaf, Volkskrant, NRC, AD, RTL Nieuws, FD) and
runs schema-guided information extraction on them with a **small-scale
LLM**. The user picks which extraction task(s) to run:

| Task | What it extracts |
|---|---|
| **NER** | Named entities and their types |
| **RE** | Relationship triples between named entities |
| **EE** | Events, their triggers, and arguments |
| **Triple** | Open subject–relation–object triples |

## Where OneKE comes in

The extraction logic is a direct, lightweight port of
[OpenSPG/OneKE](https://github.com/OpenSPG/OneKE)'s **"quick mode"**
pipeline:

| OneKE file | Ported to |
|---|---|
| `src/models/prompt_template.py` (`EXTRACT_INSTRUCTION`, `SUMMARIZE_INSTRUCTION`, `instruction_mapper`) | `src/prompts.py` |
| `src/modules/schema_agent.py` (default NER/RE/EE/Triple Pydantic schemas) | `src/prompts.py` |
| `src/modules/extraction_agent.py` (`extract_information_direct`, `summarize_answer`) | `src/extraction.py` |
| `src/utils/process.py` (`chunk_str`, `extract_json_dict`) | `src/extraction.py` |
| `src/models/llm_def.py` (`ChatGPT`/`DeepSeek`/`LocalServer` OpenAI-compatible engines) | `src/llm.py` (`ApiLLMEngine`) |
| `src/models/llm_def.py` (`Qwen`/`LLaMA`/etc. local `transformers` engines) | `src/llm.py` (`LocalHFEngine`) |

A knowledge-base case repository or its reflection
agent can be created and they need a vector store , however it add a lot of moving parts for a demo
app so not included in the demo for now, but both of  two engine *flavours* are here:

- **Hosted API** (`ApiLLMEngine`) — talks to a small, OpenAI-compatible
  hosted model. Default: Groq's Llama-3.1-8B-Instant (genuinely small-scale,
  fast, generous free tier). Also supports OpenAI, DeepSeek, Together, or
  any custom OpenAI-compatible endpoint (e.g. your own vLLM/Ollama server).
- **Local model** (`LocalHFEngine`) — runs a small open-weight instruct
  model (Qwen2.5-0.5B/1.5B-Instruct or SmolLM2-1.7B-Instruct by default,
  or any HF model id you type in) **in-process on CPU** with
  `transformers`, no external API or key required — 
  local `Qwen`/`LLaMA` engine classes, just capped to sizes that are
  actually feasible without a GPU.

It can be Picked between them from the sidebar at runtime in local deployment.

## Project layout

```
app.py                 Streamlit UI
src/
  llm.py               ApiLLMEngine (hosted) + LocalHFEngine (in-process CPU) + provider presets
  prompts.py            Ported OneKE prompt templates & NER/RE/EE/Triple schemas
  extraction.py          Ported OneKE chunk -> extract -> summarize pipeline
  news.py                 RSS headline fetching + full-article text extraction
render.yaml              Render web service blueprint
requirements.txt          Core deps (always installed)
requirements-local.txt     Optional: torch + transformers, only for the "Local model" option
```



## Notes & limitations

- RSS feed URLs occasionally change; `src/news.py` lists the current known
  feeds and the sidebar lets you paste a replacement URL if one breaks. A
  feed that fails to parse shows an error without crashing the app.
- Some outlets (e.g. paywalled Telegraaf/AD/Volkskrant articles) may only
  yield the RSS summary rather than full body text — the app falls back to
  the summary automatically in that case.
- Long articles are chunked  and per-chunk results are
  merged with an LLM summarization pass, exactly as OneKE does.
