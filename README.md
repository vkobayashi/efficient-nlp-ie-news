# Dutch News → Information Extraction (OneKE-powered)

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

We don't port OneKE's knowledge-base case repository or its reflection
agent (they need a vector store and add a lot of moving parts for a demo
app), but both of OneKE's two engine *flavours* are here:

- **Hosted API** (`ApiLLMEngine`) — talks to a small, OpenAI-compatible
  hosted model. Default: Groq's Llama-3.1-8B-Instant (genuinely small-scale,
  fast, generous free tier). Also supports OpenAI, DeepSeek, Together, or
  any custom OpenAI-compatible endpoint (e.g. your own vLLM/Ollama server),
  mirroring OneKE's `ChatGPT`/`DeepSeek`/`LocalServer` classes.
- **Local model** (`LocalHFEngine`) — runs a small open-weight instruct
  model (Qwen2.5-0.5B/1.5B-Instruct or SmolLM2-1.7B-Instruct by default,
  or any HF model id you type in) **in-process on CPU** with
  `transformers`, no external API or key required — mirroring OneKE's own
  local `Qwen`/`LLaMA` engine classes, just capped to sizes that are
  actually feasible without a GPU.

Pick between them from the sidebar at runtime.

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

## Run locally

```bash
pip install -r requirements.txt
# Optional, only if you want the "Local model" option available:
pip install -r requirements-local.txt

export GROQ_API_KEY=your_key_here   # or set it in the sidebar at runtime, or skip entirely and use "Local model"
streamlit run app.py
```

Get a free Groq API key at <https://console.groq.com/keys>.

## Deploy to Render

1. Push this repo to GitHub.
2. In Render: **New +** → **Blueprint**, point it at the repo (it will read
   `render.yaml` automatically) — or **New +** → **Web Service** manually with:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
3. In the service's **Environment** tab, add `GROQ_API_KEY` (or
   `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` / `TOGETHER_API_KEY`, matching
   whichever provider you'll use) so users aren't required to paste a key
   into the sidebar themselves. Skip this entirely if you'll only use the
   local-model option.
4. Deploy. Render will give you a public `*.onrender.com` URL.

### Enabling the local-model option on Render

The default build only installs `requirements.txt` (fast, small slug,
works on the free plan). To also enable "Local model" in the sidebar:

1. Change the service's **Build Command** to:
   `pip install -r requirements.txt -r requirements-local.txt`
2. Use a plan with enough RAM — at least ~2GB free for the smallest
   (0.5B-parameter) model, more for the 1.5B/1.7B options. The free plan's
   512MB is generally too little.
3. Redeploy. The first extraction after each deploy will be slow while the
   model weights download and load into memory; subsequent runs reuse the
   cached, already-loaded model.

## Notes & limitations

- RSS feed URLs occasionally change; `src/news.py` lists the current known
  feeds and the sidebar lets you paste a replacement URL if one breaks. A
  feed that fails to parse shows an error without crashing the app.
- Some outlets (e.g. paywalled Telegraaf/AD/Volkskrant articles) may only
  yield the RSS summary rather than full body text — the app falls back to
  the summary automatically in that case.
- Long articles are chunked (OneKE's `chunk_str`) and per-chunk results are
  merged with an LLM summarization pass, exactly as OneKE does.
