"""
Efficient nlp for ie from news (prototype) application: fetch news articles from major Dutch outlets and run
schema-guided information extraction (Named entity recognition / Relation Extraction / Event Extraction / open Triple
extraction) on them, using a small-scale LLM through prompts and schemas.
"""

from __future__ import annotations

import streamlit as st

from src.extraction import run_extraction
from src.llm import LOCAL_MODEL_PRESETS, PROVIDER_PRESETS, create_llm_engine
from src.news import NEWS_SOURCES, fetch_article_text, fetch_headlines
from src.prompts import TASK_LABELS

st.set_page_config(page_title="Dutch News IE with OneKE", page_icon="📰", layout="wide")

st.title("📰 Dutch News → Information Extraction")
st.caption(
    "Fetches articles from major Netherlands news outlets and runs schema-guided "
    "information extraction with a small-scale LLM (local or cloud based), using prompts and schemas.  "
)


@st.cache_resource(show_spinner=False)
def _load_local_engine(model_name: str):
    """Cached so the (slow) local model load only happens once per unique
    model id per server process, not on every extraction click."""
    from src.llm import create_llm_engine

    return create_llm_engine(mode="local", model_name=model_name)

# ------------------------------------------------------------------ #
#                              Sidebar                                #
# ------------------------------------------------------------------ #
with st.sidebar:
    st.header("⚙️ Model settings")

    llm_mode = st.radio(
        "How should extraction be powered?",
        options=["api", "local"],
        format_func=lambda m: "Hosted API (small model)" if m == "api" else "Local model (runs on this server, CPU)",
        horizontal=False,
    )

    provider_key = "api"
    model_name = ""
    api_key = ""
    base_url = None

    if llm_mode == "api":
        provider_key = st.selectbox(
            "Provider",
            options=list(PROVIDER_PRESETS.keys()),
            format_func=lambda k: PROVIDER_PRESETS[k].label,
            index=0,
        )
        preset = PROVIDER_PRESETS[provider_key]
        st.caption(preset.notes)

        model_name = st.text_input("Model name", value=preset.default_model)

        api_key = st.text_input(
            f"{preset.api_key_env}",
            type="password",
            help=f"Falls back to the {preset.api_key_env} environment variable if left blank "
            "(set this as a Render environment variable so you don't have to paste it here).",
        )

        base_url = preset.base_url
        if provider_key == "custom":
            base_url = st.text_input("Base URL", value=preset.base_url)
    else:
        st.caption(
            "Runs a small open-weight instruct model in-process with `transformers` — "
            "no API key needed, but slower and needs `torch`+`transformers` installed. "
        )
        local_choice = st.selectbox("Local model", options=list(LOCAL_MODEL_PRESETS.keys()))
        default_local_id = LOCAL_MODEL_PRESETS[local_choice]
        model_name = st.text_input("HF model id (editable)", value=default_local_id)
        st.warning(
            "First run downloads the model weights and can take a while. "
            "local server computer may not have enough RAM for anything above 0.5B–1.5B params.",
            icon="⏳",
        )

    st.divider()
    st.header("📰 News source")
    source_name = st.selectbox("Outlet / feed", options=list(NEWS_SOURCES.keys()))
    custom_feed = st.text_input("...or paste a custom RSS feed URL", value="")
    num_headlines = st.slider("Headlines to fetch", 5, 30, 12)

# ------------------------------------------------------------------ #
#                         Fetch headlines                             #
# ------------------------------------------------------------------ #
if "headlines" not in st.session_state:
    st.session_state.headlines = []
if "article_text" not in st.session_state:
    st.session_state.article_text = ""
if "article_title" not in st.session_state:
    st.session_state.article_title = ""

col_fetch, col_status = st.columns([1, 3])
with col_fetch:
    fetch_clicked = st.button("🔄 Fetch headlines", use_container_width=True)

if fetch_clicked:
    with st.spinner(f"Fetching headlines from {source_name}..."):
        try:
            st.session_state.headlines = fetch_headlines(
                source_name, feed_url=custom_feed or None, limit=num_headlines
            )
            st.session_state.article_text = ""
            st.session_state.article_title = ""
        except Exception as exc:
            st.session_state.headlines = []
            st.error(f"Couldn't fetch that feed: {exc}")

headlines = st.session_state.headlines

# ------------------------------------------------------------------ #
#                    Article selection + preview                     #
# ------------------------------------------------------------------ #
left, right = st.columns([1, 1.4])

with left:
    st.subheader("Headlines")
    if not headlines:
        st.info("Click **Fetch headlines** in the sidebar to load articles.")
    else:
        titles = [h.title for h in headlines]
        selected_idx = st.radio(
            "Select an article",
            options=range(len(headlines)),
            format_func=lambda i: titles[i],
            label_visibility="collapsed",
        )
        selected = headlines[selected_idx]
        st.markdown(f"**Source:** {selected.source}")
        if selected.published:
            st.markdown(f"**Published:** {selected.published}")
        st.markdown(f"[Open original article]({selected.link})")

        if st.button("📥 Load full article text", use_container_width=True):
            with st.spinner("Downloading and extracting article text..."):
                text = fetch_article_text(selected.link, fallback_summary=selected.summary)
                st.session_state.article_text = text
                st.session_state.article_title = selected.title

with right:
    st.subheader("Article text")
    manual_text = st.text_area(
        "Text used for extraction (auto-filled from the selected article; "
        "edit freely, or paste your own text)",
        value=st.session_state.article_text,
        height=320,
        key="article_text_box",
    )

# ------------------------------------------------------------------ #
#                     Extraction task selection                       #
# ------------------------------------------------------------------ #
st.divider()
st.subheader("🔍 Choose information extraction task(s)")

task_keys = list(TASK_LABELS.keys())
selected_tasks = st.multiselect(
    "Extraction type(s) — pick one or more",
    options=task_keys,
    format_func=lambda k: TASK_LABELS[k],
    default=["NER"],
)

run_clicked = st.button("▶️ Run extraction", type="primary")

if run_clicked:
    if not manual_text.strip():
        st.warning("Load or paste some article text first.")
    elif not selected_tasks:
        st.warning("Select at least one extraction task.")
    else:
        try:
            if llm_mode == "local":
                with st.spinner(f"Loading local model {model_name} (first run can take a few minutes)..."):
                    llm = _load_local_engine(model_name)
            else:
                llm = create_llm_engine(
                    mode="api",
                    provider=provider_key,
                    model_name=model_name,
                    api_key=api_key,
                    base_url=base_url,
                )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        tabs = st.tabs([TASK_LABELS[t] for t in selected_tasks])
        for tab, task in zip(tabs, selected_tasks):
            with tab:
                with st.spinner(f"Running {TASK_LABELS[task]} with {llm.model}..."):
                    result = run_extraction(llm, task, manual_text)
                if result.error:
                    st.error(f"Extraction failed: {result.error}")
                else:
                    st.caption(f"Processed {result.chunks_used} text chunk(s) with {llm.model}.")
                    if isinstance(result.result, (dict, list)):
                        st.json(result.result)
                    else:
                        # extraction.py always tries to return a dict, but if
                        # the model's output was unrecoverable, show it as
                        # plain text instead of risking a client-side JSON
                        # parse error from st.json on a non-JSON string.
                        st.warning("Model output wasn't valid JSON; showing raw response instead.")
                        st.code(str(result.result))

st.divider()
with st.expander("ℹ️ About this app"):
    st.markdown(
        """
This app uses a  **schema-guided "quick mode"** extraction logic. It supports two ways to run the model:
small **hosted** OpenAI-compatible APIs (Groq/OpenAI/DeepSeek/Together/
custom), or a genuinely small (≤1.7B parameter) open-weight model run
**locally** in-process on CPU via `transformers` — no external API key
needed,  own local `Qwen`/`LLaMA` engine classes.

**Extraction tasks** :
- **NER** — named entities and their types
- **RE** — relationship triples between named entities
- **EE** — events, their triggers, and arguments
- **Triple** — open subject–relation–object triple extraction

News is pulled from public RSS feeds of major Dutch outlets (NOS, NU.nl,
Telegraaf, Volkskrant, NRC, AD, RTL Nieuws, FD); full article text is
extracted with `trafilatura`.
        """
    )
