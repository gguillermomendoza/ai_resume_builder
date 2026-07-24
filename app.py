"""
AI Resume Builder — Streamlit UI.

Wraps the RAG pipeline (pipeline.py) in a form-based web app: drag-and-drop
resume upload, rendered markdown output, a coverage dashboard, live progress
states, and clean error handling for the 503 / GitHub-failure cases.

Run locally:      streamlit run app.py
Deploy:           Streamlit Community Cloud or Hugging Face Spaces (see requirements.txt)

The Gemini API key is read from (in order): the sidebar input, st.secrets
["GOOGLE_API_KEY"], or the GOOGLE_API_KEY environment variable.
"""

import os
import re

import streamlit as st

# Load GOOGLE_API_KEY (and any other vars) from a local .env if present.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import pipeline
from pipeline import PipelineError

st.set_page_config(
    page_title="AI Resume Builder",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached client init — heavy (loads a SentenceTransformer), so do it once.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models…")
def _init(api_key: str):
    return pipeline.init_clients(api_key)


def _get_api_key(sidebar_value: str) -> str:
    if sidebar_value:
        return sidebar_value.strip()
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GOOGLE_API_KEY", "")


TAG_RE = re.compile(r"\s*\[SOURCE:[^\]]*\]")


def _strip_tags(markdown: str) -> str:
    return TAG_RE.sub("", markdown)


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # A key from .env / secrets / env var makes the input box redundant, so only
    # show it when none is configured (e.g. a fresh clone or a cloud deploy).
    _preset_key = _get_api_key("")
    if _preset_key:
        api_key_input = ""
        st.success("✓ Gemini API key loaded from environment")
    else:
        api_key_input = st.text_input(
            "Google Gemini API key",
            type="password",
            help="Get one free at aistudio.google.com/apikey. "
            "Or set it via a .env file, Streamlit secrets, or the GOOGLE_API_KEY env var.",
            placeholder="AIza…",
        )
    github_username = st.text_input("GitHub username (optional)", placeholder="octocat")
    top_k = st.slider(
        "Evidence per requirement",
        min_value=1,
        max_value=5,
        value=2,
        help="How many resume/GitHub snippets to retrieve for each job requirement.",
    )
    st.divider()
    st.caption(
        "Grounded RAG: your tailored resume is built **only** from evidence found "
        "in your resume and GitHub — every bullet is source-tagged and fact-checked."
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📄 AI Resume Builder")
st.markdown(
    "Tailor your resume to a specific job — grounded in your real experience, "
    "with a coverage dashboard and a fabrication check so nothing gets invented."
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
left, right = st.columns(2, gap="large")

with left:
    st.subheader("1 · Job description")
    jd_mode = st.radio(
        "How do you want to provide the job description?",
        ["Paste text", "Fetch from URL"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if jd_mode == "Paste text":
        jd_text_input = st.text_area(
            "Paste the job description",
            height=260,
            placeholder="Paste the full job posting here…",
        )
        jd_url = ""
    else:
        jd_url = st.text_input("Job posting URL", placeholder="https://…")
        jd_text_input = ""
        st.caption("Some sites block scraping or need login — if that happens, paste the text instead.")

with right:
    st.subheader("2 · Your resume")
    uploaded = st.file_uploader(
        "Drag & drop your resume (.pdf, .md, or .txt)",
        type=["pdf", "md", "txt"],
        help="Or paste it below if you don't have a file handy.",
    )
    resume_paste = st.text_area(
        "…or paste your resume",
        height=180,
        placeholder="Paste your current resume here…",
    )

run = st.button("✨ Tailor my resume", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Resolve inputs
# ---------------------------------------------------------------------------
def _resolve_resume() -> str:
    if uploaded is not None:
        data = uploaded.getvalue()
        if uploaded.name.lower().endswith(".pdf"):
            try:
                return pipeline.extract_text_from_pdf(data)
            except PipelineError as e:
                st.error(str(e))
                return ""
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            st.error("Couldn't read that file as text. Upload a .pdf, .md, or .txt resume.")
            return ""
    return resume_paste.strip()


# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------
if run:
    api_key = _get_api_key(api_key_input)
    resume_text = _resolve_resume()

    # Validation
    errors = []
    if not api_key:
        errors.append("Add your Google Gemini API key in the sidebar.")
    if not resume_text:
        errors.append("Upload or paste your resume.")

    jd_text = jd_text_input.strip()
    if jd_mode == "Fetch from URL" and not jd_text:
        if not jd_url.strip():
            errors.append("Enter a job posting URL, or switch to pasting the text.")

    if errors:
        for e in errors:
            st.warning(e)
        st.stop()

    # Init clients (cached)
    try:
        _init(api_key)
    except PipelineError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Failed to initialize the models: {e}")
        st.stop()

    # Fetch JD from URL if needed
    if jd_mode == "Fetch from URL" and not jd_text:
        try:
            with st.spinner("Fetching the job posting…"):
                jd_text = pipeline.fetch_jd_from_url(jd_url.strip())
            st.caption(f"Fetched {len(jd_text)} characters from the URL.")
        except PipelineError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Couldn't fetch that URL ({e}). Paste the job description text instead.")
            st.stop()

    # Drive the pipeline generator with live progress
    result = None
    try:
        with st.status("Tailoring your resume…", expanded=True) as status:
            for event, message, payload in pipeline.run_pipeline(
                jd_text=jd_text,
                resume_text=resume_text,
                github_username=github_username.strip(),
                top_k=top_k,
            ):
                if event == "step":
                    status.write(f"⏳ {message}")
                elif event == "warn":
                    status.write(f"⚠️ {message}")
                elif event == "done":
                    result = payload
                    status.update(label="Done ✓", state="complete", expanded=False)
                elif event == "error":
                    status.update(label="Failed", state="error")
                    st.error(message)
                    st.stop()
    except PipelineError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Something went wrong while tailoring: {e}")
        st.stop()

    # Persist across reruns (e.g. when toggling tabs / download buttons)
    st.session_state["result"] = result


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
result = st.session_state.get("result")

if result is not None:
    st.divider()

    if result.injection_hits:
        st.warning(
            "⚠️ The job description contained possible prompt-injection phrasing "
            f"({', '.join(result.injection_hits)}). The output was generated with "
            "grounding rules, but review it carefully."
        )

    tab_resume, tab_dash, tab_check, tab_evidence, tab_diff, tab_log = st.tabs(
        ["📝 Tailored Resume", "📊 Coverage", "🔍 Fabrication Check", "🎯 Evidence", "↔️ Diff", "🪵 Run Log"]
    )

    # --- Tailored resume ---
    with tab_resume:
        show_tags = st.toggle("Show [SOURCE] tags", value=False)
        body = result.tailored_output if show_tags else _strip_tags(result.tailored_output)
        st.markdown(body)
        st.download_button(
            "⬇️ Download as Markdown",
            data=body,
            file_name="tailored_resume.md",
            mime="text/markdown",
        )

    # --- Coverage dashboard ---
    with tab_dash:
        cov = result.coverage
        c1, c2, c3 = st.columns(3)
        c1.metric("Requirement coverage", f"{cov.get('coverage_pct', 0)}%")
        c2.metric("Requirements matched", f"{cov.get('covered', 0)} / {cov.get('total', 0)}")
        c3.metric("Uncovered gaps", len(cov.get("gaps", [])))

        st.progress(min(1.0, cov.get("coverage_pct", 0) / 100))

        st.markdown("#### Per-requirement evidence")
        for req, evidence in result.retrieved_evidence.items():
            if evidence:
                with st.expander(f"✅ {req}", expanded=False):
                    for e in evidence:
                        badge = "🐙 GitHub" if e["source"] == "github" else "📄 Resume"
                        st.markdown(f"- **{badge}** — {e['text']}")
            else:
                st.markdown(f"❌ **{req}** — _no matching experience found (honest gap)_")

        if cov.get("gaps"):
            st.info(
                "Gaps are real — they mean your source material didn't cover that "
                "requirement. The tailored resume flags these honestly instead of inventing them."
            )

    # --- Fabrication check ---
    with tab_check:
        text = result.fabrication or ""
        if "no fabrications found" in text.lower():
            st.success("✅ No fabrications found — every claim traces back to your source material.")
        else:
            st.warning("⚠️ The fact-checker flagged claims to review:")
        st.markdown(text)

    # --- Evidence / retrieval ---
    with tab_evidence:
        st.caption("What the retriever pulled for each extracted job requirement.")
        for req, evidence in result.retrieved_evidence.items():
            st.markdown(f"**{req}**")
            if evidence:
                for e in evidence:
                    st.markdown(f"  - `[{e['source']}]` {e['text']}")
            else:
                st.markdown("  - _no evidence retrieved_")

    # --- Diff ---
    with tab_diff:
        st.caption("Line-level diff: original resume → tailored resume.")
        if result.diff.strip():
            st.code(result.diff, language="diff")
        else:
            st.info("No line-level differences to show.")

    # --- Run log ---
    with tab_log:
        for line in result.log:
            st.text(f"• {line}")

else:
    st.info("Fill in a job description and your resume, then click **Tailor my resume**.")
