"""
AI Resume Builder — Gradio UI.

Wraps the RAG pipeline (pipeline.py) in a form-based web app: resume upload,
rendered markdown output, a coverage dashboard, live progress, and error
handling for the 503 / GitHub-failure cases.

Run locally:      python gradio_app.py
Deploy:           Hugging Face Spaces (see requirements.txt)

The Gemini API key is read from (in order): the form input, or the
GOOGLE_API_KEY environment variable (including a local .env file).
"""

from __future__ import annotations

import os
import re

import gradio as gr

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import pipeline
from pipeline import PipelineError

TAG_RE = re.compile(r"\s*\[SOURCE:[^\]]*\]")

_ENV_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# Cache the initialized clients per API key so repeated runs don't reload the
# embedding model every time.
_client_cache: dict[str, object] = {}


def _strip_tags(markdown: str) -> str:
    return TAG_RE.sub("", markdown)


def _get_clients(api_key: str):
    if api_key not in _client_cache:
        _client_cache.clear()
        _client_cache[api_key] = pipeline.init_clients(api_key)
    return _client_cache[api_key]


def _read_upload(upload) -> str:
    """Read a Gradio upload, delegating PDF parsing to the pipeline."""
    path = upload.name if hasattr(upload, "name") else upload
    if path.lower().endswith(".pdf"):
        with open(path, "rb") as f:
            return pipeline.extract_text_from_pdf(f.read())
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _resolve_resume(resume_file, resume_paste: str) -> str:
    if resume_file is not None:
        return _read_upload(resume_file)
    return (resume_paste or "").strip()


def _resolve_writing_sample(sample_file, sample_paste: str) -> str:
    """Resolve an optional PDF, Markdown, text, or pasted writing sample."""
    if sample_file is not None:
        path = sample_file.name if hasattr(sample_file, "name") else sample_file
        if not path.lower().endswith((".pdf", ".md", ".txt")):
            raise PipelineError("Writing samples must be a .pdf, .md, or .txt file.")
        return _read_upload(sample_file)
    return (sample_paste or "").strip()


def _coverage_markdown(cov: dict) -> str:
    pct = cov.get("coverage_pct", 0)
    covered = cov.get("covered", 0)
    total = cov.get("total", 0)
    gaps = cov.get("gaps", [])
    bar_filled = int(round(pct / 5))
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    lines = [
        f"**Requirement coverage:** {pct}%  `{bar}`",
        f"**Requirements matched:** {covered} / {total}",
        f"**Uncovered gaps:** {len(gaps)}",
    ]
    return "\n\n".join(lines)


def _evidence_markdown(retrieved_evidence: dict) -> str:
    parts = []
    for req, evidence in retrieved_evidence.items():
        if evidence:
            parts.append(f"**{req}**")
            for e in evidence:
                tag = "GitHub" if e["source"] == "github" else "Resume"
                parts.append(f"- ({tag}) {e['text']}")
        else:
            parts.append(f"**{req}** — no matching experience found (honest gap)")
        parts.append("")
    return "\n".join(parts) if parts else "No requirements extracted."


def _evidence_raw_markdown(retrieved_evidence: dict) -> str:
    parts = []
    for req, evidence in retrieved_evidence.items():
        parts.append(f"**{req}**")
        if evidence:
            for e in evidence:
                parts.append(f"- `[{e['source']}]` {e['text']}")
        else:
            parts.append("- _no evidence retrieved_")
        parts.append("")
    return "\n".join(parts) if parts else "No requirements extracted."


def run(
    api_key_input,
    jd_mode,
    jd_text_input,
    jd_url,
    resume_file,
    resume_paste,
    github_username,
    top_k,
    show_tags,
):
    api_key = (api_key_input or "").strip() or _ENV_API_KEY
    github_username = (github_username or "").strip()

    log_lines = []

    def status(msg):
        return "\n".join(f"- {line}" for line in (log_lines + [msg]))

    try:
        resume_text = _resolve_resume(resume_file, resume_paste)
    except PipelineError as e:
        yield str(e), "", "", "", "", "", "", gr.update(visible=False)
        return

    errors = []
    if not api_key:
        errors.append("Add your Google Gemini API key.")
    if not resume_text:
        errors.append("Upload or paste your resume.")

    jd_text = (jd_text_input or "").strip()
    if jd_mode == "Fetch from URL" and not jd_text and not (jd_url or "").strip():
        errors.append("Enter a job posting URL, or switch to pasting the text.")

    if errors:
        yield "\n".join(f"- {e}" for e in errors), "", "", "", "", "", "", gr.update(visible=False)
        return

    yield status("Loading models…"), "", "", "", "", "", "", gr.update(visible=False)
    try:
        _get_clients(api_key)
    except PipelineError as e:
        yield str(e), "", "", "", "", "", "", gr.update(visible=False)
        return
    except Exception as e:
        yield f"Failed to initialize the models: {e}", "", "", "", "", "", "", gr.update(visible=False)
        return

    if jd_mode == "Fetch from URL" and not jd_text:
        log_lines.append("Fetching the job posting…")
        yield status(""), "", "", "", "", "", "", gr.update(visible=False)
        try:
            jd_text = pipeline.fetch_jd_from_url(jd_url.strip())
            log_lines.append(f"Fetched {len(jd_text)} characters from the URL.")
        except PipelineError as e:
            yield str(e), "", "", "", "", "", "", gr.update(visible=False)
            return
        except Exception as e:
            yield (
                f"Couldn't fetch that URL ({e}). Paste the job description text instead.",
                "", "", "", "", "", "", gr.update(visible=False),
            )
            return

    result = None
    try:
        for event, message, payload in pipeline.run_pipeline(
            jd_text=jd_text,
            resume_text=resume_text,
            github_username=github_username,
            top_k=top_k,
        ):
            if event in ("step", "warn"):
                log_lines.append(message)
                yield status(""), "", "", "", "", "", "", gr.update(visible=False)
            elif event == "done":
                result = payload
            elif event == "error":
                yield message, "", "", "", "", "", "", gr.update(visible=False)
                return
    except PipelineError as e:
        yield str(e), "", "", "", "", "", "", gr.update(visible=False)
        return
    except Exception as e:
        yield f"Something went wrong while tailoring: {e}", "", "", "", "", "", "", gr.update(visible=False)
        return

    warning = ""
    if result.injection_hits:
        warning = (
            "Note: the job description contained possible prompt-injection phrasing "
            f"({', '.join(result.injection_hits)}). The output was generated with "
            "grounding rules, but review it carefully."
        )

    body = result.tailored_output if show_tags else _strip_tags(result.tailored_output)
    coverage_md = _coverage_markdown(result.coverage)
    evidence_md = _evidence_markdown(result.retrieved_evidence)
    fabrication_md = result.fabrication or "_No output._"
    evidence_raw_md = _evidence_raw_markdown(result.retrieved_evidence)
    diff_md = f"```diff\n{result.diff}\n```" if result.diff.strip() else "No line-level differences to show."
    log_md = "\n".join(f"- {line}" for line in result.log)

    final_status = "Done." + (f"\n\n{warning}" if warning else "")

    download_path = os.path.join(os.path.dirname(__file__), "tailored_resume.md")
    with open(download_path, "w", encoding="utf-8") as f:
        f.write(body)

    yield (
        final_status,
        body,
        f"{coverage_md}\n\n---\n\n#### Per-requirement evidence\n\n{evidence_md}",
        fabrication_md,
        evidence_raw_md,
        diff_md,
        log_md,
        gr.update(visible=True, value=download_path),
    )


with gr.Blocks(title="AI Resume Builder") as demo:
    gr.Markdown(
        "# AI Resume Builder\n"
        "Tailor your resume to a specific job, grounded in your real experience. "
        "Includes a requirement-coverage dashboard and a fabrication check so nothing gets invented."
    )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("Configuration", open=not bool(_ENV_API_KEY)):
                if _ENV_API_KEY:
                    gr.Markdown("Gemini API key loaded from environment.")
                api_key_input = gr.Textbox(
                    label="Google Gemini API key",
                    type="password",
                    placeholder="AIza…" if not _ENV_API_KEY else "(using environment key)",
                    info="Get one free at aistudio.google.com/apikey, or set GOOGLE_API_KEY / a .env file.",
                )
                github_username = gr.Textbox(label="GitHub username (optional)", placeholder="octocat")
                top_k = gr.Slider(
                    label="Evidence per requirement",
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1,
                    info="How many resume/GitHub snippets to retrieve for each job requirement.",
                )

            gr.Markdown("### Job description")
            jd_mode = gr.Radio(["Paste text", "Fetch from URL"], value="Paste text", label=None, show_label=False)
            jd_text_input = gr.Textbox(
                label="Paste the job description", lines=10, placeholder="Paste the full job posting here…"
            )
            jd_url = gr.Textbox(label="Job posting URL", placeholder="https://…", visible=False)

            gr.Markdown("### Resume")
            resume_file = gr.File(label="Upload resume (.pdf, .md, .txt)", file_types=[".pdf", ".md", ".txt"])
            resume_paste = gr.Textbox(label="…or paste your resume", lines=8, placeholder="Paste your current resume here…")

            show_tags = gr.Checkbox(label="Show [SOURCE] tags in output", value=False)
            run_btn = gr.Button("Tailor my resume", variant="primary")

        with gr.Column(scale=2):
            status_box = gr.Markdown(label="Status")
            with gr.Tabs():
                with gr.Tab("Tailored resume"):
                    resume_out = gr.Markdown()
                    download_file = gr.File(label="Download as Markdown", visible=False)
                with gr.Tab("Coverage"):
                    coverage_out = gr.Markdown()
                with gr.Tab("Fabrication check"):
                    fabrication_out = gr.Markdown()
                with gr.Tab("Evidence"):
                    evidence_out = gr.Markdown()
                with gr.Tab("Diff"):
                    diff_out = gr.Markdown()
                with gr.Tab("Run log"):
                    log_out = gr.Markdown()

    def _toggle_jd_mode(mode):
        return (
            gr.update(visible=mode == "Paste text"),
            gr.update(visible=mode == "Fetch from URL"),
        )

    jd_mode.change(_toggle_jd_mode, inputs=jd_mode, outputs=[jd_text_input, jd_url])

    run_event = run_btn.click(
        run,
        inputs=[
            api_key_input,
            jd_mode,
            jd_text_input,
            jd_url,
            resume_file,
            resume_paste,
            github_username,
            top_k,
            show_tags,
        ],
        outputs=[
            status_box,
            resume_out,
            coverage_out,
            fabrication_out,
            evidence_out,
            diff_out,
            log_out,
            download_file,
        ],
    )


if __name__ == "__main__":
    demo.queue().launch(theme=gr.themes.Soft())
