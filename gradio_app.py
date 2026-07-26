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
import tempfile

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


def _style_profile_markdown(profile: dict) -> str:
    """Render the model's structured style analysis as readable Markdown."""
    if not profile:
        return "_No writing-style profile was produced._"

    lines = []
    for key, value in profile.items():
        heading = str(key).replace("_", " ").strip().title()
        if isinstance(value, (list, tuple)):
            rendered = ", ".join(str(item) for item in value) or "_None_"
        elif isinstance(value, dict):
            rendered = "; ".join(
                f"**{str(item_key).replace('_', ' ')}:** {item_value}"
                for item_key, item_value in value.items()
            ) or "_None_"
        else:
            rendered = str(value)
        lines.append(f"**{heading}:** {rendered}")
    return "\n\n".join(lines)


def run_cover_letter(
    api_key_input,
    jd_mode,
    jd_text_input,
    jd_url,
    resume_file,
    resume_paste,
    writing_sample_file,
    writing_sample_paste,
    github_username,
    user_motivation,
    length_selection,
):
    """Resolve cover-letter inputs and stream pipeline progress to Gradio."""
    api_key = (api_key_input or "").strip() or _ENV_API_KEY
    github_username = (github_username or "").strip()
    messages = []

    def outputs(status, result=None, download_path=None):
        if result is None:
            return status, "", "", "", "", "", "", gr.update(visible=False)
        return (
            status,
            result.cover_letter,
            _coverage_markdown(result.coverage),
            _style_profile_markdown(result.style_profile),
            _evidence_raw_markdown(result.retrieved_evidence),
            result.fact_check or "_No fact-check report was produced._",
            "\n".join(f"- {line}" for line in result.log) or "_No log entries._",
            gr.update(visible=True, value=download_path),
        )

    def live_status():
        return "\n".join(f"- {message}" for message in messages)

    try:
        resume_text = _resolve_resume(resume_file, resume_paste)
        writing_sample = _resolve_writing_sample(writing_sample_file, writing_sample_paste)
    except (PipelineError, OSError) as exc:
        yield outputs(str(exc))
        return

    jd_text = (jd_text_input or "").strip()
    errors = []
    if not api_key:
        errors.append("Add your Google Gemini API key.")
    if not resume_text:
        errors.append("Upload or paste your resume.")
    if not writing_sample:
        errors.append("Upload or paste a writing sample.")
    if jd_mode == "Fetch from URL" and not jd_text and not (jd_url or "").strip():
        errors.append("Enter a job posting URL, or switch to pasting the text.")
    elif jd_mode != "Fetch from URL" and not jd_text:
        errors.append("Paste the job description.")
    if errors:
        yield outputs("\n".join(f"- {error}" for error in errors))
        return

    messages.append("Loading models…")
    yield outputs(live_status())
    try:
        _get_clients(api_key)
        if jd_mode == "Fetch from URL" and not jd_text:
            messages.append("Fetching the job posting…")
            yield outputs(live_status())
            jd_text = pipeline.fetch_jd_from_url(jd_url.strip())

        result = None
        for event, message, payload in pipeline.run_cover_letter_pipeline(
            jd_text=jd_text,
            resume_text=resume_text,
            writing_sample=writing_sample,
            github_username=github_username,
            user_motivation=(user_motivation or "").strip(),
            length_preference="standard" if length_selection == "Standard" else "concise",
        ):
            if event in ("step", "warn"):
                messages.append(message)
                yield outputs(live_status())
            elif event == "error":
                messages.append(message)
                yield outputs(live_status())
                return
            elif event == "done":
                result = payload
        if result is None:
            yield outputs("The cover-letter pipeline ended without a result.")
            return
    except (PipelineError, OSError) as exc:
        yield outputs(str(exc))
        return
    except Exception as exc:
        yield outputs(f"Something went wrong while generating the cover letter: {exc}")
        return

    messages.append("Done.")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".md", prefix="cover-letter-", delete=False
    ) as download:
        download.write(result.cover_letter)
        download_path = download.name
    yield outputs(live_status(), result, download_path)


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


def run_both(
    api_key_input,
    jd_mode,
    jd_text_input,
    jd_url,
    resume_file,
    resume_paste,
    github_username,
    top_k,
    show_tags,
    writing_sample_file,
    writing_sample_paste,
    user_motivation,
    length_selection,
):
    resume_outputs = ("", "", "", "", "", "", "", gr.update(visible=False))
    cover_outputs = ("", "", "", "", "", "", "", gr.update(visible=False))

    for resume_outputs in run(
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
        yield resume_outputs + cover_outputs

    for cover_outputs in run_cover_letter(
        api_key_input,
        jd_mode,
        jd_text_input,
        jd_url,
        resume_file,
        resume_paste,
        writing_sample_file,
        writing_sample_paste,
        github_username,
        user_motivation,
        length_selection,
    ):
        yield resume_outputs + cover_outputs


with gr.Blocks(title="AI Resume Builder") as demo:
    gr.Markdown(
        "# AI Resume Builder\n"
        "Tailor your résumé or generate a grounded cover letter for a specific job."
    )
    gr.Markdown(
        "**Privacy:** Résumé, writing-sample, and job-description content may be sent "
        "to the configured Gemini API for processing."
    )
    workflow_type = gr.Radio(
        ["Tailor résumé", "Generate cover letter", "Both"],
        value="Tailor résumé",
        label="What would you like to create?",
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
                )
                github_username = gr.Textbox(label="GitHub username (optional)")
            gr.Markdown("### Job description")
            jd_mode = gr.Radio(
                ["Paste text", "Fetch from URL"], value="Paste text", show_label=False
            )
            jd_text_input = gr.Textbox(label="Paste the job description", lines=10)
            jd_url = gr.Textbox(label="Job posting URL", visible=False)
            gr.Markdown("### Résumé")
            gr.Markdown(
                "Used as source material for either workflow. It is only rewritten "
                "when résumé tailoring is selected."
            )
            resume_file = gr.File(
                label="Upload your current résumé",
                file_types=[".pdf", ".md", ".txt"],
            )
            resume_paste = gr.Textbox(
                label="Upload your current résumé",
                placeholder="…or paste your résumé",
                lines=8,
            )

            with gr.Group(visible=True) as resume_options:
                top_k = gr.Slider(
                    label="Evidence per requirement",
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1,
                )
                show_tags = gr.Checkbox(
                    label="Show [SOURCE] tags in output", value=False
                )

            with gr.Group(visible=False) as cover_options:
                gr.Markdown("### Cover-letter details")
                sample_file = gr.File(
                    label="Upload writing sample (.pdf, .md, .txt)",
                    file_types=[".pdf", ".md", ".txt"],
                )
                sample_paste = gr.Textbox(
                    label="…or paste your writing sample", lines=6
                )
                motivation = gr.Textbox(
                    label="Why are you interested in this company or role?", lines=3
                )
                length = gr.Radio(
                    ["Concise", "Standard"], value="Standard", label="Length"
                )
            run_btn = gr.Button("Tailor my résumé", variant="primary")

        with gr.Column(scale=2):
            with gr.Group(visible=True) as resume_results:
                status_box = gr.Markdown(label="Résumé status")
                with gr.Tabs():
                    with gr.Tab("Tailored résumé"):
                        resume_out = gr.Markdown()
                        download_file = gr.File(
                            label="Download as Markdown", visible=False
                        )
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

            with gr.Group(visible=False) as cover_results:
                cover_status = gr.Markdown(label="Cover-letter status / progress")
                with gr.Tabs():
                    with gr.Tab("Cover letter"):
                        cover_out = gr.Markdown()
                        cover_download = gr.File(
                            label="Download as Markdown", visible=False
                        )
                    with gr.Tab("Coverage summary"):
                        cover_coverage = gr.Markdown()
                    with gr.Tab("Writing-style profile"):
                        style_out = gr.Markdown()
                    with gr.Tab("Retrieved evidence"):
                        cover_evidence = gr.Markdown()
                    with gr.Tab("Fact-check report"):
                        fact_check = gr.Markdown()
                    with gr.Tab("Run log"):
                        cover_log = gr.Markdown()

    def _toggle_jd_mode(mode):
        return (
            gr.update(visible=mode == "Paste text"),
            gr.update(visible=mode == "Fetch from URL"),
        )

    def _select_workflow(workflow):
        includes_resume = workflow in ("Tailor résumé", "Both")
        includes_cover = workflow in ("Generate cover letter", "Both")
        button_labels = {
            "Tailor résumé": "Tailor my résumé",
            "Generate cover letter": "Generate cover letter",
            "Both": "Create both",
        }
        return (
            gr.update(visible=includes_resume),
            gr.update(visible=includes_cover),
            gr.update(visible=includes_resume),
            gr.update(visible=includes_cover),
            gr.update(value=button_labels[workflow]),
        )

    def _run_selected(workflow, *inputs):
        shared = inputs[:7]
        top_k_value, show_tags_value = inputs[7:9]
        cover_values = inputs[9:]
        if workflow == "Tailor résumé":
            for output in run(*shared, top_k_value, show_tags_value):
                yield output + ("", "", "", "", "", "", "", gr.update(visible=False))
        elif workflow == "Generate cover letter":
            for output in run_cover_letter(*shared, *cover_values):
                yield ("", "", "", "", "", "", "", gr.update(visible=False)) + output
        else:
            yield from run_both(
                *shared, top_k_value, show_tags_value, *cover_values
            )

    jd_mode.change(
        _toggle_jd_mode,
        inputs=jd_mode,
        outputs=[jd_text_input, jd_url],
    )
    workflow_type.change(
        _select_workflow,
        inputs=workflow_type,
        outputs=[
            resume_options,
            cover_options,
            resume_results,
            cover_results,
            run_btn,
        ],
    )
    run_btn.click(
        _run_selected,
        inputs=[
            workflow_type,
            api_key_input,
            jd_mode,
            jd_text_input,
            jd_url,
            resume_file,
            resume_paste,
            github_username,
            top_k,
            show_tags,
            sample_file,
            sample_paste,
            motivation,
            length,
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
            cover_status,
            cover_out,
            cover_coverage,
            style_out,
            cover_evidence,
            fact_check,
            cover_log,
            cover_download,
        ],
    )


if __name__ == "__main__":
    demo.queue().launch(theme=gr.themes.Soft())
