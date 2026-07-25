"""Offline tests for Gradio UI input-resolution helpers."""

from types import SimpleNamespace

import pytest

import gradio_app
from pipeline import PipelineError
from pipeline import CoverLetterPipelineResult


def test_writing_sample_uses_pasted_text_without_file():
    assert gradio_app._resolve_writing_sample(None, "  A pasted sample.  ") == "A pasted sample."


@pytest.mark.parametrize(
    ("suffix", "content"),
    [(".txt", "Plain text sample"), (".md", "# Markdown sample")],
)
def test_writing_sample_reads_utf8_text_upload(tmp_path, suffix, content):
    upload = tmp_path / f"sample{suffix}"
    upload.write_text(content, encoding="utf-8")

    assert gradio_app._resolve_writing_sample(SimpleNamespace(name=str(upload)), "") == content


def test_writing_sample_file_takes_precedence_over_paste(tmp_path):
    upload = tmp_path / "sample.txt"
    upload.write_text("File sample", encoding="utf-8")

    assert gradio_app._resolve_writing_sample(str(upload), "Pasted sample") == "File sample"


def test_writing_sample_empty_input_returns_empty_string():
    assert gradio_app._resolve_writing_sample(None, "") == ""
    assert gradio_app._resolve_writing_sample(None, None) == ""


def test_writing_sample_pdf_delegates_to_pipeline(tmp_path, monkeypatch):
    upload = tmp_path / "sample.pdf"
    upload.write_bytes(b"fake pdf bytes")
    seen = []

    def fake_extract(data):
        seen.append(data)
        return "Extracted PDF sample"

    monkeypatch.setattr(gradio_app.pipeline, "extract_text_from_pdf", fake_extract)

    assert gradio_app._resolve_writing_sample(str(upload), "") == "Extracted PDF sample"
    assert seen == [b"fake pdf bytes"]


def test_writing_sample_rejects_unsupported_upload(tmp_path):
    upload = tmp_path / "sample.docx"
    upload.write_bytes(b"not a supported writing sample")

    with pytest.raises(PipelineError, match=r"\.pdf, \.md, or \.txt"):
        gradio_app._resolve_writing_sample(str(upload), "fallback text")


def test_cover_letter_runs_pipeline_renders_profile_and_uses_unique_downloads(monkeypatch):
    monkeypatch.setattr(gradio_app, "_get_clients", lambda api_key: object())
    calls = []

    def fake_pipeline(**kwargs):
        calls.append(kwargs)
        yield ("warn", "Review this warning.", None)
        result = CoverLetterPipelineResult(
            cover_letter="# Dear Hiring Team\n\nA grounded letter.",
            coverage={"coverage_pct": 100, "covered": 1, "total": 1, "gaps": []},
            style_profile={"tone": "direct", "sentence_patterns": ["short", "active"]},
            retrieved_evidence={"Python": [{"source": "resume", "text": "Built a tool"}]},
            fact_check="No unsupported claims.",
            log=["Coverage: 100%"],
        )
        yield ("done", "Done", result)

    monkeypatch.setattr(gradio_app.pipeline, "run_cover_letter_pipeline", fake_pipeline)
    arguments = (
        "test-key", "Paste text", "Python required", "", None, "My resume",
        None, "My writing sample", "", "The mission matters", "Standard",
    )

    first = list(gradio_app.run_cover_letter(*arguments))[-1]
    second = list(gradio_app.run_cover_letter(*arguments))[-1]

    assert "Review this warning." in first[0]
    assert first[1].startswith("# Dear Hiring Team")
    assert "**Tone:** direct" in first[3]
    assert "**Sentence Patterns:** short, active" in first[3]
    assert first[7]["value"] != second[7]["value"]
    assert calls[0]["length_preference"] == "standard"


def test_cover_letter_validates_required_inputs_without_initializing_clients(monkeypatch):
    def unexpected(_api_key):
        raise AssertionError("clients should not be initialized")

    monkeypatch.setattr(gradio_app, "_get_clients", unexpected)
    monkeypatch.setattr(gradio_app, "_ENV_API_KEY", "")
    output = list(gradio_app.run_cover_letter(
        "", "Paste text", "", "", None, "", None, "", "", "", "Concise"
    ))[-1]

    assert "Google Gemini API key" in output[0]
    assert "job description" in output[0]
    assert "writing sample" in output[0]
