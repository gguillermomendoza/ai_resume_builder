"""Offline tests for Gradio UI input-resolution helpers."""

from types import SimpleNamespace

import pytest

import gradio_app
from pipeline import PipelineError


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
