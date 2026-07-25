# AI Resume Builder

[![Tests](https://github.com/gguillermomendoza/ai_resume_builder/actions/workflows/tests.yml/badge.svg?branch=work)](https://github.com/gguillermomendoza/ai_resume_builder/actions/workflows/tests.yml?query=branch%3Awork)

AI Resume Builder is a Gradio application that tailors a résumé to a job description and generates a matching cover letter, grounding candidate claims in evidence from the supplied résumé and optional public GitHub repository metadata while using a separate writing sample to guide the letter's style.

## Features

- Extracts the role's requirements from a pasted or fetched job description.
- Produces grounded résumé tailoring from résumé and optional GitHub evidence.
- Analyzes a writing sample for tone, formality, sentence style, and structure.
- Generates a grounded cover letter from candidate evidence.
- Reports which extracted requirements have supporting evidence and identifies coverage gaps.
- Checks tailored résumés and cover letters for claims that are not supported by the supplied sources.

See `pipeline.py` for the pipeline logic and `gradio_app.py` for the UI. The
`AI_Resume_Builder_v1.ipynb` / `AI_Resume_Builder_v2_Phase2.ipynb` notebooks
are the original, exploratory versions of this pipeline.

## Prerequisites

- Python 3.10+
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

## Setup

The project uses `pip` and the dependencies listed in `requirements.txt`.

### macOS and Linux

```bash
git clone https://github.com/gguillermomendoza/ai_resume_builder.git
cd ai_resume_builder
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and replace the placeholder with your Gemini API key. At startup,
`gradio_app.py` calls python-dotenv's `load_dotenv()` to load `.env`, then reads
the key with `os.environ.get("GOOGLE_API_KEY")`.

```bash
python gradio_app.py
```

### Windows PowerShell

```powershell
git clone https://github.com/gguillermomendoza/ai_resume_builder.git
cd ai_resume_builder
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder with your Gemini API key, then run:

```powershell
python gradio_app.py
```

The app is served locally by Gradio (normally at `http://localhost:7860`). If no
key was loaded from `.env`, the Configuration panel also accepts a key for the
current app process.

## Using the app

### Résumé tailoring

1. Open the **Resume Tailor** tab.
2. Paste a job description or provide its URL.
3. Paste or upload a résumé (`.pdf`, `.md`, or `.txt`).
4. Optionally enter a GitHub username to add public repository evidence and adjust the evidence count.
5. Click **Tailor my resume**, then review the tailored résumé, coverage, fabrication check, evidence, and diff tabs before downloading the Markdown output.

### Cover letter

1. Open the **Cover Letter** tab.
2. Paste a job description or provide its URL.
3. Paste or upload a résumé and a writing sample (`.pdf`, `.md`, or `.txt`).
4. Optionally enter a GitHub username, explicitly describe your motivation for the company or role, and choose the letter length.
5. Click **Generate cover letter**, then review the letter, coverage summary, style profile, retrieved evidence, and fact-check report before downloading the Markdown output.

The **Concise** option targets 250–350 words in 3–4 paragraphs. The **Standard**
option targets 450–600 words in 4–5 paragraphs.

## Cover Letter

The cover-letter workflow assigns a distinct role to each input:

- **Job description (JD):** defines what matters for the role and is used to extract requirements; it is not evidence that the candidate meets them.
- **Résumé and optional GitHub data:** provide the factual evidence used to support candidate claims.
- **Writing sample:** provides style guidance—such as tone, formality, and structure—only.

> **Warning:** The writing sample is not treated as factual evidence. Names, employers, achievements, dates, motivations, and other facts appearing only in that sample must not be used as claims about the candidate.

## Privacy

Job-description text, résumé text, writing-sample text, user-supplied motivation,
and fetched public GitHub repository metadata (repository names, descriptions,
and languages) leave your machine and are sent to the configured Gemini API as
part of model prompts. A job-posting URL is fetched from its remote website, and
a supplied GitHub username is sent to the public GitHub API to fetch repository
metadata. This application does not provide fully local or fully private
processing; review the relevant providers' data policies before submitting
sensitive content.

## Limitations

- Generated cover letters require user review and editing before use.
- Company- or role-specific motivation should be supplied explicitly by the user; the system should not invent it.
- Fact and fabrication checks can compare claims only with content that was uploaded, pasted, or fetched; they cannot independently verify those source facts.
- Scanned PDFs without selectable text are not supported. Convert them with OCR or provide `.md`/`.txt` content first.

## Running the tests

```bash
python -m pytest tests/ -q
```

Tests cover the pipeline and Gradio integration. They do not require a Gemini
API key; PDF-related tests may be skipped when their optional test dependency is
not installed.
