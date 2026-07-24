# AI Resume Builder

A Streamlit app that tailors your resume to a specific job description using a
grounded RAG pipeline (Gemini + chromadb): it extracts JD requirements,
retrieves matching evidence from your resume/GitHub, generates a tailored
resume with source tags, checks it for fabrications, and shows a diff and a
requirement-coverage dashboard.

See `pipeline.py` for the pipeline logic and `app.py` for the UI. The
`AI_Resume_Builder_v1.ipynb` / `AI_Resume_Builder_v2_Phase2.ipynb` notebooks
are the original, exploratory versions of this pipeline.

## Prerequisites

- Python 3.10+
- A free Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

## Setup

```bash
git clone https://github.com/gguillermomendoza/ai_resume_builder.git
cd ai_resume_builder

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Set your API key (pick one):

```bash
# Option A: one-off env var
export GOOGLE_API_KEY=your_key_here

# Option B: persist it in a .env file (auto-loaded by app.py)
echo "GOOGLE_API_KEY=your_key_here" > .env
```

Run the app:

```bash
streamlit run app.py
```


## Using the app

Once running, Streamlit opens `http://localhost:8501` in your browser:

1. Paste a job description (or a URL to fetch it from).
2. Paste or upload your resume (`.pdf`, `.md`, or `.txt`).
3. Optionally add a GitHub username to pull in project evidence.
4. Click **Tailor my resume**.

If you didn't set `GOOGLE_API_KEY` via env var or `.env`, the sidebar will
prompt you for it instead.

## Running the tests

```bash
python -m pytest tests/ -q
```

Tests cover the pure pipeline functions (chunking, coverage scoring, diffing,
injection screening) and don't require an API key. A few PDF-related tests
are skipped automatically if `reportlab` isn't installed.
