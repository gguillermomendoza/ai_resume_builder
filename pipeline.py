"""
Resume-tailoring RAG pipeline.

This is the V2 notebook (AI_Resume_Builder_v2_Phase2.ipynb) refactored into an
importable module. The underlying logic of each step is unchanged — the functions
were only parameterized so they no longer depend on Colab-specific globals
(`userdata`, `files.upload`, `input()`) and so they can be driven by a UI.

Public pipeline steps (same names / behavior as the notebook):
    fetch_github_repos, chunk_source_material, extract_jd_requirements,
    retrieve_relevant_experience, generate_tailored_resume,
    evaluate_rag_coverage, fabrication_check, screen_for_injection

`run_pipeline()` is a generator that runs the whole flow and yields progress
events so a UI can render loading/step states.
"""

from __future__ import annotations

import difflib
import re
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

# `requests` and `bs4` are imported lazily inside the two functions that use
# them, so the pure functions (chunking, eval, diff, injection screen) can be
# imported and tested without any third-party dependencies installed.

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------
# In the notebook these were module-level globals configured inline. Here they
# are set once by init_clients() so the same functions work under Streamlit.

_CLIENT = None
_EMBED = None
_CHROMA = None
_MODEL_NAME = "gemini-flash-latest"


class PipelineError(Exception):
    """Raised for user-surfaceable pipeline failures (bad key, model 503, etc.)."""


def init_clients(api_key: str, model_name: str = _MODEL_NAME):
    """Configure the Gemini client, embedding model, and Chroma client.

    Heavy to call (loads a SentenceTransformer), so callers should cache it
    (e.g. Streamlit's st.cache_resource). Returns the client bundle and also
    stores it in module globals so the notebook-style functions keep working.
    """
    global _CLIENT, _EMBED, _CHROMA, _MODEL_NAME

    if not api_key:
        raise PipelineError("No Google API key provided. Add your Gemini API key to continue.")

    from google import genai
    from sentence_transformers import SentenceTransformer
    import chromadb

    _CLIENT = genai.Client(api_key=api_key)
    _MODEL_NAME = model_name
    _EMBED = SentenceTransformer("all-MiniLM-L6-v2")
    _CHROMA = chromadb.Client()
    return {"client": _CLIENT, "embed": _EMBED, "chroma": _CHROMA}


def _require_clients():
    if _CLIENT is None or _EMBED is None or _CHROMA is None:
        raise PipelineError("Pipeline not initialized. Call init_clients(api_key) first.")


def _generate(prompt: str, retries: int = 3) -> str:
    """Wrap client.models.generate_content with retry + friendly errors (handles the 503 case)."""
    _require_clients()

    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = _CLIENT.models.generate_content(model=_MODEL_NAME, contents=prompt)
            text = getattr(resp, "text", None)
            if not text:
                raise PipelineError("The model returned an empty response. Try again.")
            return text
        except PipelineError:
            raise
        except Exception as e:  # google.genai.errors.ServerError/ClientError, etc.
            last_exc = e
            msg = str(e).lower()
            transient = "503" in msg or "unavailable" in msg or "overloaded" in msg or "429" in msg
            if transient and attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            break

    detail = str(last_exc) if last_exc else "unknown error"
    if "503" in detail or "unavailable" in detail.lower() or "overloaded" in detail.lower():
        raise PipelineError(
            "The Gemini model is temporarily overloaded (503). Please wait a moment and try again."
        ) from last_exc
    if "api key" in detail.lower() or "permission" in detail.lower() or "401" in detail:
        raise PipelineError("The API call was rejected — check that your Gemini API key is valid.") from last_exc
    raise PipelineError(f"Model call failed: {detail}") from last_exc


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def fetch_jd_from_url(url: str) -> str:
    """Scrape visible text from a job-posting URL (was inline in the notebook)."""
    resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    if len(text) < 200:
        raise PipelineError(
            "That page returned very little text — it may require login or JavaScript. "
            "Paste the job description text directly instead."
        )
    return text


def extract_text_from_pdf(data: bytes) -> str:
    """Extract selectable text from a PDF file's raw bytes.

    Raises PipelineError for unreadable PDFs and for scanned / image-only PDFs
    that contain no selectable text (so the UI can tell the user to paste
    instead of silently indexing nothing).
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover - depends on install
        raise PipelineError("PDF support requires the 'pypdf' package (pip install pypdf).") from e

    import io

    try:
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:
        raise PipelineError(
            f"Couldn't read that PDF ({e}). Try re-exporting it, or paste the text instead."
        ) from e

    if len(text.strip()) < 30:
        raise PipelineError(
            "This PDF has no selectable text — it may be a scan or image-only export. "
            "Paste your resume text instead."
        )
    return text


def fetch_github_repos(username: str) -> list[dict]:
    """Fetch public repo names, descriptions, and languages. (Unchanged logic.)"""
    if not username:
        return []
    r = requests.get(f"https://api.github.com/users/{username}/repos", timeout=10)
    r.raise_for_status()
    return [
        {"name": x["name"], "desc": x.get("description"), "lang": x.get("language")}
        for x in r.json()
    ]


# ---------------------------------------------------------------------------
# Grounding rules (shared system prompt) — unchanged from the notebook
# ---------------------------------------------------------------------------

system_rules = """
You are a resume tailoring assistant. Follow these rules strictly:

1. GROUNDING: Only use skills, experience, and projects that are explicitly present in
   the RESUME or GITHUB PROJECTS provided below. Do NOT invent projects, metrics,
   technologies, or experience that are not stated in the source material.
2. If the candidate lacks a skill/technology the job requires, do NOT fabricate exposure
   to it. Instead, note the gap in the "WHAT CHANGED AND WHY" section as an honest gap,
   or reframe genuinely transferable experience — never invent a new project or credential.
3. If GITHUB PROJECTS is empty, say so explicitly rather than working around it silently.
4. SOURCE TAGGING: after every bullet point in the tailored resume, add a tag showing
   where it came from: [SOURCE: RESUME], [SOURCE: GITHUB], or [SOURCE: REFRAMED] for
   language that reframes an existing point without adding new facts.
"""


# ---------------------------------------------------------------------------
# Step 1: chunk + index
# ---------------------------------------------------------------------------

_BULLET_RE = re.compile(r'^[ \t]*[-*][ \t]+', re.MULTILINE)
_HEADER_RE = re.compile(r'^[ \t]*#{1,6}[ \t]*', re.MULTILINE)
_BOLD_RE = re.compile(r'\*\*(.+?)\*\*')


def _clean_chunk_text(text: str) -> str:
    text = _HEADER_RE.sub('', text)
    text = _BOLD_RE.sub(r'\1', text)
    text = text.replace('*', '').replace('_', '')
    return ' '.join(text.split())  # collapse wrapped-line whitespace/newlines


def chunk_source_material(resume_text: str, repos: list[dict]) -> list[dict]:
    """Splits resume into bullet-level chunks and repos into one chunk each.

    A bullet's text can wrap across several physical lines, so chunks are built
    by splitting on bullet markers (- , * ) rather than assuming one line == one
    bullet. Markdown formatting (#, *, **) is stripped before the length check so
    a heading or bold marker doesn't distort what counts as a real chunk.
    """
    chunks = []
    bullets = list(_BULLET_RE.finditer(resume_text))

    if bullets:
        # Non-bullet text before the first bullet (name, title, summary lines)
        for line in resume_text[:bullets[0].start()].splitlines():
            text = _clean_chunk_text(line)
            if len(text) > 20:
                chunks.append({"text": text, "source": "resume"})
        # One chunk per bullet, spanning however many lines it wraps across
        for i, m in enumerate(bullets):
            end = bullets[i + 1].start() if i + 1 < len(bullets) else len(resume_text)
            text = _clean_chunk_text(resume_text[m.end():end])
            if len(text) > 20:
                chunks.append({"text": text, "source": "resume"})
    else:
        # No bullet markers at all -- fall back to per-line chunking
        for line in resume_text.splitlines():
            text = _clean_chunk_text(line)
            if len(text) > 20:
                chunks.append({"text": text, "source": "resume"})

    for repo in repos:
        text = f"{repo['name']}: {repo.get('desc') or ''} ({repo.get('lang') or 'unknown language'})"
        chunks.append({"text": text, "source": "github"})

    resume_chunk_count = sum(1 for c in chunks if c["source"] == "resume")
    if resume_chunk_count < 5:
        print(
            f"Warning: only {resume_chunk_count} resume chunk(s) found — this is suspiciously low. "
            "Chunking may have silently collapsed the resume into one block instead of "
            "per-bullet chunks. Check the resume's bullet formatting before proceeding."
        )

    return chunks


def build_index(chunks: list[dict]):
    """Embed chunks and load them into a fresh Chroma collection.

    Uses a fresh collection each run so re-runs don't accumulate stale chunks
    (in the notebook this was inline after chunk_source_material).
    """
    _require_clients()
    if not chunks:
        raise PipelineError("No usable content found in the resume. Paste a resume with more detail.")
    try:
        _CHROMA.delete_collection("resume_chunks")
    except Exception:
        pass
    collection = _CHROMA.get_or_create_collection("resume_chunks")
    embeddings = _EMBED.encode([c["text"] for c in chunks]).tolist()
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=embeddings,
        metadatas=chunks,
    )
    return collection


# ---------------------------------------------------------------------------
# Step 2: extract requirements + retrieve evidence
# ---------------------------------------------------------------------------

def extract_jd_requirements(jd_text: str) -> list[str]:
    """Tool: pulls a structured list of requirements out of the JD. (Unchanged logic.)"""
    prompt = f"""Extract the 6-10 most important skills/requirements from this job description.
Return ONLY a plain list, one requirement per line, no numbering or extra text.

JOB DESCRIPTION: {jd_text}"""
    text = _generate(prompt)
    return [r.strip("-* ") for r in text.splitlines() if r.strip()]


def retrieve_relevant_experience(requirements: list[str], collection, top_k: int = 2) -> dict:
    """Tool: for each requirement, retrieve the top-k matching source chunks. (Unchanged logic.)"""
    _require_clients()
    retrieved = {}
    for req in requirements:
        q_embedding = _EMBED.encode([req]).tolist()
        results = collection.query(query_embeddings=q_embedding, n_results=top_k)
        retrieved[req] = [
            {"text": m["text"], "source": m["source"]}
            for m in results["metadatas"][0]
        ]
    return retrieved


# ---------------------------------------------------------------------------
# Step 3: generate tailored resume
# ---------------------------------------------------------------------------

def generate_tailored_resume(requirements, retrieved_evidence, full_resume) -> str:
    """Generate the tailored resume from retrieved evidence only. (Unchanged logic.)"""
    evidence_block = "\n".join(
        f"- {req}: " + "; ".join(f"[{e['source']}] {e['text']}" for e in ev)
        for req, ev in retrieved_evidence.items()
    )
    prompt = f"""{system_rules}

You must build the tailored resume using ONLY the RETRIEVED EVIDENCE below plus the
FULL RESUME for formatting/contact info. If a JD requirement has no retrieved evidence,
say so honestly in "what changed and why" — do not invent a bridge.

JD REQUIREMENTS: {requirements}
RETRIEVED EVIDENCE: {evidence_block}
FULL RESUME (for formatting/contact info only): {full_resume}

Return: 1. TAILORED RESUME (markdown, [SOURCE: ...] tags)  2. WHAT CHANGED AND WHY
"""
    return _generate(prompt)


# ---------------------------------------------------------------------------
# Step 4: evaluate retrieval coverage
# ---------------------------------------------------------------------------

def evaluate_rag_coverage(requirements, retrieved_evidence) -> dict:
    """Simple eval: what % of JD requirements had retrieved evidence at all. (Unchanged logic.)"""
    covered = sum(1 for ev in retrieved_evidence.values() if ev)
    total = len(requirements) or 1
    coverage_pct = round(100 * covered / total, 1)
    gaps = [req for req, ev in retrieved_evidence.items() if not ev]
    return {"coverage_pct": coverage_pct, "covered": covered, "total": len(requirements), "gaps": gaps}


# ---------------------------------------------------------------------------
# Step 5: fabrication self-check
# ---------------------------------------------------------------------------

def fabrication_check(resume: str, repos: list[dict], tailored_output: str) -> str:
    """Second LLM call that fact-checks the tailored resume against sources. (Unchanged logic.)"""
    critique_prompt = f"""
You are a fact-checker. Compare the TAILORED RESUME below against the ORIGINAL RESUME
and GITHUB PROJECTS. Flag any claim, project, metric, or skill in the tailored version
that is NOT supported by the original sources. Be strict — reframing existing facts is
fine, inventing new ones is not.

ORIGINAL RESUME: {resume}
GITHUB PROJECTS: {repos if repos else "None provided."}
TAILORED RESUME: {tailored_output}

Return a bulleted list titled "FABRICATION CHECK" — one line per issue found,
quoting the unsupported claim. If nothing is unsupported, say "No fabrications found."
"""
    return _generate(critique_prompt)


def section_diff(original: str, tailored: str) -> str:
    """Line-level diff between the original and tailored resume. (Unchanged logic.)"""
    orig_lines = [l.strip() for l in original.splitlines() if l.strip()]
    tailored_lines = [l.strip() for l in tailored.splitlines() if l.strip()]
    diff = difflib.unified_diff(orig_lines, tailored_lines, lineterm="", n=0)
    return "\n".join(list(diff)[2:])  # skip the file-header lines


# ---------------------------------------------------------------------------
# Guardrail: prompt-injection screen — unchanged from the notebook
# ---------------------------------------------------------------------------

INJECTION_MARKERS = [
    "ignore previous instructions", "ignore all prior", "disregard the above",
    "you are now", "new instructions:", "system prompt:", "reveal your prompt",
]


def screen_for_injection(text: str) -> list[str]:
    """Return any injection markers found in the text (empty list = clean)."""
    lowered = (text or "").lower()
    return [m for m in INJECTION_MARKERS if m in lowered]


# ---------------------------------------------------------------------------
# Orchestrator — generator yielding progress events for the UI
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    repos: list = field(default_factory=list)
    chunks: list = field(default_factory=list)
    requirements: list = field(default_factory=list)
    retrieved_evidence: dict = field(default_factory=dict)
    tailored_output: str = ""
    coverage: dict = field(default_factory=dict)
    fabrication: str = ""
    diff: str = ""
    injection_hits: list = field(default_factory=list)
    log: list = field(default_factory=list)


def run_pipeline(
    jd_text: str,
    resume_text: str,
    github_username: str = "",
    top_k: int = 2,
) -> Iterator[tuple]:
    """Run the full pipeline, yielding (event, message, result) tuples.

    event is one of: "step" (in progress), "done" (final, result populated),
    "warn" (non-fatal), "error" (fatal — message is user-facing).
    The final yield is always ("done", ..., PipelineResult) unless an error
    is raised, in which case an ("error", ...) event is yielded and iteration ends.
    """
    result = PipelineResult()

    # Guardrail: screen the (possibly scraped) JD before it reaches any prompt.
    hits = screen_for_injection(jd_text)
    result.injection_hits = hits
    if hits:
        yield ("warn", f"Possible prompt-injection markers in the job description: {hits}", None)

    # GitHub (non-fatal on failure — the 503/fetch-failure requirement)
    yield ("step", "Fetching GitHub projects…", None)
    try:
        result.repos = fetch_github_repos(github_username) if github_username else []
        result.log.append(f"GitHub: {len(result.repos)} repos fetched")
    except Exception as e:
        result.repos = []
        result.log.append(f"GitHub fetch failed ({e}) — continuing without repo evidence")
        yield ("warn", f"Couldn't fetch GitHub repos for '{github_username}' — continuing without them.", None)

    # Chunk + index
    yield ("step", "Indexing your experience…", None)
    result.chunks = chunk_source_material(resume_text, result.repos)
    collection = build_index(result.chunks)
    gh = sum(1 for c in result.chunks if c["source"] == "github")
    result.log.append(f"Indexed {len(result.chunks)} chunks ({gh} from GitHub)")

    # Extract requirements
    yield ("step", "Reading the job description…", None)
    result.requirements = extract_jd_requirements(jd_text)
    result.log.append(f"Extracted {len(result.requirements)} JD requirements")

    # Retrieve evidence
    yield ("step", "Matching your experience to the role…", None)
    result.retrieved_evidence = retrieve_relevant_experience(result.requirements, collection, top_k)

    # Generate
    yield ("step", "Writing your tailored resume…", None)
    result.tailored_output = generate_tailored_resume(
        result.requirements, result.retrieved_evidence, resume_text
    )

    # Evaluate
    yield ("step", "Scoring requirement coverage…", None)
    result.coverage = evaluate_rag_coverage(result.requirements, result.retrieved_evidence)
    result.log.append(f"Coverage: {result.coverage['coverage_pct']}%")

    # Fabrication check
    yield ("step", "Fact-checking for fabrications…", None)
    result.fabrication = fabrication_check(resume_text, result.repos, result.tailored_output)

    # Diff
    result.diff = section_diff(resume_text, result.tailored_output)

    yield ("done", "Done", result)
