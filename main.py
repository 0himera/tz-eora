import os
import re
import ipaddress
from urllib.parse import urlparse
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
import uuid

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
# CORS disabled: same-origin only
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# OpenAI-compatible client for OpenRouter
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
SRCLIST_PATH = APP_DIR / "srclist.txt"

load_dotenv()  # Load environment variables from .env if present

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("eora")

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.getenv("OPENROUTER_MODEL", os.getenv("MODEL", "openai/gpt-4o-mini"))
# Normalize legacy/invalid names like *-oss
if MODEL.endswith("-oss"):
    MODEL = MODEL[: -4]

API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

if not API_KEY:
    logger.error("OPENROUTER_API_KEY (or OPENAI_API_KEY) is not set. Set it in environment or .env.")

@dataclass
class Doc:
    idx: int
    url: str
    title: str
    text: str

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    citations: List[Dict[str, str]]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan startup: проверка API-ключа и загрузка источников в _docs."""
    global _docs
    if not API_KEY:
        # Fail fast if API key is not configured
        raise RuntimeError("API key not configured. Set OPENROUTER_API_KEY in environment or .env")
    try:
        _docs = _load_docs()
        logger.info("Startup loaded %d documents from srclist.txt", (_docs and len(_docs) or 0))
    except Exception as e:
        logger.exception("Failed to load documents: %s", e)
    yield


app = FastAPI(title="Eora LLM Bot", version="0.1.0", lifespan=lifespan)

# CORS middleware removed: same-origin only. Configure reverse proxy or add CORS if needed.

# Serve static frontend
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_docs: List[Doc] = []
_stopwords = {
    # Russian minimal stopwords
    "и", "в", "на", "по", "с", "к", "до", "из", "за", "что", "как", "для", "мы", "они", "вы", "он", "она",
    "о", "об", "от", "не", "это", "тот", "эта", "эти", "или", "да", "нет", "же", "бы", "у", "при", "про",
    # English minimal
    "the", "is", "are", "a", "an", "of", "to", "in", "and", "for", "on", "with", "by", "at", "as",
}

_retail_terms = {"ритейл", "ритейлер", "ритейлеры", "магазин", "магазины", "retail", "e-commerce", "маркетплейс", "маркетплейсы"}
_brand_terms = {"magnit", "магнит", "kazanexpress", "lamoda", "dodo", "pizza", "dodo pizza"}


def _tokenize(text: str) -> List[str]:
    """Токенизация текста с учетом стоп-слов."""
    tokens = re.findall(r"[\wёЁ]+", text.lower(), flags=re.UNICODE)
    return [t for t in tokens if t not in _stopwords]


def _fetch_url(url: str) -> tuple[str, str]:
    """Загружает URL и возвращает (title, text): чистый текст извлекается из HTML.
    При ошибке логируем и возвращаем (url, "").
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; EoraBot/0.1; +https://eora.ru)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        logger.debug("Fetch URL: %s", url)
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        html = r.text
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else url
        # Remove scripts/styles
        for bad in soup(["script", "style", "noscript", "template"]):
            bad.decompose()
        text = soup.get_text(" ")
        text = re.sub(r"\s+", " ", text).strip()
        return title, text
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return url, ""


def _load_docs() -> List[Doc]:
    docs: List[Doc] = []
    if not SRCLIST_PATH.exists():
        logger.error("srclist not found at %s", SRCLIST_PATH)
        return docs
    urls = [line.strip() for line in SRCLIST_PATH.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip() and not line.strip().startswith("#")]
    for url in urls:
        if not _is_allowed_url(url):
            logger.warning("Skipping disallowed URL: %s", url)
            continue
        title, text = _fetch_url(url)
        # Keep text reasonably sized per doc to reduce prompt
        snippet_limit = 5000
        if len(text) > snippet_limit:
            text = text[:snippet_limit]
        docs.append(Doc(idx=len(docs) + 1, url=url, title=title, text=text))
    return docs


def _score_doc(query: str, doc: Doc) -> float:
    q_tokens = _tokenize(query)
    d_tokens = _tokenize(doc.text)
    if not q_tokens or not d_tokens:
        return 0.0
    q_set = set(q_tokens)
    # Simple term overlap
    overlap = sum(1 for t in d_tokens if t in q_set)
    score = overlap / (len(d_tokens) ** 0.5)
    # Heuristic: boost brand names for retail-like questions
    if any(t in _retail_terms for t in q_set):
        url_l = doc.url.lower()
        text_l = doc.text.lower()
        if any(b in url_l or b in text_l for b in _brand_terms):
            score *= 2.0
    return score


def _top_k(query: str, k: int = 3) -> List[Doc]:
    scored = [(doc, _score_doc(query, doc)) for doc in _docs]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, s in scored[:k] if s > 0]


def _is_allowed_url(url: str) -> bool:
    """Allow only http(s) and block localhost/private/reserved networks.
    This protects against SSRF if srclist.txt is ever user-influenced.
    """
    try:
        u = urlparse(url)
        if u.scheme not in ("http", "https"):
            return False
        host = u.hostname
        if not host:
            return False
        try:
            ip = ipaddress.ip_address(host)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
        except ValueError:
            # Not an IP, simple hostname checks
            if host in ("localhost", "localhost.localdomain"):
                return False
        return True
    except Exception:
        return False


def _build_context_rel(docs: List[Doc]) -> tuple[str, Dict[int, Doc]]:
    """Build context with relative indices [1..len(docs)] and return mapping.
    """
    lines = ["Источники (используйте метки [N] в ответе):"]
    rel_map: Dict[int, Doc] = {}
    for i, d in enumerate(docs, start=1):
        lines.append(f"[{i}] {d.title} — {d.url}")
        preview = (d.text[:700] + "…") if len(d.text) > 700 else d.text
        if preview:
            lines.append(f"Фрагмент: {preview}")
        rel_map[i] = d
    return "\n".join(lines), rel_map


def _call_llm(question: str, ctx: str) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Please install requirements.")
    client = OpenAI(api_key=API_KEY, base_url=OPENROUTER_BASE_URL)

    system = (
        "Ты ассистент компании Eora. Отвечай кратко и по делу. "
        "Используй только факты из предоставленных источников. "
        "Обязательно вставляй ссылки на источники в виде меток [N] сразу после утверждений, "
        "где N — номер источника из списка. Если не уверены — скажи об этом."
    )

    prompt = (
        f"Вопрос клиента: {question}\n\n{ctx}\n\n"
        "Сформируй связный ответ на русском языке, добавляя метки [N] сразу после соответствующих фактов."
    )

    try:
        logger.info("LLM call model=%s base=%s", MODEL, OPENROUTER_BASE_URL)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content  # type: ignore
        return content or ""
    except Exception as e:
        logger.exception("LLM request failed: %s", e)
        raise RuntimeError(f"LLM request failed: {e}")


def _extract_citation_indices(text: str, allowed: List[int]) -> List[int]:
    nums = re.findall(r"\[(\d+)\]", text)
    seen = []
    for n in nums:
        try:
            i = int(n)
            if i in allowed and i not in seen:
                seen.append(i)
        except Exception:
            pass
    return seen


@app.get("/")
async def root() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found. Build missing static/index.html")
    return FileResponse(str(index_path))


@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """Обрабатывает вопрос: выбирает топ‑источники, строит контекст с относительными
    метками [1..k], вызывает LLM, извлекает использованные метки и возвращает
    ответ и список цитат.
    """
    global _docs
    q = (req.question or "").strip()
    req_id = uuid.uuid4().hex[:8]
    logger.info("ASK %s question_len=%d", req_id, len(q))
    if len(q) > 2000:
        q = q[:2000]
    if not q:
        raise HTTPException(status_code=400, detail="question is required")
    if not _docs:
        # Try lazy load
        try:
            loaded = _load_docs()
            if loaded:
                _docs = loaded
        except Exception:
            logger.exception("ASK %s lazy load failed", req_id)
    if not _docs:
        raise HTTPException(status_code=500, detail="Sources not available")

    top = _top_k(q, k=3)
    if not top:
        # fallback: take first 2
        top = _docs[:2]

    # Build context with RELATIVE indices 1..len(top)
    ctx, rel_map = _build_context_rel(top)
    try:
        answer = _call_llm(q, ctx)
    except Exception as e:
        logger.exception("ASK %s LLM error: %s", req_id, e)
        # Generic detail to avoid leaking internal errors
        raise HTTPException(status_code=502, detail="Upstream model request failed")

    allowed_rel = list(rel_map.keys())
    used = _extract_citation_indices(answer, allowed_rel)
    logger.info("ASK %s used citations: %s (allowed=%s)", req_id, used, allowed_rel)

    # Build citation objects in the order used; if none detected, include top all
    used_set = used or allowed_rel
    citations = []
    for i in used_set:
        d = rel_map.get(i)
        if d:
            citations.append({"index": str(i), "url": d.url, "title": d.title})

    # Replace [N] in answer with clickable anchors for the frontend convenience (optional)
    # We will keep raw text; frontend can decorate.

    return AskResponse(answer=answer, citations=citations)


@app.get("/api/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "docs": str(len(_docs))}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
