"""Retrieve the text of NYC Council legislation to answer a question from.

This replaces the internal `document_rag` tool, which queried a RAG service on
the council network (INTERNAL_HOST:8000) indexed over Council legal memos. Those
memos are not public, so there is nothing to proxy them with: this tool answers
from the public legislative record instead — the formal legal text and plain
summary of bills and local laws, from NYC Open Data (6ctv-n46c).

It is deliberately different from `legistar_search`, which returns a compact
list of matching bills (file number, status, sponsor) for "what bills exist
about X". This returns fewer results with much more text, for "what does the
legislation actually say about X".
"""

import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

# Same layering as the orchestrator: NYCC_AI/.env, then the project root above it.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

OPEN_DATA_URL = "https://data.cityofnewyork.us/resource/6ctv-n46c.json"
LEGISLATION_DETAIL_URL = "https://legistar.council.nyc.gov/LegislationDetail.aspx?ID="

SOCRATA_APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN", "")
REQUEST_TIMEOUT = (5, 20)

MAX_DOCUMENTS = 3
MAX_CHARS_PER_DOCUMENT = 900

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "is", "are", "was", "were", "what",
    "whats", "which", "who", "how", "does", "do", "did", "say", "says", "about",
    "tell", "me", "and", "or", "to", "from", "with", "that", "this", "these",
    "there", "any", "council", "nyc", "new", "york", "city", "law", "laws",
    "legislation", "bill", "bills", "memo", "memos", "document", "documents",
    "summarize", "summary", "explain", "regarding", "concerning",
}


def _headers() -> dict:
    return {"X-App-Token": SOCRATA_APP_TOKEN} if SOCRATA_APP_TOKEN else {}


def _escape(value: str) -> str:
    return value.replace("'", "''")


def _keywords(query: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", query.lower())
    seen, keep = set(), []
    for word in words:
        if word in _STOPWORDS or word in seen:
            continue
        seen.add(word)
        keep.append(word)
    return keep


def _fetch(where: str, limit: int) -> list:
    response = requests.get(
        OPEN_DATA_URL,
        params={"$where": where, "$order": "intro_date DESC", "$limit": limit},
        headers=_headers(),
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _like(field: str, term: str) -> str:
    return f"upper({field}) like upper('%{term}%')"


def _rank(rows: list, keywords: list[str]) -> list:
    """Keep rows that match a keyword as a whole word, best first.

    Socrata's `like` is a plain substring test, so a search for "rent" comes
    back with every bill containing "currently". Scoring on word boundaries
    here is what keeps those out of the answer.
    """
    scored = []
    for row in rows:
        haystack = " ".join(str(row.get(f, "")) for f in ("name", "title", "summary")).lower()
        score = sum(1 for k in keywords if re.search(rf"\b{re.escape(k)}", haystack))
        if score:
            scored.append((score, row))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [row for _, row in scored[:MAX_DOCUMENTS]]


def _condense(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def search_legislative_text(query: str) -> str:
    """Find the NYC Council legislation most relevant to a question and return its text."""
    query = (query or "").strip()
    if not query:
        return "Legislative text search error: empty query."

    keywords = _keywords(query)
    if not keywords:
        return "Legislative text search error: no searchable terms in that question."

    fields = ("name", "title", "summary")
    try:
        # 1. The whole phrase — the most precise match when it lands.
        matches = _fetch(" OR ".join(_like(f, _escape(query)) for f in fields), MAX_DOCUMENTS)

        # 2. Every keyword present (AND). Much tighter than OR: a bill about
        #    "rent stabilization" beats one that merely says "currently".
        if not matches:
            terms = keywords[:4]
            clause = " AND ".join(
                "(" + " OR ".join(_like(f, _escape(k)) for f in fields) + ")"
                for k in terms
            )
            matches = _rank(_fetch(clause, 30), keywords)

        # 3. Any keyword (OR), ranked by how many match as whole words.
        if not matches:
            clause = " OR ".join(_like(f, _escape(k)) for k in keywords[:6] for f in fields)
            matches = _rank(_fetch(clause, 50), keywords)
    except Exception as e:
        return f"Legislative text search error: {e}"

    if not matches:
        return (
            f"No NYC Council legislation found matching: {query}\n"
            "(This searches the public legislative record. Council legal memos are "
            "not published as open data and are not covered here.)"
        )

    blocks = []
    for row in matches:
        file_num = row.get("file_num", "N/A")
        name = _condense(row.get("name") or "Untitled", 200)
        status = row.get("status", "Unknown status")
        intro_date = (row.get("intro_date") or "")[:10]
        law_number = row.get("law_number", "")
        matter_id = row.get("matter_id")

        block = f"### {file_num} — {name}\nStatus: {status}"
        if intro_date:
            block += f" | Introduced: {intro_date}"
        if law_number and law_number != "NA":
            block += f" | Enacted as Local Law {law_number}"

        summary = _condense(row.get("summary", ""), MAX_CHARS_PER_DOCUMENT)
        if summary and summary != "NA":
            block += f"\n\nSummary: {summary}"

        legal_text = _condense(row.get("title", ""), MAX_CHARS_PER_DOCUMENT)
        if legal_text and legal_text != "NA" and legal_text != summary:
            block += f"\n\nLegal text: {legal_text}"

        if matter_id:
            block += f"\n\nFull text: {LEGISLATION_DETAIL_URL}{matter_id}"
        blocks.append(block)

    return (
        f"{len(blocks)} relevant item(s) from the public NYC Council legislative record:\n\n"
        + "\n\n---\n\n".join(blocks)
    )
