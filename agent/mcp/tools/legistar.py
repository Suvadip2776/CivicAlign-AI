"""Search NYC Council legislation.

The original version called the Legistar Web API (webapi.legistar.com), which
for the "nyc" client requires a token issued by Granicus — without one every
request comes back 403. This clone uses NYC Open Data instead: the Council
publishes the same legislation as the public dataset "City Council Legislation:
Bills and Local Laws" (6ctv-n46c), no token and no VPN required.
"""

import os

import requests

# Socrata dataset: https://data.cityofnewyork.us/d/6ctv-n46c
OPEN_DATA_URL = "https://data.cityofnewyork.us/resource/6ctv-n46c.json"
LEGISLATION_DETAIL_URL = "https://legistar.council.nyc.gov/LegislationDetail.aspx?ID="

# Optional. Socrata serves anonymous requests with a shared rate limit; an app
# token (free, https://data.cityofnewyork.us/profile/edit/developer_settings)
# raises it. Nothing breaks without one.
SOCRATA_APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN", "")

# (connect, read)
REQUEST_TIMEOUT = (5, 20)


def _headers() -> dict:
    return {"X-App-Token": SOCRATA_APP_TOKEN} if SOCRATA_APP_TOKEN else {}


def _escape(value: str) -> str:
    """Single quotes terminate a SoQL string literal; doubling them escapes."""
    return value.replace("'", "''")


def search_legislation(query: str, max_results: int = 5) -> str:
    """Search NYC Council legislation (bills, resolutions, intros) by keyword."""
    term = _escape(query.strip())
    if not term:
        return "Legistar search error: empty query."

    # Match the plain-language name, the formal legal title, or the summary, so
    # "artificial grass" finds the bill whether or not the phrase is in its title.
    params = {
        "$where": (
            f"upper(name) like upper('%{term}%') "
            f"OR upper(title) like upper('%{term}%') "
            f"OR upper(summary) like upper('%{term}%')"
        ),
        "$order": "intro_date DESC",
        "$limit": max_results,
    }

    try:
        response = requests.get(
            OPEN_DATA_URL, params=params, headers=_headers(), timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        matters = response.json()
    except Exception as e:
        return f"Legislation search error: {e}"

    if not matters:
        return f"No legislation found matching: {query}"

    formatted = []
    for m in matters:
        file_num = m.get("file_num", "N/A")
        name = m.get("name") or m.get("title", "Untitled")
        status = m.get("status", "Unknown status")
        intro_date = (m.get("intro_date") or "Unknown date")[:10]
        sponsor = m.get("primary_sponsor", "")
        committee = m.get("committee", "")
        law_number = m.get("law_number", "")
        matter_id = m.get("matter_id")

        line = f"{file_num}: {name}\nStatus: {status} | Introduced: {intro_date}"
        if sponsor and sponsor != "NA":
            line += f" | Primary sponsor: {sponsor}"
        if committee and committee != "NA":
            line += f"\nCommittee: {committee}"
        if law_number and law_number != "NA":
            line += f" | Enacted as: {law_number}"
        if matter_id:
            line += f"\n{LEGISLATION_DETAIL_URL}{matter_id}"
        formatted.append(line)

    return "\n\n".join(formatted)
