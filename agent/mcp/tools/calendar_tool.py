"""Upcoming NYC Council meetings and hearings.

The original version used the Legistar Web API's events endpoint, which needs a
Granicus-issued token for the "nyc" client. Two public sources were considered
for this clone:

  * NYC Open Data "City Council Meetings" (m48u-yjt8) — clean and tokenless, but
    it stops at 2024-12-19 and holds no future meetings, so it cannot answer
    "what is coming up".
  * Legistar's public InSite calendar page — the same calendar staff and the
    public read in a browser, and it does carry future meetings.

So this reads the public calendar page. It is HTML rather than an API, which
means the parsing below is tied to Legistar's table markup; if the page is
redesigned this returns "could not read", never wrong data.
"""

import html
import re
from datetime import date, datetime, timedelta

import requests

CALENDAR_URL = "https://legistar.council.nyc.gov/Calendar.aspx"
MEETING_DETAIL_URL = "https://legistar.council.nyc.gov/"

# (connect, read)
REQUEST_TIMEOUT = (5, 25)

_ROW_RE = re.compile(r'<tr[^>]*class="(?:rgRow|rgAltRow)"[^>]*>(.*?)</tr>', re.S)
_CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}\s*[AP]M$", re.I)
_DETAIL_RE = re.compile(r'href="([^"]*MeetingDetail\.aspx[^"]*)"', re.I)

# Rows carry these as link text; they are controls, not meeting information.
_NOISE = {"meeting details", "agenda", "minutes", "video", "not available", "captions"}


def _text(fragment: str) -> str:
    """Strip tags and collapse whitespace from one table cell."""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", fragment)).split())


def get_upcoming_meetings(days_ahead: int = 14) -> str:
    """List NYC Council meetings/hearings scheduled in the next N days."""
    try:
        days_ahead = max(1, min(int(days_ahead), 365))
    except (TypeError, ValueError):
        days_ahead = 14

    try:
        response = requests.get(
            CALENDAR_URL,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "NYCC-AI-Assistant/1.0"},
        )
        response.raise_for_status()
    except Exception as e:
        return f"Council calendar error: {e}"

    rows = _ROW_RE.findall(response.text)
    if not rows:
        return (
            "Council calendar error: could not read the meeting table on "
            f"{CALENDAR_URL} (the page layout may have changed)."
        )

    today = date.today()
    horizon = today + timedelta(days=days_ahead)

    meetings = []
    for row in rows:
        cells = [_text(c) for c in _CELL_RE.findall(row)]
        cells = [c for c in cells if c and c.lower() not in _NOISE]
        if not cells:
            continue

        when = next((c for c in cells if _DATE_RE.match(c)), None)
        if not when:
            continue
        try:
            meeting_date = datetime.strptime(when, "%m/%d/%Y").date()
        except ValueError:
            continue
        if not (today <= meeting_date <= horizon):
            continue

        time_str = next((c for c in cells if _TIME_RE.match(c)), "")
        # Everything that is not the date, the time, or a bare day-of-week label:
        # the first such cell is the body name, the last is usually the location.
        rest = [c for c in cells if c != when and c != time_str and len(c) > 3]
        name = rest[0] if rest else "NYC Council meeting"
        location = rest[-1] if len(rest) > 1 else ""

        link = _DETAIL_RE.search(row)
        url = MEETING_DETAIL_URL + html.unescape(link.group(1)) if link else ""

        meetings.append((meeting_date, name, time_str, location, url))

    if not meetings:
        return f"No NYC Council meetings found on the public calendar in the next {days_ahead} days."

    meetings.sort(key=lambda m: m[0])

    lines = []
    for meeting_date, name, time_str, location, url in meetings:
        line = f"{meeting_date.strftime('%a %b %d, %Y')}"
        if time_str:
            line += f" at {time_str}"
        line += f" — {name}"
        if location:
            line += f"\nLocation: {location}"
        if url:
            line += f"\n{url}"
        lines.append(line)

    header = f"{len(meetings)} NYC Council meeting(s) in the next {days_ahead} days:"
    return header + "\n\n" + "\n\n".join(lines)
