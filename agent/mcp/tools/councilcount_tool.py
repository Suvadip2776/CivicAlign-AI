"""Census/demographic data for NYC Council districts.

The original version POSTed to councilcount-llm, a Flask service on the council
network (INTERNAL_HOST:5000) that put an LLM in front of a SQL database. That host
is unreachable off-VPN, so this clone uses the public `councilcount` package the
NYC Council publishes on PyPI instead:

    https://pypi.org/project/councilcount/
    https://github.com/NewYorkCityCouncil/councilcount-py

It ships American Community Survey estimates as data files inside the package,
so lookups are local — no network call, no key, nothing to be off-network from.
The trade-off versus the internal service is that there is no LLM translating
the question into SQL, so matching a question to a variable is keyword-based
here (as the internal tool's own variable matching was) and the tool reports
which variable it picked so a wrong guess is visible rather than silent.

Install note: the package pins exact versions of requests, certifi, urllib3,
numpy and pandas. This project leaves those five unpinned so pip can resolve
them from councilcount, which keeps `pip install -r requirements.txt` working
on hosts that allow no extra pip flags.
"""

import re
from functools import lru_cache

# Estimates are published per 5-year ACS survey; "2023" is the 2019-2023 survey.
DEFAULT_ACS_YEAR = "2023"
GEO = "councildist"
BOUNDARY_YEAR = 2023
DISTRICT_COLUMN = f"{GEO}_{BOUNDARY_YEAR}"

_INSTALL_HINT = "councilcount is not installed. Run: pip install -r requirements.txt"

# Common ways people ask for the handful of variables that come up constantly.
# Anything not listed here still resolves through the keyword scoring below.
# Checked first: a specific measure always beats the generic "how many people"
# reading of the same sentence ("how many people are in poverty in district 17"
# is a poverty question, not a population one).
_SPECIFIC_ALIASES = {
    "poverty": "B06012_002E",
    "poor": "B06012_002E",
    "unemployment": "DP03_0005E",
    "unemployed": "DP03_0005E",
    # Negatives first: "lack health insurance" contains "health insurance", so
    # the positive variable would otherwise swallow the question.
    "uninsured": "DP03_0099E",
    "no health insurance": "DP03_0099E",
    "lack health insurance": "DP03_0099E",
    "lacks health insurance": "DP03_0099E",
    "without health insurance": "DP03_0099E",
    "health insurance": "DP03_0096E",
    "foreign born": "DP02_0094E",
    "immigrant": "DP02_0094E",
    "veteran": "DP02_0070E",
}

# Fallback: a plain head-count question with no other measure in it.
_GENERIC_ALIASES = {
    "population": "B01001_001E",
    "how many people": "B01001_001E",
    "people": "B01001_001E",
    "residents": "B01001_001E",
}

# Asking for a share rather than a count. Every variable comes back with a
# companion "<code>PE" percentage column, so this decides which one to read.
_PERCENT_WORDS = ("rate", "percent", "percentage", "share", "proportion", "%")

# Words that carry no signal when scoring a question against a variable name.
_STOPWORDS = {
    "the", "a", "an", "of", "in", "for", "is", "are", "what", "whats", "how",
    "many", "much", "there", "council", "district", "districts", "nyc", "new",
    "york", "city", "and", "to", "me", "tell", "give", "show", "data", "number",
    "total", "percent", "percentage", "rate", "please", "that", "this", "with",
}


def _lazy_import():
    import councilcount  # noqa: PLC0415 - optional dependency, imported on use
    return councilcount


@lru_cache(maxsize=4)
def _dictionary(acs_year: str):
    """The data dictionary: every variable code and its plain-English description."""
    return _lazy_import().get_available_councilcount_codes(acs_year)


@lru_cache(maxsize=32)
def _estimates(acs_year: str, code: str):
    """Values for one variable across all 51 council districts."""
    return _lazy_import().get_councilcount_estimates(
        acs_year, GEO, var_codes=[code], boundary_year=BOUNDARY_YEAR
    )


def _districts_in(query: str) -> list[int]:
    """Pull council district numbers out of a question, in the order asked."""
    found = []
    for match in re.finditer(r"\b(?:district|cd|d)\s*#?\s*(\d{1,2})\b", query, re.I):
        number = int(match.group(1))
        if 1 <= number <= 51 and number not in found:
            found.append(number)
    return found


def _pick_variable(query: str, acs_year: str):
    """Return (code, description) for the variable a question is asking about."""
    lowered = query.lower()

    for aliases in (_SPECIFIC_ALIASES, _GENERIC_ALIASES):
        for phrase, code in aliases.items():
            if phrase in lowered:
                table = _dictionary(acs_year)
                row = table[table.estimate_var_code == code]
                description = row.iloc[0].estimate_description if len(row) else phrase
                return code, description

    words = {w for w in re.findall(r"[a-z]+", lowered) if w not in _STOPWORDS and len(w) > 2}
    if not words:
        return None, None

    table = _dictionary(acs_year)
    # "...PE" entries are derived percentages that the estimates call refuses as
    # inputs; they arrive as columns alongside their base variable instead.
    table = table[~table.estimate_var_code.str.endswith("PE")]

    best, best_score = None, 0
    for _, row in table.iterrows():
        description = str(row.estimate_description).lower()
        score = sum(1 for w in words if w in description)
        if score > best_score:
            best, best_score = row, score

    if best is None or best_score == 0:
        return None, None
    return best.estimate_var_code, best.estimate_description


def _wants_percentage(query: str) -> bool:
    lowered = query.lower()
    return any(word in lowered for word in _PERCENT_WORDS)


def _format_value(value: float, as_percent: bool) -> str:
    if value is None or value != value:  # None or NaN
        return "not available"
    if as_percent:
        return f"{value:.1f}%"
    return f"{value:,.0f}"


def council_count(query: str) -> str:
    """Answer a demographic/population question for NYC Council districts."""
    try:
        _lazy_import()
    except ImportError:
        return f"councilcount error: {_INSTALL_HINT}"

    acs_year = DEFAULT_ACS_YEAR

    try:
        code, description = _pick_variable(query, acs_year)
    except Exception as e:
        return f"councilcount error: could not read the variable dictionary: {e}"

    if not code:
        return (
            "councilcount: could not match that question to an ACS variable. "
            "Try wording it with a concrete measure — population, poverty, "
            "unemployment, health insurance, foreign born, veterans — and a "
            "council district number."
        )

    try:
        frame = _estimates(acs_year, code)
    except Exception as e:
        return f"councilcount error: {e}"

    if DISTRICT_COLUMN not in frame.columns or code not in frame.columns:
        return f"councilcount error: unexpected data shape for {code}."

    districts = _districts_in(query)
    survey = f"{int(acs_year) - 4}-{acs_year} 5-year ACS"

    # Read the share instead of the count when the question asked for one.
    as_percent = _wants_percentage(query)
    percent_col = f"{code[:-1]}PE"
    value_col = percent_col if (as_percent and percent_col in frame.columns) else code
    as_percent = value_col.endswith("PE")
    label = f"{description} (share of the relevant population)" if as_percent else description

    if districts:
        lines = []
        for number in districts:
            row = frame[frame[DISTRICT_COLUMN].astype(str) == str(number)]
            if row.empty:
                lines.append(f"Council District {number}: no data")
                continue
            value = row.iloc[0][value_col]
            margin_col = f"{value_col[:-1]}M" if not as_percent else f"{value_col}M".replace("PEM", "PM")
            margin = row.iloc[0][margin_col] if margin_col in frame.columns else None
            line = f"Council District {number} — {label}: {_format_value(value, as_percent)}"
            if margin is not None and margin == margin:  # not NaN
                suffix = f"±{margin:.1f} points" if as_percent else f"±{margin:,.0f}"
                line += f" (margin of error {suffix})"
            lines.append(line)
        body = "\n".join(lines)
    elif as_percent:
        return (
            f"councilcount: '{description}' as a share varies by district and has no "
            "single citywide value here. Name a council district (e.g. 'district 31')."
        )
    else:
        citywide = frame[code].sum()
        body = (
            f"All 51 council districts combined — {description}: "
            f"{_format_value(citywide, False)}"
        )

    return (
        f"Data ({survey}, variable {code}):\n{body}\n"
        "Source: councilcount (NYC Council), https://github.com/NewYorkCityCouncil/councilcount-py"
    )
