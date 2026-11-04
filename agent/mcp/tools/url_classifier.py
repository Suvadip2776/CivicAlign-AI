from urllib.parse import urlparse

import yaml

from .resources import RESOURCE_FILE


def classify_url(url: str) -> str:
    """Match a URL against known NYC Council resources (url_resources.yaml) by domain."""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = parsed.netloc.lower().removeprefix("www.")

    if not host:
        return f"Could not parse a valid URL from: {url}"

    with open(RESOURCE_FILE, "r") as f:
        data = yaml.safe_load(f)

    for resource in data["resources"]:
        resource_host = urlparse(resource.get("url", "")).netloc.lower().removeprefix("www.")
        if resource_host and (host == resource_host or host.endswith(f".{resource_host}")):
            name = resource.get("name", "")
            description = resource.get("description", "No description provided.")
            return (
                f"Matched internal resource: {name}\n"
                f"Description: {description}\n"
                f"URL: {resource.get('url', '')}"
            )

    return (
        f"No known internal NYC Council resource matches this URL ({host}). "
        "This appears to be an external site — a web_search may be more useful."
    )
