from pathlib import Path

import yaml

RESOURCE_FILE = Path(__file__).resolve().parent.parent.parent / "url_resources.yaml"


def lookup_resource(keyword: str) -> str:
    """Search NYC Council resources by keyword."""
    with open(RESOURCE_FILE, "r") as f:
        data = yaml.safe_load(f)

    matches = []

    for resource in data["resources"]:
        name = resource.get("name", "")
        url = resource.get("url", "")
        description = resource.get("description", "")
        keywords = resource.get("keywords", [])

        if (
            keyword.lower() in name.lower()
            or keyword.lower() in description.lower()
            or any(keyword.lower() in k.lower() for k in keywords)
        ):
            matches.append(f"{name}: {url}\nDescription: {description}")

    if not matches:
        return f"No resource found for keyword: {keyword}"

    return "\n\n".join(matches)
