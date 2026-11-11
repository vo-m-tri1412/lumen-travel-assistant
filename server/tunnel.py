"""Find a free Pinggy tunnel URL while ignoring dashboard and help links."""
import re
from urllib.parse import urlsplit


def extract_tunnel_url(line):
    line=re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line)
    for candidate in re.findall(r"https://[A-Za-z0-9.-]+", line):
        if (urlsplit(candidate).hostname or "").endswith(".pinggy.link"):
            return candidate
    return None
