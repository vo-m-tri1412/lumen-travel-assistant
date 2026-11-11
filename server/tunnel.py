"""Find a free Pinggy tunnel URL while ignoring dashboard and help links."""
import re
from urllib.parse import urlsplit


def valid_tunnel_url(url):
    parsed=urlsplit(url)
    host=parsed.hostname or ""
    return parsed.scheme=="https" and not parsed.username and not parsed.password and host.endswith((".pinggy.link", ".pinggy-free.link"))


def select_tunnel_url(payload):
    if not isinstance(payload,dict):return None
    return next((url for url in payload.get("urls",[]) if isinstance(url,str) and valid_tunnel_url(url)),None)


def extract_tunnel_url(line):
    line=re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line)
    for candidate in re.findall(r"https://[A-Za-z0-9.-]+", line):
        if valid_tunnel_url(candidate):
            return candidate
    return None
