"""Read local configuration without putting credentials in source code."""

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    firebase_api_key: str
    ollama_base_url: str
    ollama_api_token: str = ""
    top_k: int = 3
    min_similarity: float = 0.35

    @classmethod
    def load(cls):
        load_dotenv(ROOT / ".env", override=False)
        url = os.getenv("OLLAMA_BASE_URL", "").strip().rstrip("/")
        if url:
            parsed = urlsplit(url)
            local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            if parsed.scheme != "https" and not (local and parsed.scheme == "http"):
                raise ValueError("OLLAMA_BASE_URL cần dùng HTTPS, hoặc HTTP trên localhost.")
            if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("OLLAMA_BASE_URL không hợp lệ.")
        try:
            top_k = int(os.getenv("RAG_TOP_K") or "3")
            minimum = float(os.getenv("RAG_MIN_SIMILARITY") or "0.35")
        except ValueError as error:
            raise ValueError("Cấu hình top-k hoặc ngưỡng tương đồng không hợp lệ.") from error
        if not 1 <= top_k <= 5 or not -1 <= minimum <= 1:
            raise ValueError("Top-k cần từ 1–5; ngưỡng tương đồng cần từ -1 đến 1.")
        return cls(os.getenv("FIREBASE_API_KEY", "").strip(), url,
                   os.getenv("OLLAMA_API_TOKEN", "").strip(), top_k, minimum)
