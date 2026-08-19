"""Manual smoke check for a running local RAG API."""

from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.request

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_token = os.getenv("RAG_API_TOKEN")
    if not api_token:
        raise RuntimeError("RAG_API_TOKEN is not configured in .env")

    request = urllib.request.Request(
        "http://127.0.0.1:8000/chat",
        data=json.dumps(
            {
                "message": "What is the recommended sodium intake?",
                "top_k": 3,
            }
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-RAG-API-Token": api_token,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        print(response.read().decode("utf-8"))


if __name__ == "__main__":
    main()
