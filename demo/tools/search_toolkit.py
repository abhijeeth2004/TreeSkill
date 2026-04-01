"""Search tool code generator used by demos.

Provides a shared cache-first search/fetch script pair so demos don't keep
duplicating inline script strings. The generated scripts are local command-line
helpers that can be mounted into each Kode workspace.

It supports:
- local cache retrieval from `search_cache.json`
- optional external fallback command for web search/fetch
"""

from __future__ import annotations

from textwrap import dedent


def build_search_web_script() -> str:
    """Build the local `search_web.py` helper script."""
    return dedent(
        """
        #!/usr/bin/env python3
        import argparse
        import json
        import os
        import re
        import subprocess
        from pathlib import Path

        CACHE_PATH = Path(__file__).resolve().parent / "search_cache.json"

        DEFAULT_TOP_K = 3


        def tokenize(text: str):
            if not text:
                return set()
            return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 1}


        def _parse_keywords(raw: str):
            raw = (raw or "").strip()
            if not raw:
                return []
            return [item.strip() for item in raw.split(",") if item.strip()]


        def score(item, query: str):
            q_tokens = tokenize(query)
            if not q_tokens:
                return 0

            corpus = " ".join(
                [
                    item.get("question", ""),
                    item.get("title", ""),
                    item.get("snippet", ""),
                    item.get("text", ""),
                    " ".join(_parse_keywords(item.get("keywords", ""))),
                ]
            )
            if not corpus:
                return 0

            item_tokens = tokenize(corpus)
            if not item_tokens:
                return 0
            return len(q_tokens & item_tokens)


        def _parse_external_json(text: str):
            text = (text or "").strip()
            if not text:
                return []

            def _extract_payload(raw_text: str):
                try:
                    return json.loads(raw_text)
                except Exception:
                    pass

                fenced = re.findall(
                    r"```(?:json)?\\s*(\\[[\\s\\S]*?\\]|\\{[\\s\\S]*?\\})\\s*```",
                    raw_text,
                    flags=re.IGNORECASE,
                )
                for block in fenced:
                    try:
                        return json.loads(block)
                    except Exception:
                        continue

                for start in ("[", "{"):
                    if start not in raw_text:
                        continue
                    idx = raw_text.find(start)
                    if idx < 0:
                        continue
                    candidate = raw_text[idx:]
                    for end in ("]", "}"):
                        end_idx = candidate.rfind(end)
                        if end_idx <= 0:
                            continue
                        chunk = candidate[: end_idx + 1]
                        try:
                            return json.loads(chunk)
                        except Exception:
                            continue
                return None

            payload = _extract_payload(text)
            if payload is None:
                return []

            if isinstance(payload, dict):
                if "result" in payload and isinstance(payload["result"], str):
                    inner = _parse_external_json(payload["result"])
                    if inner:
                        return inner
                payload = [payload]

            if not isinstance(payload, list):
                return []

            out = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url", "")).strip()
                if not url:
                    continue
                out.append(
                    {
                        "id": str(item.get("id", url)),
                        "topic": str(item.get("topic", "")),
                        "title": str(item.get("title", "")),
                        "url": url,
                        "snippet": str(item.get("snippet", "")),
                        "source": "web_fallback",
                    }
                )
            return out


        def _parse_external_lines(text: str):
            out = []
            urls: set[str] = set()
            url_re = re.compile(r"https?://\\S+")

            for line in text.splitlines():
                line = line.strip()
                matches = url_re.findall(line)
                if not matches:
                    continue
                for match in matches:
                    match = match.rstrip(")]}.,;:!?")
                    match = match.strip(chr(34) + chr(39))
                    if match in urls:
                        continue
                    urls.add(match)
                    out.append(
                        {
                            "id": match[:64],
                            "topic": "",
                            "title": "",
                            "url": match,
                            "snippet": "",
                            "source": "web_fallback",
                        }
                    )
            return out


        def _fallback_search_command(query: str, top_k: int):
            cmd = os.getenv("SEALQA_WEB_SEARCH_CMD", "").strip()
            if not cmd:
                return []

            if "{query}" in cmd:
                cmd = cmd.replace("{query}", query.replace('"', '\\"'))
            if "{top_k}" in cmd:
                cmd = cmd.replace("{top_k}", str(top_k))

            try:
                completed = subprocess.run(
                    cmd,
                    shell=True,
                    text=True,
                    check=False,
                    capture_output=True,
                    timeout=int(os.getenv("SEALQA_WEB_SEARCH_TIMEOUT", "10")),
                )
            except Exception:
                return []

            if completed.returncode != 0:
                return []

            parsed = _parse_external_json(completed.stdout)
            if parsed:
                return parsed[:top_k]
            return _parse_external_lines(completed.stdout)[:top_k]


        def _load_cache():
            if not CACHE_PATH.exists():
                return []
            try:
                return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            except Exception:
                return []


        def _search_cached(cache, query: str, top_k: int):
            scored = []
            for item in cache:
                scored.append((score(item, query), item))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            results = []
            for item_score, item in scored[:top_k]:
                if item_score <= 0:
                    break
                results.append(
                    {
                        "id": item.get("id", ""),
                        "topic": item.get("topic", ""),
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("snippet", ""),
                        "source": "cache",
                    }
                )
            if results:
                return results
            if os.getenv("SEALQA_ASO_ENABLE_WEB_FALLBACK", "0") != "1":
                return []
            return _fallback_search_command(query, top_k)


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--query", required=True)
            parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
            args = parser.parse_args()

            cache = _load_cache()
            results = _search_cached(cache, args.query, args.top_k)
            print(json.dumps(results, ensure_ascii=False, indent=2))


        if __name__ == "__main__":
            main()
        """
    ).strip() + "\n"


def build_fetch_script() -> str:
    """Build the local `fetch_url.py` helper script."""
    return dedent(
        """
        #!/usr/bin/env python3
        import argparse
        import json
        import os
        import subprocess
        from pathlib import Path

        CACHE_PATH = Path(__file__).resolve().parent / "search_cache.json"


        def _load_cache():
            if not CACHE_PATH.exists():
                return []
            try:
                return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            except Exception:
                return []


        def _fallback_fetch(url: str):
            cmd = os.getenv("SEALQA_WEB_FETCH_CMD", "").strip()
            if not cmd:
                return ""

            if "{url}" in cmd:
                cmd = cmd.replace("{url}", url.replace('"', '\\"'))

            try:
                completed = subprocess.run(
                    cmd,
                    shell=True,
                    text=True,
                    check=False,
                    capture_output=True,
                    timeout=int(os.getenv("SEALQA_WEB_FETCH_TIMEOUT", "10")),
                )
            except Exception:
                return ""

            if completed.returncode != 0:
                return ""
            return completed.stdout.strip()


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--url", required=True)
            args = parser.parse_args()

            refs = _load_cache()
            for item in refs:
                if item.get("url", "") == args.url:
                    print(item.get("text", ""))
                    return

            if os.getenv("SEALQA_ASO_ENABLE_WEB_FALLBACK", "0") != "1":
                return

            fallback = _fallback_fetch(args.url)
            if fallback:
                print(fallback)


        if __name__ == "__main__":
            main()
        """
    ).strip() + "\n"
