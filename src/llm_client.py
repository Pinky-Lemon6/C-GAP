"""Generic LLM wrapper for C-GAP.

Design goals:
- Use the official `openai` Python library.
- Allow custom `base_url` and `api_key` so we can switch between
  OpenAI / DeepSeek-compatible endpoints / local vLLM OpenAI-style servers.
- Provide a minimal `one_step_chat` call for the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import threading
import time
import itertools
from urllib.parse import urlparse, urlunparse


_REQUEST_COUNTER = itertools.count(1)


def _concurrency_debug_enabled() -> bool:
    v = os.getenv("CGAP_DEBUG_CONCURRENCY", "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _strip_inline_comment(value: str) -> str:
    """Strip inline comments in .env values.

    Example:
        https://example.com/v1  # comment  -> https://example.com/v1

    This is only used by the fallback parser (when python-dotenv is unavailable).
    """

    v = value.strip()
    if not v:
        return v
    if "#" not in v:
        return v
    # Respect quoted values.
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v.strip('"').strip("'").strip()
    # Remove inline comment starting with whitespace-#
    hash_pos = v.find("#")
    if hash_pos == -1:
        return v
    prefix = v[:hash_pos]
    # If there's no whitespace before '#', it could be part of a URL fragment.
    if prefix and not prefix[-1].isspace():
        return v
    return prefix.strip()


def _normalize_base_url(base_url: Optional[str]) -> Optional[str]:
    """Normalize OpenAI-compatible base_url.

    Common mistakes:
    - Providing a full endpoint like .../v1/chat/completions (should be .../v1)
    - Omitting /v1 entirely for OpenAI-compatible servers
    """

    if base_url is None:
        return None

    raw = str(base_url).strip()
    if not raw:
        return None

    # If someone pasted an endpoint, trim to the API root.
    for suffix in (
        "/chat/completions",
        "/v1/chat/completions",
        "/completions",
        "/v1/completions",
    ):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
            raw = raw.rstrip("/")
            break

    parsed = urlparse(raw)
    if parsed.scheme and parsed.netloc:
        path = parsed.path or ""

        # If no path provided, default to /v1 for OpenAI-compatible servers.
        if path in {"", "/"}:
            path = "/v1"

        # If user provided something like https://host/v1/, normalize trailing slash.
        if path.endswith("/") and path != "/":
            path = path[:-1]

        return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))

    # If it's not a normal URL (rare), return as-is.
    return raw


def _load_dotenv_if_present(dotenv_path: str | os.PathLike[str] = ".env") -> None:
    """Best-effort .env loader.

    Prefers python-dotenv if installed; otherwise parses simple KEY=VALUE lines.
    Does not override existing environment variables.
    """

    p = Path(dotenv_path)
    if not p.exists() or not p.is_file():
        return

    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=p, override=False)
        return
    except Exception:
        pass

    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_inline_comment(value)
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class LLMClientConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class LLMClient:
    """Thin wrapper around OpenAI-compatible chat completions."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        dotenv_path: str | os.PathLike[str] = ".env",
    ) -> None:
        _load_dotenv_if_present(dotenv_path)

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL")

        resolved_base_url = _normalize_base_url(resolved_base_url)

        self.api_key = resolved_api_key
        self.base_url = resolved_base_url

        try:
            from openai import OpenAI  # openai>=1

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self._mode = "openai_v1"
        except Exception as exc:
            raise ImportError(
                "openai library (v1+) is required. Install with: pip install openai"
            ) from exc

    def one_step_chat(
        self,
        messages: List[Dict[str, Any]],
        model_name: str,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> str:
        """Single chat completion call.

        Args:
            messages: OpenAI-style messages, e.g. [{"role":"user","content":"..."}]
            model_name: model identifier
            json_mode: if True, requests JSON object output when supported
            kwargs: forwarded to the underlying SDK call

        Returns:
            The assistant message content as a string.
        """

        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")

        request: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            **kwargs,
        }

        if json_mode:
            # Supported by OpenAI and many OpenAI-compatible servers.
            request["response_format"] = {"type": "json_object"}

        debug = _concurrency_debug_enabled()
        req_id = next(_REQUEST_COUNTER)
        if debug:
            now = time.time()
            th = threading.current_thread().name
            print(
                f"[CGAP_CONCURRENCY] t={now:.6f} thread={th} id={req_id} phase=begin model={model_name} json_mode={json_mode}",
                flush=True,
            )

        t0 = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(**request)
        except Exception as exc:
            # Improve diagnostics for the most common misconfiguration: wrong base_url.
            if exc.__class__.__name__ == "NotFoundError":
                raise RuntimeError(
                    "LLM request returned 404 Not Found. This usually means your base_url is wrong "
                    "(e.g., you set the full /v1/chat/completions endpoint instead of the API root), "
                    "or the model name does not exist on that provider. "
                    f"base_url={self.base_url!r}, model={model_name!r}"
                ) from exc
            raise
        finally:
            if debug:
                dt = time.perf_counter() - t0
                now = time.time()
                th = threading.current_thread().name
                print(
                    f"[CGAP_CONCURRENCY] t={now:.6f} thread={th} id={req_id} phase=end elapsed_s={dt:.3f}",
                    flush=True,
                )

        try:
            content = resp.choices[0].message.content
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Unexpected response format: {resp!r}") from exc

        # Debug: if content is empty, log the full response
        if not content:
            print(f"[LLMClient] WARNING: Empty content in response. Full response: {resp!r}")
            # Check for refusal or other issues
            if hasattr(resp.choices[0].message, 'refusal') and resp.choices[0].message.refusal:
                print(f"[LLMClient] Model refused: {resp.choices[0].message.refusal}")
            if hasattr(resp, 'usage'):
                print(f"[LLMClient] Token usage: {resp.usage}")

        return content or ""
