"""LLM provider adapter with four backends and a disk cache.

The adapter exposes a uniform :class:`LLMProvider` API used by every agent.
It auto-selects a backend in this order:

1. **Gemini** via ``langchain-google-genai`` when ``GEMINI_API_KEY`` (or
   ``GOOGLE_API_KEY``) is set and the package is importable.
2. **OpenAI-compatible** server (LM Studio, vLLM, llama.cpp server, ...)
   when ``OPENAI_BASE_URL`` is set or LM Studio is reachable on
   ``http://localhost:1234/v1``.
3. **Ollama** local server when ``OLLAMA_HOST`` is set or a daemon is
   reachable on ``http://127.0.0.1:11434``.
4. **Rule-based** deterministic stub that raises :class:`LLMUnavailable`
   so callers fall back to their own deterministic logic.

Every backend respects an on-disk JSON cache keyed by
``hash(prompt, model, schema)`` so repeated runs are cheap and reproducible.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_PATH = PROJECT_ROOT / "data" / "llm_cache.sqlite"


class LLMUnavailable(RuntimeError):
    """Raised when no live LLM backend can serve a request."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    backend: str
    cached: bool = False
    latency_seconds: float = 0.0


@dataclass
class ProviderConfig:
    # "auto" | "gemini" | "openai_compatible" (alias "lmstudio") | "ollama" | "rule_based"
    backend: str = "auto"
    gemini_model: str = "gemini-1.5-flash"
    ollama_model: str = "llama3.2:3b"
    ollama_host: str = "http://127.0.0.1:11434"
    openai_base_url: str = "http://localhost:1234/v1"
    openai_model: str = ""  # empty -> auto-discover via /models
    openai_api_key: str = "lm-studio"  # LM Studio ignores the value but the header is required
    temperature: float = 0.0
    max_output_tokens: int = 1024
    request_timeout_seconds: float = 30.0
    cache_path: Path = field(default_factory=lambda: DEFAULT_CACHE_PATH)
    enable_cache: bool = True


# ---------------------------------------------------------------------------
# Cache (sqlite-backed; thread-safe)
# ---------------------------------------------------------------------------


class _ResponseCache:
    """Thread-safe SQLite cache for prompt -> response text."""

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS llm_cache (
            cache_key TEXT PRIMARY KEY,
            backend TEXT NOT NULL,
            response TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connect()

    def _connect(self) -> None:
        conn = sqlite3.connect(str(self.path), check_same_thread=False)
        conn.execute(self._SCHEMA)
        conn.commit()
        self._conn = conn

    @staticmethod
    def make_key(backend: str, model: str, prompt: str, schema: str) -> str:
        digest = hashlib.sha256()
        digest.update(backend.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(model.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(schema.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(prompt.encode("utf-8"))
        return digest.hexdigest()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT response FROM llm_cache WHERE cache_key = ?", (key,)
            )
            row = cur.fetchone()
        return row[0] if row else None

    def put(self, key: str, backend: str, response: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache(cache_key, backend, response, created_at) VALUES (?, ?, ?, ?)",
                (key, backend, response, time.time()),
            )
            self._conn.commit()


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """Abstract LLM backend that returns plain text completions."""

    name: str = "abstract"
    model: str = ""

    def __init__(self, config: ProviderConfig, cache: Optional[_ResponseCache] = None) -> None:
        self.config = config
        self._cache = cache

    def complete(self, prompt: str, *, schema_name: str = "freeform") -> LLMResponse:
        cache_key: Optional[str] = None
        if self._cache is not None:
            cache_key = self._cache.make_key(self.name, self.model, prompt, schema_name)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return LLMResponse(text=cached, backend=self.name, cached=True)
        start = time.perf_counter()
        text = self._complete_uncached(prompt)
        elapsed = time.perf_counter() - start
        if self._cache is not None and cache_key is not None:
            self._cache.put(cache_key, self.name, text)
        return LLMResponse(text=text, backend=self.name, cached=False, latency_seconds=elapsed)

    @abstractmethod
    def _complete_uncached(self, prompt: str) -> str:
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, config: ProviderConfig, cache: Optional[_ResponseCache] = None) -> None:
        super().__init__(config, cache)
        self.model = config.gemini_model
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise LLMUnavailable(f"langchain-google-genai not importable: {exc}") from exc
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise LLMUnavailable("GEMINI_API_KEY / GOOGLE_API_KEY not set")
        self._chat = ChatGoogleGenerativeAI(
            model=config.gemini_model,
            google_api_key=api_key,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            timeout=config.request_timeout_seconds,
        )

    def _complete_uncached(self, prompt: str) -> str:
        from langchain_core.messages import HumanMessage  # type: ignore

        result = self._chat.invoke([HumanMessage(content=prompt)])
        if hasattr(result, "content"):
            content = result.content
        else:  # pragma: no cover - defensive
            content = str(result)
        if isinstance(content, list):
            content = "".join(part.get("text", str(part)) if isinstance(part, dict) else str(part) for part in content)
        return str(content)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, config: ProviderConfig, cache: Optional[_ResponseCache] = None) -> None:
        super().__init__(config, cache)
        self.model = config.ollama_model
        self._host = os.environ.get("OLLAMA_HOST", config.ollama_host).rstrip("/")
        try:
            import urllib.request  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise LLMUnavailable(f"urllib not available: {exc}") from exc
        # Probe daemon to fail fast.
        if not self._probe():
            raise LLMUnavailable(f"Ollama daemon not reachable at {self._host}")

    def _probe(self) -> bool:
        import urllib.error
        import urllib.request

        try:
            req = urllib.request.Request(f"{self._host}/api/tags")
            with urllib.request.urlopen(req, timeout=2.0) as response:  # noqa: S310
                return response.status == 200
        except (urllib.error.URLError, ValueError, TimeoutError):
            return False

    def _complete_uncached(self, prompt: str) -> str:
        import urllib.request

        payload = json.dumps(
            {
                "model": self.config.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_output_tokens,
                },
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self._host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.request_timeout_seconds) as response:  # noqa: S310
            body = response.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"Ollama returned non-JSON body: {exc}") from exc
        return str(parsed.get("response", ""))


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI-compatible chat-completions backend.

    Works with LM Studio, vLLM, llama.cpp server, oobabooga's text-generation-webui,
    and similar gateways that implement the ``/v1/chat/completions`` schema.

    Configure via env:
      * ``OPENAI_BASE_URL``  e.g. ``http://localhost:1234/v1`` (LM Studio default)
      * ``OPENAI_MODEL``     model identifier; if empty, the provider queries
        ``GET /models`` and uses the first entry.
      * ``OPENAI_API_KEY``   any non-empty string (LM Studio ignores the value
        but the header must be present).
    """

    name = "openai_compatible"

    def __init__(self, config: ProviderConfig, cache: Optional[_ResponseCache] = None) -> None:
        super().__init__(config, cache)
        self._base_url = os.environ.get("OPENAI_BASE_URL", config.openai_base_url).rstrip("/")
        self._api_key = os.environ.get("OPENAI_API_KEY", config.openai_api_key) or "local"
        # Probe + (optionally) auto-discover model.
        models = self._list_models()
        if not models:
            raise LLMUnavailable(
                f"OpenAI-compatible server at {self._base_url} is not reachable or returned no models."
            )
        configured_model = (os.environ.get("OPENAI_MODEL") or config.openai_model or "").strip()
        if configured_model:
            if configured_model not in models:
                logger.info(
                    "OPENAI_MODEL=%s not in /models response %s; sending request anyway.",
                    configured_model,
                    models,
                )
            self.model = configured_model
        else:
            self.model = models[0]
            logger.info("OpenAI-compatible auto-selected model: %s (from %s)", self.model, self._base_url)

    def _list_models(self) -> list[str]:
        import urllib.error
        import urllib.request

        url = f"{self._base_url}/models"
        try:
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            with urllib.request.urlopen(req, timeout=2.5) as response:  # noqa: S310
                if response.status != 200:
                    return []
                body = response.read().decode("utf-8")
        except (urllib.error.URLError, ValueError, TimeoutError, ConnectionError):
            return []
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return []
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return []
        models: list[str] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            ident = entry.get("id") or entry.get("name")
            if isinstance(ident, str) and ident:
                models.append(ident)
        return models

    def _complete_uncached(self, prompt: str) -> str:
        import urllib.request

        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_output_tokens,
                "stream": False,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.request_timeout_seconds) as response:  # noqa: S310
            body = response.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"OpenAI-compatible server returned non-JSON body: {exc}") from exc
        choices = parsed.get("choices") if isinstance(parsed, dict) else None
        if not isinstance(choices, list) or not choices:
            raise LLMUnavailable(f"OpenAI-compatible response missing 'choices': {parsed}")
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
        # Fall back to legacy /completions text field if a server returned that.
        if isinstance(first.get("text"), str):
            return first["text"]
        raise LLMUnavailable(f"OpenAI-compatible response missing message content: {parsed}")


class RuleBasedProvider(LLMProvider):
    """Stub backend that always raises so callers use deterministic fallbacks."""

    name = "rule_based"

    def __init__(self, config: ProviderConfig, cache: Optional[_ResponseCache] = None) -> None:
        super().__init__(config, cache)
        self.model = "deterministic-fallback"

    def complete(self, prompt: str, *, schema_name: str = "freeform") -> LLMResponse:
        raise LLMUnavailable("RuleBasedProvider is a no-op; callers must use deterministic logic.")

    def _complete_uncached(self, prompt: str) -> str:  # pragma: no cover - never called
        raise LLMUnavailable("RuleBasedProvider does not perform completions.")


# ---------------------------------------------------------------------------
# Factory + selection
# ---------------------------------------------------------------------------


def _truthy(value: str | None) -> bool:
    return bool(value) and value.lower() not in {"0", "false", "no", "off"}


def build_provider(config: Optional[ProviderConfig] = None) -> LLMProvider:
    """Build an :class:`LLMProvider` according to ``config`` (or env defaults)."""

    config = config or _config_from_env()
    cache = _ResponseCache(config.cache_path) if config.enable_cache else None

    backend = config.backend.lower()
    # Aliases for convenience.
    backend_aliases = {
        "lmstudio": "openai_compatible",
        "lm-studio": "openai_compatible",
        "openai": "openai_compatible",
        "vllm": "openai_compatible",
    }
    backend = backend_aliases.get(backend, backend)

    candidates: list[type[LLMProvider]]
    if backend == "gemini":
        candidates = [GeminiProvider]
    elif backend == "openai_compatible":
        candidates = [OpenAICompatibleProvider]
    elif backend == "ollama":
        candidates = [OllamaProvider]
    elif backend == "rule_based":
        return RuleBasedProvider(config, cache)
    else:  # auto
        candidates = [GeminiProvider, OpenAICompatibleProvider, OllamaProvider]

    last_err: Exception | None = None
    for candidate in candidates:
        try:
            provider = candidate(config, cache)
            logger.info("LLM backend selected: %s (%s)", provider.name, provider.model)
            return provider
        except LLMUnavailable as exc:
            last_err = exc
            logger.info("LLM backend %s unavailable: %s", candidate.__name__, exc)
            continue
        except Exception as exc:  # pragma: no cover - defensive
            last_err = exc
            logger.warning("LLM backend %s failed during init: %s", candidate.__name__, exc)
            continue

    if backend in {"gemini", "openai_compatible", "ollama"}:
        raise LLMUnavailable(
            f"Requested backend '{backend}' is not available: {last_err}"
        )
    return RuleBasedProvider(config, cache)


def _config_from_env() -> ProviderConfig:
    backend = os.environ.get("LLM_BACKEND", "auto").lower()
    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    openai_base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1")
    openai_model = os.environ.get("OPENAI_MODEL", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "lm-studio")
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
    max_output = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", "1024"))
    timeout_seconds = float(os.environ.get("LLM_TIMEOUT_SECONDS", "30"))
    enable_cache = _truthy(os.environ.get("LLM_ENABLE_CACHE", "1"))
    cache_path_env = os.environ.get("LLM_CACHE_PATH")
    cache_path = Path(cache_path_env) if cache_path_env else DEFAULT_CACHE_PATH
    return ProviderConfig(
        backend=backend,
        gemini_model=gemini_model,
        ollama_model=ollama_model,
        ollama_host=ollama_host,
        openai_base_url=openai_base_url,
        openai_model=openai_model,
        openai_api_key=openai_api_key,
        temperature=temperature,
        max_output_tokens=max_output,
        request_timeout_seconds=timeout_seconds,
        enable_cache=enable_cache,
        cache_path=cache_path,
    )


_DEFAULT_PROVIDER: Optional[LLMProvider] = None
_DEFAULT_LOCK = threading.Lock()


def get_default_provider() -> LLMProvider:
    """Return a process-wide singleton provider (built lazily)."""

    global _DEFAULT_PROVIDER
    if _DEFAULT_PROVIDER is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_PROVIDER is None:
                _DEFAULT_PROVIDER = build_provider()
    return _DEFAULT_PROVIDER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_json_block(text: str) -> Any:
    """Best-effort JSON extraction from an LLM response.

    LLMs often wrap JSON in markdown fences or prose. This helper strips
    common wrappers and returns parsed JSON, or raises ``ValueError``.
    """

    if not text:
        raise ValueError("Empty response")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Strip leading ``` or ```json
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3]
    cleaned = cleaned.strip()
    # Prefer the substring between the first '{' or '[' and the last matching close.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    return json.loads(cleaned)
