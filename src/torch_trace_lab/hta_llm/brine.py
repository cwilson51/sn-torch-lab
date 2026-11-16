"""Client for SHARCNET Brine AIaaS (OpenAI-compatible chat completions, spec §10).

The API key is never logged, printed, or written to any output, including error messages.
"""

import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

from torch_trace_lab.hta_llm import settings

KEY_HELP = ("Set BRINE_API_KEY, or put the key in ~/.config/torch-lab/brine_key with "
            "permissions 600 (chmod 600 ~/.config/torch-lab/brine_key). Each user needs their "
            "own Brine key; see https://brine.sharcnet.ca.")


class BrineError(Exception):
    """A Brine problem with a message that says what to do about it."""


@dataclass
class Completion:
    content: str
    usage: dict
    latency_s: float
    model: str


def base_url(environ=None) -> str:
    environ = os.environ if environ is None else environ
    return environ.get("BRINE_BASE_URL", settings.BRINE_BASE_URL).rstrip("/")


def model_name(cli_model: str | None, environ=None) -> str:
    environ = os.environ if environ is None else environ
    return (cli_model or environ.get("TORCH_LAB_BRINE_MODEL") or environ.get("BRINE_MODEL")
            or settings.BRINE_DEFAULT_MODEL)


def check_egress(url: str) -> None:
    """Refuse any base URL that isn't HTTPS to an allowlisted host."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in settings.ALLOWED_HOSTS:
        raise BrineError(f"Refusing to send data to {parsed.scheme}://{parsed.hostname}: only "
                         f"https to {', '.join(sorted(settings.ALLOWED_HOSTS))} is allowed. "
                         "Unset BRINE_BASE_URL to use the default.")


def load_key(environ=None, key_file: str | None = None) -> str:
    environ = os.environ if environ is None else environ
    key_file = key_file or settings.BRINE_KEY_FILE
    key = environ.get("BRINE_API_KEY", "").strip()
    if key:
        return key
    path = Path(os.path.expanduser(key_file))
    if not path.exists():
        raise BrineError(f"No Brine API key found. {KEY_HELP}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise BrineError(f"Refusing to read {key_file}: its permissions are {oct(mode)}, "
                         f"looser than 600. Run: chmod 600 {key_file}")
    key = path.read_text().strip()
    if not key:
        raise BrineError(f"{key_file} is empty. {KEY_HELP}")
    return key


class Client:
    def __init__(self, key: str, url: str | None = None, session=None, sleep=time.sleep):
        self.url = url or base_url()
        check_egress(self.url)
        self._key = key
        self.session = session or requests.Session()
        self.sleep = sleep

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}

    def _scrub(self, text: str) -> str:
        return text.replace(self._key, "[key]")[:300]

    def _request(self, method: str, path: str, **kwargs):
        last = None
        for attempt in range(settings.RETRIES):
            if attempt:
                self.sleep(settings.RETRY_BACKOFF_S * 2 ** (attempt - 1))
            try:
                resp = self.session.request(method, f"{self.url}{path}", headers=self._headers(),
                                            timeout=settings.REQUEST_TIMEOUT_S, **kwargs)
            except requests.Timeout:
                last = "the request timed out"
                continue
            except requests.ConnectionError:
                last = (f"could not connect to {urlparse(self.url).hostname}. Check that this "
                        "machine can reach Brine (some cluster compute nodes cannot)")
                continue
            if resp.status_code in (401, 403):
                raise BrineError(f"Brine rejected the API key (HTTP {resp.status_code}). {KEY_HELP}")
            if resp.status_code == 429 or resp.status_code >= 500:
                last = f"HTTP {resp.status_code}: {self._scrub(resp.text)}"
                continue
            if resp.status_code >= 400:
                raise BrineError(f"Brine returned HTTP {resp.status_code}: {self._scrub(resp.text)}")
            try:
                return resp.json()
            except ValueError:
                raise BrineError("Brine returned a response that is not JSON.") from None
        raise BrineError(f"Brine request failed after {settings.RETRIES} attempts: {last}")

    def models(self) -> list[str]:
        data = self._request("GET", "/models")
        return sorted(m["id"] for m in data.get("data", []) if isinstance(m, dict) and "id" in m)

    def check_model(self, model: str) -> None:
        available = self.models()
        if model not in available:
            chat = [m for m in available if "embed" not in m.lower()]
            raise BrineError(f"Model {model!r} is not available on Brine. Available chat models: "
                             f"{', '.join(chat) or '(none listed)'}. Choose one with --model.")

    def chat(self, model: str, system: str, user: str) -> Completion:
        start = time.monotonic()
        data = self._request("POST", "/chat/completions", json={
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": settings.TEMPERATURE,
            "max_tokens": settings.MAX_TOKENS,
        })
        latency = time.monotonic() - start
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise BrineError("Brine returned an unexpected response (no choices).") from None
        if choice.get("finish_reason") == "length":
            raise BrineError("The model ran out of output tokens before finishing the report. "
                             "Try a smaller --top-n.")
        if not content.strip():
            raise BrineError("The model returned an empty report. Try again, or a smaller --top-n.")
        return Completion(content, data.get("usage") or {}, latency, model)
