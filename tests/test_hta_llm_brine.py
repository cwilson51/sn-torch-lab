import os

import pytest
import requests

from torch_trace_lab.hta_llm import brine
from torch_trace_lab.hta_llm.brine import BrineError

KEY = "sk-test-DUMMY-KEY-123"


class Resp:
    def __init__(self, status=200, data=None, text=""):
        self.status_code, self._data, self.text = status, data, text or str(data)

    def json(self):
        if self._data is None:
            raise ValueError("no json")
        return self._data


class FakeSession:
    """Returns queued responses (or raises queued exceptions) in order."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def models(*ids):
    return Resp(data={"data": [{"id": i} for i in ids]})


def completion(content="## Summary\nok", finish="stop"):
    return Resp(data={"choices": [{"message": {"content": content}, "finish_reason": finish}],
                      "usage": {"prompt_tokens": 10, "completion_tokens": 5}})


def client(*responses):
    return brine.Client(KEY, url="https://brine.sharcnet.ca/v1", session=FakeSession(*responses),
                        sleep=lambda s: None)


def test_chat_request_shape():
    c = client(completion())
    result = c.chat("m", "sys", "user")
    method, url, kw = c.session.calls[0]
    assert (method, url) == ("POST", "https://brine.sharcnet.ca/v1/chat/completions")
    assert kw["headers"]["Authorization"] == f"Bearer {KEY}"
    assert kw["json"]["temperature"] == 0.2 and kw["json"]["max_tokens"] == 6144
    assert [m["role"] for m in kw["json"]["messages"]] == ["system", "user"]
    assert result.content.startswith("## Summary") and result.usage["prompt_tokens"] == 10


@pytest.mark.parametrize("failure", [Resp(429, text="slow down"), Resp(503, text="busy"),
                                     requests.Timeout(), requests.ConnectionError()])
def test_retries_then_succeeds(failure):
    c = client(failure, failure, completion())
    assert c.chat("m", "s", "u").content
    assert len(c.session.calls) == 3


def test_gives_up_after_three_attempts():
    c = client(Resp(500), Resp(500), Resp(500))
    with pytest.raises(BrineError, match="after 3 attempts"):
        c.chat("m", "s", "u")


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_are_clear_and_not_retried(status):
    c = client(Resp(status, text=f"bad key {KEY}"))
    with pytest.raises(BrineError, match="rejected the API key") as e:
        c.chat("m", "s", "u")
    assert KEY not in str(e.value)
    assert len(c.session.calls) == 1


def test_error_text_never_contains_key():
    c = client(Resp(400, text=f"echo {KEY}"))
    with pytest.raises(BrineError) as e:
        c.chat("m", "s", "u")
    assert KEY not in str(e.value)


def test_model_not_available_lists_models():
    c = client(models("Qwen3.8-27B", "bge-embed-large", "North-Mini-Code-1.0"))
    with pytest.raises(BrineError, match="Qwen3.8-27B, North-Mini-Code-1.0|North-Mini-Code-1.0, Qwen3.8-27B") as e:
        c.check_model("nope")
    assert "embed" not in str(e.value)


def test_finish_reason_length_is_an_error():
    with pytest.raises(BrineError, match="smaller --top-n"):
        client(completion(finish="length")).chat("m", "s", "u")


def test_empty_output_is_an_error():
    with pytest.raises(BrineError, match="empty"):
        client(completion(content="  ")).chat("m", "s", "u")


@pytest.mark.parametrize("url", ["https://api.openai.com/v1", "http://brine.sharcnet.ca/v1",
                                 "https://brine.sharcnet.ca.evil.com/v1"])
def test_egress_guard(url):
    with pytest.raises(BrineError, match="Refusing to send"):
        brine.Client(KEY, url=url, session=FakeSession())


def test_base_url_comes_from_env_only():
    assert brine.base_url({"BRINE_BASE_URL": "https://brine.sharcnet.ca/v1/"}) == \
        "https://brine.sharcnet.ca/v1"
    with pytest.raises(BrineError, match="BRINE_BASE_URL is not set"):
        brine.base_url({})


def test_key_from_env_wins(tmp_path):
    assert brine.load_key({"BRINE_API_KEY": KEY}, str(tmp_path / "none")) == KEY


def test_key_file_permissions(tmp_path):
    f = tmp_path / "brine_key"
    f.write_text(KEY + "\n")
    os.chmod(f, 0o644)
    with pytest.raises(BrineError, match="looser than 600"):
        brine.load_key({}, str(f))
    os.chmod(f, 0o600)
    assert brine.load_key({}, str(f)) == KEY


def test_missing_key(tmp_path):
    with pytest.raises(BrineError, match="No Brine API key"):
        brine.load_key({}, str(tmp_path / "none"))


def test_model_precedence():
    env = {"BRINE_MODEL": "b", "TORCH_LAB_BRINE_MODEL": "t"}
    assert brine.model_name("cli", env) == "cli"
    assert brine.model_name(None, env) == "t"
    assert brine.model_name(None, {"BRINE_MODEL": "b"}) == "b"
    assert brine.model_name(None, {}) == "DeepSeek-V4-Flash-0731"


def test_max_tokens_and_timeout_can_be_raised():
    c = brine.Client(KEY, url="https://brine.sharcnet.ca/v1", session=FakeSession(completion()),
                     sleep=lambda s: None, timeout=900)
    result = c.chat("m", "s", "u", max_tokens=16000)
    kw = c.session.calls[0][2]
    assert kw["json"]["max_tokens"] == 16000 and kw["timeout"] == 900
    assert result.finish_reason == "stop"


def test_length_error_reports_reasoning_tokens():
    resp = Resp(data={"choices": [{"message": {"content": ""}, "finish_reason": "length"}],
                      "usage": {"completion_tokens": 6144,
                                "completion_tokens_details": {"reasoning_tokens": 6000}}})
    with pytest.raises(BrineError, match="6144 output tokens, 6000 of them reasoning"):
        client(resp).chat("m", "s", "u")


@pytest.mark.parametrize("name, chat", [
    ("DeepSeek-V4-Flash-0731", True), ("Qwen3.8-27B", True), ("North-Mini-Code-1.0", True),
    ("qwen3-embed", False), ("whisper-large-v3-turbo", False), ("all-proxy-models", False),
])
def test_is_chat_model(name, chat):
    assert brine.is_chat_model(name) is chat
