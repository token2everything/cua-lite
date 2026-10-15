"""Resource GET retries preserve ownership errors and complete pagination."""

from __future__ import annotations

import json

import pytest
import requests
import sandbox


@pytest.fixture
def client(tmp_path):
    return sandbox.SandboxClient(
        {"BASE": "http://sandbox.test", "PROJECT": "test-project",
         "OPEN_SANDBOX_ACCESS_KEY": "fixture-ak", "OPEN_SANDBOX_SECRET_KEY": "fixture-sk",
         "OPENAI_API_KEY": "fixture-model-key", "OPENAI_BASE_URL": "http://model.test"},
        api_key_env="OPENAI_API_KEY", receipts=tmp_path,
    )


def response(status, body):
    result = requests.Response()
    result.status_code = status
    result.url = "http://sandbox.test/v1/sandboxes"
    result._content = json.dumps(body).encode()
    return result


@pytest.mark.parametrize("first", [429, 502, 503, requests.Timeout("read timeout"),
                                   requests.ConnectionError("connection reset")])
def test_detail_recovers_only_transient_resource_reads(client, monkeypatch, first):
    calls = []
    outcomes = [first if isinstance(first, Exception) else response(first, {"error": "transient"}),
                response(200, {"id": "owned-sandbox", "metadata": {"session_nonce": "nonce"}})]
    delays = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        result = outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(sandbox.requests, "get", get)
    monkeypatch.setattr(sandbox.time, "sleep", delays.append)
    assert client.detail("owned-sandbox") == {"id": "owned-sandbox", "metadata": {"session_nonce": "nonce"}}
    assert len(calls) == 2 and calls[0] == calls[1]
    assert calls[0][0] == "http://sandbox.test/v1/sandboxes/owned-sandbox"
    assert calls[0][1]["timeout"] == 30
    assert delays == [2]


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_detail_preserves_immediate_resource_rejection(client, monkeypatch, status):
    calls = []
    monkeypatch.setattr(sandbox.requests, "get", lambda *a, **kw: calls.append(kw) or response(status, {}))
    monkeypatch.setattr(sandbox.time, "sleep", lambda _: pytest.fail("Permanent rejection was retried"))
    with pytest.raises(requests.HTTPError) as caught:
        client.detail("owned-sandbox")
    assert caught.value.response.status_code == status
    assert len(calls) == 1


@pytest.mark.parametrize("failure", [502, requests.Timeout("read timeout")])
def test_resource_read_retry_limit_preserves_last_error(client, monkeypatch, failure):
    calls = []
    delays = []

    def get(*args, **kwargs):
        calls.append(kwargs)
        if isinstance(failure, Exception):
            raise failure
        return response(failure, {"error": "unavailable"})

    monkeypatch.setattr(sandbox.requests, "get", get)
    monkeypatch.setattr(sandbox.time, "sleep", delays.append)
    with pytest.raises(requests.Timeout if isinstance(failure, Exception) else requests.HTTPError):
        client.detail("owned-sandbox")
    assert len(calls) == 4 and delays == [2, 4, 8]


def test_page_retry_preserves_page_number_order_and_each_item_once(client, monkeypatch):
    pages = []
    failures = {2: 1}

    def get(url, *, params, **kwargs):
        assert url == "http://sandbox.test/v1/sandboxes"
        assert params["pageSize"] == 1000 and kwargs["timeout"] == 45
        page = params["page"]
        pages.append(page)
        if failures.get(page):
            failures[page] -= 1
            return response(502, {"error": "transient"})
        return response(200, {"items": [{"id": f"sandbox-{page}"}],
                              "pagination": {"hasNextPage": page < 3}})

    monkeypatch.setattr(sandbox.requests, "get", get)
    monkeypatch.setattr(sandbox.time, "sleep", lambda _: None)
    assert client.list_sandboxes() == [{"id": f"sandbox-{page}"} for page in range(1, 4)]
    assert pages == [1, 2, 2, 3]


def test_invalid_success_body_is_not_retried_as_a_transport_failure(client, monkeypatch):
    calls = []
    invalid = response(200, {})
    invalid._content = b"not JSON"
    monkeypatch.setattr(sandbox.requests, "get", lambda *a, **kw: calls.append(kw) or invalid)
    with pytest.raises(requests.JSONDecodeError):
        client.detail("owned-sandbox")
    assert len(calls) == 1
