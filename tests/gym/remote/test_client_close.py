"""Remote DELETE acknowledgments retain the identity needed for cleanup retries."""

from __future__ import annotations

import httpx
import pytest

from lite.gym.errors import EnvAccessDenied
from lite.gym.remote import client as remote_client


@pytest.fixture
def client_with_delete(monkeypatch):
    clients = []
    real_async_client = httpx.AsyncClient

    def make(handler):
        client = remote_client.LiteEnvClient.__new__(remote_client.LiteEnvClient)
        client._url = "http://test-env-server"
        client._id = "owned-instance"
        client._client = real_async_client(headers={"Authorization": "Bearer test-token"})
        clients.append(client._client)

        def fresh_client(**kwargs):
            fresh = real_async_client(transport=httpx.MockTransport(handler), **kwargs)
            clients.append(fresh)
            return fresh

        monkeypatch.setattr(remote_client.httpx, "AsyncClient", fresh_client)
        return client, clients

    return make


@pytest.mark.parametrize("status", [403, 500, 503])
async def test_http_close_failure_can_retry_same_identity(client_with_delete, status):
    requests = []

    def respond(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(status, text="DELETE failed")
        return httpx.Response(200, json={"ok": True})

    client, clients = client_with_delete(respond)
    with pytest.raises(httpx.HTTPStatusError) as error:
        await client.close()
    assert error.value.response.status_code == status
    assert client._id == "owned-instance"
    assert all(item.is_closed for item in clients)

    await client.close()
    assert client._id is None
    assert all(item.is_closed for item in clients)
    assert len(requests) == 2
    assert all(request.method == "DELETE" for request in requests)
    assert all(str(request.url) == "http://test-env-server/instances/owned-instance"
               for request in requests)
    assert all(request.headers["Authorization"] == "Bearer test-token" for request in requests)
    await client.close()
    assert len(requests) == 2


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
async def test_transport_close_failure_keeps_identity_and_auth(client_with_delete, error_type):
    requests = []

    def respond(request):
        requests.append(request)
        if len(requests) == 1:
            raise error_type("DELETE response lost", request=request)
        return httpx.Response(200, json={"ok": True})

    client, clients = client_with_delete(respond)
    with pytest.raises(error_type, match="DELETE response lost"):
        await client.close()
    assert client._id == "owned-instance"
    assert all(item.is_closed for item in clients)
    await client.close()
    assert client._id is None
    assert len(requests) == 2
    assert requests[0].url == requests[1].url
    assert requests[1].headers["Authorization"] == "Bearer test-token"


async def test_close_keeps_typed_server_errors(client_with_delete):
    failure = EnvAccessDenied(what="Instance is owned by a different session")
    client, clients = client_with_delete(
        lambda request: httpx.Response(403, json=failure.to_payload())
    )
    with pytest.raises(EnvAccessDenied, match="different session"):
        await client.close()
    assert client._id == "owned-instance"
    assert all(item.is_closed for item in clients)


@pytest.mark.parametrize("status", [200, 204])
async def test_successful_close_clears_identity_once(client_with_delete, status):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(status)

    client, clients = client_with_delete(respond)
    await client.close()
    assert client._id is None
    assert all(item.is_closed for item in clients)
    await client.close()
    assert len(requests) == 1
