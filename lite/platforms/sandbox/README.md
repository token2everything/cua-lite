# Sandbox platform client

Use the shared transport through its owning module:

```python
from lite.platforms.sandbox.client import SandboxClient, load_credentials

client = SandboxClient(
    base_url="https://manager.example",
    project="my-project",
    afs_url="https://afs.example",
    credentials=load_credentials(),
)
```

Provide `OPEN_SANDBOX_ACCESS_KEY` and `OPEN_SANDBOX_SECRET_KEY`, or pass an
explicit credentials file path to `load_credentials`. Constructing the client
does not contact the platform or create a Sandbox.

[client.py](/lite/platforms/sandbox/client.py) owns platform API transport for
Sandbox lifecycle operations, credential loading and redaction, proxy endpoint
resolution, AFS transfers through `rclone`, private JSON writes, and optional
Feishu webhook configuration and connection tests. Credentials and webhook
configuration come only from explicit file paths or the documented environment
variables; this package does not search benchmark directories or local state.
`SANDBOX_FEISHU_WEBHOOK` supplies the optional webhook when no file is given.
Sending a connection-test message requires an explicit call to `feishu_test`.

Benchmark packages own tasks, evaluators, release versions, Worker protocols,
deployment configuration, and resource ownership decisions. They call this
client with their resolved configuration and enforce which resources they may
delete. The platform layer has no benchmark or local dataset dependency and
does not register Gym environments or load an evaluator on import.

Offline transport tests live in
[tests/platforms/sandbox/test_client.py](/tests/platforms/sandbox/test_client.py).
They use fake platform, AFS, and Feishu responses and require no credentials or
running Sandbox.
