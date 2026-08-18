from types import SimpleNamespace

from gr00t.policy.server_client import PolicyClient, PolicyServer


def test_policy_server_registers_deployment_metadata_endpoint():
    metadata = {"protocol": "sonic_vla_v1", "action_horizon": 40}
    policy = SimpleNamespace(
        get_action=lambda: None,
        reset=lambda: None,
        get_deployment_metadata=lambda: metadata,
    )
    server = PolicyServer(policy, host="127.0.0.1", port=0)
    try:
        endpoint = server._endpoints["get_deployment_metadata"]
        assert endpoint.requires_input is False
        assert endpoint.handler() == metadata
    finally:
        server.socket.close(linger=0)
        server.context.term()


def test_policy_client_requests_deployment_metadata_without_input():
    calls = []
    client = SimpleNamespace()
    client.call_endpoint = lambda *args, **kwargs: calls.append((args, kwargs)) or {"ok": True}

    assert PolicyClient.get_deployment_metadata(client) == {"ok": True}
    assert calls == [(("get_deployment_metadata",), {"requires_input": False})]
