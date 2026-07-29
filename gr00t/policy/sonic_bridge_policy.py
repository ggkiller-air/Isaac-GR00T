"""Backend-neutral name for the canonical SONIC websocket bridge."""

from gr00t.policy.openpi_bridge_policy import OpenpiBridgePolicy


class SonicBridgePolicy(OpenpiBridgePolicy):
    """Bridge any backend implementing the ``sonic_vla_v1`` wire contract."""
