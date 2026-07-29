"""Expose a canonical SONIC websocket backend through GR00T PolicyServer."""

from dataclasses import dataclass

from gr00t.policy.server_client import PolicyServer
from gr00t.policy.sonic_bridge_policy import SonicBridgePolicy
import tyro


@dataclass
class BridgeServerConfig:
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    host: str = "0.0.0.0"
    port: int = 5550
    strict: bool = True
    default_prompt: str | None = None


def main(config: BridgeServerConfig) -> None:
    print("Starting canonical SONIC bridge server...")
    print(f"  backend websocket: {config.backend_host}:{config.backend_port}")
    print(f"  GR00T PolicyServer: {config.host}:{config.port}")
    policy = SonicBridgePolicy(
        host=config.backend_host,
        port=config.backend_port,
        strict=config.strict,
        default_prompt=config.default_prompt,
    )
    print(f"  backend metadata: {policy.backend_metadata}")
    server = PolicyServer(policy=policy, host=config.host, port=config.port)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down bridge server...")


if __name__ == "__main__":
    main(tyro.cli(BridgeServerConfig))
