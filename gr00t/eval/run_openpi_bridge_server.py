# SPDX-License-Identifier: Apache-2.0
"""Launch a GR00T PolicyServer backed by an openpi (pi0.5) SONIC model via the bridge.

Two-process bridge for driving the Unitree G1 SONIC embodiment with an openpi checkpoint:

  process 1 (openpi venv):   pi0.5 finetuned policy served over a websocket
      uv run python scripts/serve_policy.py --port 8000 \
          policy:checkpoint --policy.config pi05_sonic --policy.dir <ckpt>

  process 2 (GR00T venv):    this launcher — a ZeroMQ PolicyServer on :5550 that forwards
                             observations to the openpi websocket via OpenpiBridgePolicy
      python -m gr00t.eval.run_openpi_bridge_server --port 5550 --openpi-port 8000

The existing sonic client (gear_sonic/scripts/launch_inference.py --camera-host ...) then
connects to :5550 unchanged.
"""

from dataclasses import dataclass

import tyro

from gr00t.policy.openpi_bridge_policy import OpenpiBridgePolicy
from gr00t.policy.server_client import PolicyServer


@dataclass
class BridgeServerConfig:
    # openpi websocket server (process 1).
    openpi_host: str = "127.0.0.1"
    openpi_port: int = 8000

    # This GR00T-facing ZeroMQ PolicyServer (what the sonic client connects to).
    host: str = "0.0.0.0"
    port: int = 5550

    # Validation + optional fallback prompt if the observation carries no language.
    strict: bool = True
    default_prompt: str | None = None


def main(config: BridgeServerConfig) -> None:
    print("Starting openpi->GR00T SONIC bridge server...")
    print(f"  openpi websocket: {config.openpi_host}:{config.openpi_port}")
    print(f"  GR00T PolicyServer: {config.host}:{config.port}")

    policy = OpenpiBridgePolicy(
        host=config.openpi_host,
        port=config.openpi_port,
        strict=config.strict,
        default_prompt=config.default_prompt,
    )

    server = PolicyServer(policy=policy, host=config.host, port=config.port)
    print(f"\n✓ Bridge ready — listening on {config.host}:{config.port}\n")
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down bridge server...")


if __name__ == "__main__":
    main(tyro.cli(BridgeServerConfig))
