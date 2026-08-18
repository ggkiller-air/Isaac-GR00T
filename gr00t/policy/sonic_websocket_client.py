"""Small synchronous client for the canonical SONIC websocket contract."""

import functools
import inspect
import logging
import time
from typing import Any

import msgpack
import numpy as np
import websockets.sync.client


def _pack_array(value):
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def _unpack_array(value):
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=value[b"shape"],
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


_packb = functools.partial(msgpack.packb, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class SonicWebsocketClient:
    """Connect to an openpi-compatible backend and exchange NumPy observations."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        retry_interval_sec: float = 5.0,
    ) -> None:
        self._uri = host if host.startswith("ws") else f"ws://{host}"
        self._uri += f":{port}"
        self._retry_interval_sec = retry_interval_sec
        self._ws, self._metadata = self._wait_for_server()

    def _wait_for_server(self):
        logging.info("Waiting for SONIC backend at %s...", self._uri)
        while True:
            try:
                kwargs: dict[str, Any] = {"compression": None, "max_size": None}
                if "proxy" in inspect.signature(websockets.sync.client.connect).parameters:
                    kwargs["proxy"] = None
                connection = websockets.sync.client.connect(self._uri, **kwargs)
                return connection, _unpackb(connection.recv())
            except (ConnectionRefusedError, OSError):
                logging.info("SONIC backend is not ready; retrying...")
                time.sleep(self._retry_interval_sec)

    def get_server_metadata(self) -> dict[str, Any]:
        return self._metadata

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(_packb(observation))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in SONIC backend:\n{response}")
        return _unpackb(response)

    def close(self) -> None:
        self._ws.close()
