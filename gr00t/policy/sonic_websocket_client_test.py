import numpy as np
import pytest

from gr00t.policy import sonic_websocket_client


def test_numpy_wire_codec_round_trip_and_refuses_object_arrays():
    value = {
        "image": np.zeros((4, 5, 3), dtype=np.uint8),
        "state": np.arange(46, dtype=np.float32),
    }
    restored = sonic_websocket_client._unpackb(sonic_websocket_client._packb(value))
    np.testing.assert_array_equal(restored["image"], value["image"])
    np.testing.assert_array_equal(restored["state"], value["state"])

    with pytest.raises(ValueError, match="Unsupported dtype"):
        sonic_websocket_client._packb({"unsafe": np.array([object()], dtype=object)})
