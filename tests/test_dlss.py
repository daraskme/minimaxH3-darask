from __future__ import annotations

from io import BytesIO
import json

import pytest

from h3studio.dlss import (
    DLSSProtocolError,
    WIRE_HEADER,
    WIRE_MAGIC,
    WIRE_PREFLIGHT,
    WIRE_RESULT,
    WIRE_VERSION,
    decode_preflight,
    decode_result,
    iter_wire_messages,
    _is_quantized_cfr_delta,
)


class Fragmented(BytesIO):
    def read(self, size: int = -1) -> bytes:
        return super().read(min(size, 2) if size >= 0 else 2)


def valid_result_payload(detail: str = "完了") -> bytes:
    detail_bytes = detail.encode("utf-16-le")
    fixed = WIRE_RESULT.pack(
        1, 0, 0, 1, 1, 1, 1, 1, 0, 0,
        24, 10_000_000, 24, 24, 24, 42,
        1, 0, 0,
        3.1, 3.8, 4.5, 0.4, 0.7,
        512, 24, 1, 2, 3, len(detail_bytes),
    )
    return fixed + detail_bytes


def test_v6_wire_decoder_handles_fragmented_reads() -> None:
    payload = valid_result_payload()
    stream = Fragmented(WIRE_HEADER.pack(WIRE_MAGIC, WIRE_VERSION, 2, len(payload)) + payload)
    messages = list(iter_wire_messages(stream))
    assert len(messages) == 1
    result = decode_result(messages[0][1])
    assert result.ok is True
    assert result.frame_count == result.native_evaluations == result.verified_neural_frames == 24
    assert result.detail == "完了"


def test_v6_result_rejects_nonboolean_and_inconsistent_success() -> None:
    payload = bytearray(valid_result_payload())
    payload[0] = 2
    with pytest.raises(DLSSProtocolError):
        decode_result(bytes(payload))

    payload = bytearray(valid_result_payload())
    payload[7] = 0  # feature18Evaluated
    with pytest.raises(DLSSProtocolError):
        decode_result(bytes(payload))


def test_v6_header_rejects_wrong_identity_and_truncation() -> None:
    with pytest.raises(DLSSProtocolError):
        list(iter_wire_messages(BytesIO(WIRE_HEADER.pack(0, WIRE_VERSION, 2, 0))))
    with pytest.raises(DLSSProtocolError):
        list(iter_wire_messages(Fragmented(b"short")))


def test_v6_preflight_is_strict_and_data_only() -> None:
    body = json.dumps({"runtime": "locked"}).encode()
    value = decode_preflight(WIRE_PREFLIGHT.pack(1, len(body)) + body)
    assert value == {"runtime": "locked", "ok": True}
    invalid = bytearray(WIRE_PREFLIGHT.pack(1, len(body)) + body)
    invalid[1] = 1
    with pytest.raises(DLSSProtocolError):
        decode_preflight(bytes(invalid))


def test_worker_matroska_cfr_accepts_one_millisecond_quantization() -> None:
    assert _is_quantized_cfr_delta(0.017, 60.0, 0.001)
    assert _is_quantized_cfr_delta(0.016, 60.0, 0.001)
    assert _is_quantized_cfr_delta(0.017, 60000 / 1001, 0.001)
    assert not _is_quantized_cfr_delta(0.015, 60.0, 0.001)
    assert not _is_quantized_cfr_delta(0.0, 60.0, 0.001)
