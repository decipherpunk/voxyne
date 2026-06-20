"""Byte-level conversational markers for Voxyne.

Marker format: STX + ascii-tag + ETX (0x01 + tag + 0x02). Control bytes are rare
in real UTF-8 text, so these give the model a clear, collision-free signal.
This module holds the conversational subset (the data-building wrappers live in
the training package).
"""

from __future__ import annotations

import numpy as np

STX = 0x01
ETX = 0x02


def _tag(name: str) -> bytes:
    if any(c <= 0x02 for c in name.encode("ascii")):
        raise ValueError("tag name cannot contain control bytes")
    return bytes([STX]) + name.encode("ascii") + bytes([ETX])


BOS = _tag("bos")
EOS = _tag("eos")
USER = _tag("user")
ASSISTANT = _tag("assistant")
SYSTEM = _tag("system")
ENDTURN = _tag("endturn")
PAD = _tag("pad")


def sigma_k_for_dialogue_bytes(data: bytes) -> bytes:
    """Role-bit stream over a wrapped dialogue: user-turn bytes -> -1, everything
    else (assistant, system, markers) -> +1. Serialized as int8 bytes (0xff=-1,
    0x01=+1), directly memmap-able as np.int8."""
    n = len(data)
    sig = np.ones(n, dtype=np.int8)
    in_user = False
    i = 0
    while i < n:
        if data[i:i + len(USER)] == USER:
            in_user = True
            i += len(USER)
        elif data[i:i + len(ASSISTANT)] == ASSISTANT:
            in_user = False
            i += len(ASSISTANT)
        elif data[i:i + len(SYSTEM)] == SYSTEM:
            in_user = False
            i += len(SYSTEM)
        elif data[i:i + len(ENDTURN)] == ENDTURN:
            i += len(ENDTURN)
        else:
            sig[i] = -1 if in_user else +1
            i += 1
    return sig.tobytes()
