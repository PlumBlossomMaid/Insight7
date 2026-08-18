#!/usr/bin/env python3
"""Timer Demo — Verifies the Timer API works correctly."""

import time
import insight as ins


def main():
    ins.init()

    # CPU Timer
    t = ins.Timer((0, 0))
    t.start()
    time.sleep(0.005)
    t.stop()
    ms = t.elapsed_ms()
    print(f"CPU timer: {ms:.3f} ms")
    assert 0.0 <= ms <= 100.0, f"CPU timer out of range: {ms}"

    # GPU Timer (if available)
    if ins.has_device("gpu"):
        ins.init()
        t = ins.Timer((1, 0))
        t.start()
        a = ins.ones([256, 256], dtype=ins.float32, place=ins.GPUPlace(0))
        b = ins.full([256, 256], 2.0, dtype=ins.float32, place=ins.GPUPlace(0))
        _ = a + b
        t.stop()
        ms = t.elapsed_ms()
        print(f"GPU timer: {ms:.3f} ms")
        assert ms >= 0.0, f"GPU timer negative: {ms}"
    else:
        print("GPU timer: SKIPPED (GPU not available)")

    # Context manager
    with ins.Timer((0, 0)) as t:
        time.sleep(0.003)
    ms = t.elapsed_ms()
    print(f"Context manager timer: {ms:.3f} ms")

    print("OK")


if __name__ == "__main__":
    main()
