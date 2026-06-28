#pragma once
#include <gtest/gtest.h>

// Iluvatar GPU skip utilities for IXUCA backend tests.
// These clearly distinguish hardware limitations from compatibility issues.

// Skip test because Iluvatar BI-V150S lacks native FP64 ALU.
// All double-precision operations fall back to software emulation or
// are truncated to float32. Paddle-iluvatar uses --skip-double for this.
#define SKIP_NO_DOUBLE(msg)                                                    \
  GTEST_SKIP() << "Iluvatar: no native FP64" << (msg ? " - " : "")             \
               << (msg ? msg : "")

// Skip test because the CoreX CUDA-compatibility layer has a known
// limitation that hasn't been fixed yet.
#define SKIP_COREX_LIMITATION(msg)                                             \
  GTEST_SKIP() << "CoreX compatibility: " << msg

#endif // ILUVATAR_SKIP_H
