#pragma once
#ifndef IXUCA_SKIP_H
#define IXUCA_SKIP_H

#include <gtest/gtest.h>

// IXUCA GPU skip utilities for IXUCA backend tests.
// These clearly distinguish hardware limitations from compatibility issues.

// Skip test because IXUCA BI-V150S lacks native FP64 ALU.
// All double-precision operations fall back to software emulation or
// are truncated to float32. Paddle-ixuca uses --skip-double for this.
#define SKIP_NO_DOUBLE(msg)                                                    \
  GTEST_SKIP() << "IXUCA: no native FP64" << (msg ? " - " : "")             \
               << (msg ? msg : "")

// Skip test because the CoreX CUDA-compatibility layer has a known
// limitation that hasn't been fixed yet.
#define SKIP_COREX_LIMITATION(msg)                                             \
  GTEST_SKIP() << "CoreX compatibility: " << msg

#endif // IXUCA_SKIP_H
