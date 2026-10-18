// tests/cuda/cpp/test_profiler.cpp
#include "iluvatar_skip.h"
#include "insight/c_api/place.h"
#include "insight/c_api/profiler.h"
#include "insight/core/array.h"
#include "insight/core/place.h"
#include "insight/core/profiler.h"
#include "insight/insight.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "gtest/gtest.h"
#include <thread>

using namespace ins;

class ProfilerTestGPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    INSIGHT_GPU_TEST_SETUP();
  }
};

// ========== C API GPU Tests ==========

TEST_F(ProfilerTestGPU, TimerCreateDestroy) {
  InsightPlace gpu_place = {INSIGHT_DEVICE_GPU, 0};
  InsightTimer timer = nullptr;
  ASSERT_EQ(insight_timer_create(&gpu_place, &timer), C_SUCCESS);
  ASSERT_NE(timer, nullptr);
  ASSERT_EQ(insight_timer_destroy(timer), C_SUCCESS);
}

TEST_F(ProfilerTestGPU, TimerStartStopElapsed) {
  InsightPlace gpu_place = {INSIGHT_DEVICE_GPU, 0};
  InsightTimer timer = nullptr;
  ASSERT_EQ(insight_timer_create(&gpu_place, &timer), C_SUCCESS);

  ASSERT_EQ(insight_timer_start(timer), C_SUCCESS);
  ASSERT_EQ(insight_timer_stop(timer), C_SUCCESS);

  float ms = 0.0f;
  ASSERT_EQ(insight_timer_elapsed_ms(timer, &ms), C_SUCCESS);
  EXPECT_GE(ms, 0.0f);

  ASSERT_EQ(insight_timer_destroy(timer), C_SUCCESS);
}

TEST_F(ProfilerTestGPU, TimerWithRealWork) {
  InsightPlace gpu_place = {INSIGHT_DEVICE_GPU, 0};
  InsightTimer timer = nullptr;
  ASSERT_EQ(insight_timer_create(&gpu_place, &timer), C_SUCCESS);

  ASSERT_EQ(insight_timer_start(timer), C_SUCCESS);

  Array a = ones({1024, 1024}, DType::F32, GPUPlace(0));
  Array b = full({1024, 1024}, 2.0f, DType::F32, GPUPlace(0));
  Array c = add(a, b);

  ASSERT_EQ(insight_timer_stop(timer), C_SUCCESS);

  float ms = 0.0f;
  ASSERT_EQ(insight_timer_elapsed_ms(timer, &ms), C_SUCCESS);
  EXPECT_GE(ms, 0.0f);

  ASSERT_EQ(insight_timer_destroy(timer), C_SUCCESS);
}

TEST_F(ProfilerTestGPU, TimerReset) {
  InsightPlace gpu_place = {INSIGHT_DEVICE_GPU, 0};
  InsightTimer timer = nullptr;
  ASSERT_EQ(insight_timer_create(&gpu_place, &timer), C_SUCCESS);

  ASSERT_EQ(insight_timer_start(timer), C_SUCCESS);
  ASSERT_EQ(insight_timer_stop(timer), C_SUCCESS);
  float ms1 = 0.0f;
  insight_timer_elapsed_ms(timer, &ms1);

  ASSERT_EQ(insight_timer_reset(timer), C_SUCCESS);
  ASSERT_EQ(insight_timer_start(timer), C_SUCCESS);
  ASSERT_EQ(insight_timer_stop(timer), C_SUCCESS);
  float ms2 = 0.0f;
  ASSERT_EQ(insight_timer_elapsed_ms(timer, &ms2), C_SUCCESS);
  EXPECT_GE(ms2, 0.0f);

  ASSERT_EQ(insight_timer_destroy(timer), C_SUCCESS);
}

// ========== C++ Timer Class GPU Tests ==========

TEST_F(ProfilerTestGPU, CppTimerCreate) {
  ins::Timer timer(ins::GPUPlace(0));
  EXPECT_FALSE(timer.started());
}

TEST_F(ProfilerTestGPU, CppTimerMeasure) {
  ins::Timer timer(ins::GPUPlace(0));
  timer.start();
  Array a = ones({512, 512}, DType::F32, GPUPlace(0));
  Array b = full({512, 512}, 3.0f, DType::F32, GPUPlace(0));
  Array c = add(a, b);
  timer.stop();
  float ms = timer.elapsed_ms();
  EXPECT_GE(ms, 0.0f);
}

TEST_F(ProfilerTestGPU, CppTimerMultipleMeasures) {
  ins::Timer timer(ins::GPUPlace(0));
  for (int i = 0; i < 3; ++i) {
    timer.reset();
    timer.start();
    Array a = ones({256, 256}, DType::F32, GPUPlace(0));
    Array b = ones({256, 256}, DType::F32, GPUPlace(0));
    Array c = add(a, b);
    timer.stop();
    float ms = timer.elapsed_ms();
    EXPECT_GE(ms, 0.0f);
  }
}
