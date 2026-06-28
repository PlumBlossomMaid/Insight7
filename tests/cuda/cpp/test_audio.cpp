// tests/cuda/test_audio.cpp
//
// CUDA audio tests — mirrors CPU test_audio.cpp with GPU transfer + compute.
#ifdef INSIGHT_USE_SNDFILE

#include "insight/core/array.h"
#include "insight/init.h"
#include "insight/io/sndfile.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/fft.h"
#include "insight/ops/reduction.h"
#include "insight/utils/features.h"
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace ins;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class AudioTestGPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu", "iluvatar"});
    try {
      set_device(GPUPlace(0));
    } catch (...) {
      GTEST_SKIP() << "GPU not available";
    }
  }
};

static Array make_sine(int64_t frames, int channels, float freq,
                       float sample_rate) {
  std::vector<float> data(frames * channels);
  for (int64_t i = 0; i < frames; ++i) {
    float val =
        std::sin(2.0f * static_cast<float>(M_PI) * freq * i / sample_rate);
    for (int ch = 0; ch < channels; ++ch) {
      data[i * channels + ch] = val;
    }
  }
  Shape shape({frames, channels});
  return to_array<float>(data, shape, ins::CPUPlace());
}

static const std::string tmp_path(const std::string &name,
                                  const std::string &ext = ".wav") {
  return "/tmp/insight_test_audio_gpu_" + name + ext;
}

// ========== Round-trip tests (CPU write -> CPU read -> GPU -> CPU compare)
// ==========

TEST_F(AudioTestGPU, ReadWriteMono) {
  std::string path = tmp_path("mono");
  int64_t frames = 1024;
  Array original = make_sine(frames, 1, 440.0f, 44100.0f);

  // Write from CPU, read back
  audio::write(path, original, 44100);
  auto [read_back, info] = audio::read(path, ins::CPUPlace());
  EXPECT_EQ(info.frames, frames);
  EXPECT_EQ(info.samplerate, 44100);
  EXPECT_EQ(info.channels, 1);

  // Transfer to GPU and back, verify no data loss
  Array gpu_data = read_back.to(GPUPlace(0));
  Array cpu_data = gpu_data.to(CPUPlace());
  Array diff_arr = abs(sub(cpu_data, original));
  Array max_diff = max(diff_arr);
  EXPECT_LT(max_diff.item<float>(), 1e-6f);

  std::remove(path.c_str());
}

TEST_F(AudioTestGPU, ReadWriteStereo) {
  std::string path = tmp_path("stereo");
  int64_t frames = 2048;
  Array original = make_sine(frames, 2, 880.0f, 44100.0f);

  audio::write(path, original, 48000);
  auto [read_back, info] = audio::read(path, ins::CPUPlace());
  EXPECT_EQ(info.frames, frames);
  EXPECT_EQ(info.samplerate, 48000);
  EXPECT_EQ(info.channels, 2);

  Array gpu_data = read_back.to(GPUPlace(0));
  Array cpu_data = gpu_data.to(CPUPlace());
  Array diff_arr = abs(sub(cpu_data, original));
  Array max_diff = max(diff_arr);
  EXPECT_LT(max_diff.item<float>(), 1e-6f);

  std::remove(path.c_str());
}

TEST_F(AudioTestGPU, WriteMono1D) {
  std::string path = tmp_path("mono1d");
  int64_t frames = 512;

  std::vector<float> data(frames);
  for (int64_t i = 0; i < frames; ++i)
    data[i] = std::sin(2.0f * static_cast<float>(M_PI) * 220.0f * i / 44100.0f);
  Array original = to_array<float>(data, ins::CPUPlace());

  // Create on GPU, write should auto-transfer to CPU
  Array gpu_original = original.to(GPUPlace(0));
  audio::write(path, gpu_original, 44100);

  auto [read_back, info] = audio::read(path, ins::CPUPlace());
  EXPECT_EQ(info.frames, frames);
  EXPECT_EQ(info.channels, 1);

  Array read_flat = read_back.reshape({frames});
  Array diff_arr = abs(sub(read_flat, original));
  Array max_diff = max(diff_arr);
  EXPECT_LT(max_diff.item<float>(), 1e-6f);

  std::remove(path.c_str());
}

TEST_F(AudioTestGPU, ReadNonexistentThrows) {
  EXPECT_THROW(audio::read("/tmp/nonexistent_audio_file_xyz_gpu.wav"),
               std::exception);
}

TEST_F(AudioTestGPU, WriteInvalidDimsThrows) {
  Array bad = zeros({2, 3, 4}, DType::F32, ins::CPUPlace());
  EXPECT_THROW(audio::write(tmp_path("bad"), bad), std::exception);
}

TEST_F(AudioTestGPU, WriteReadFlac) {
  std::string path = tmp_path("flac_roundtrip", ".flac");
  int64_t frames = 2048;
  Array original = make_sine(frames, 1, 440.0f, 44100.0f);

  audio::write(path, original, 44100);
  auto [read_back, info] = audio::read(path, ins::CPUPlace());
  EXPECT_EQ(info.frames, frames);
  EXPECT_EQ(info.samplerate, 44100);
  EXPECT_EQ(info.channels, 1);

  // GPU round-trip + FLAC 24-bit quantization
  Array gpu_data = read_back.to(GPUPlace(0));
  Array cpu_data = gpu_data.to(CPUPlace());
  Array diff_arr = abs(sub(cpu_data, original));
  Array max_diff = max(diff_arr);
  EXPECT_LT(max_diff.item<float>(), 1e-5f);

  std::remove(path.c_str());
}

TEST_F(AudioTestGPU, WriteReadOgg) {
  std::string path = tmp_path("ogg_roundtrip", ".ogg");
  int64_t frames = 4096;
  Array original = make_sine(frames, 1, 440.0f, 44100.0f);

  audio::write(path, original, 44100);
  auto [read_back, info] = audio::read(path, ins::CPUPlace());
  EXPECT_EQ(info.frames, frames);
  EXPECT_EQ(info.samplerate, 44100);
  EXPECT_EQ(info.channels, 1);

  Array gpu_data = read_back.to(GPUPlace(0));
  Array cpu_data = gpu_data.to(CPUPlace());
  Array diff_arr = abs(sub(cpu_data, original));
  Array max_diff = max(diff_arr);
  EXPECT_LT(max_diff.item<float>(), 0.05f);

  std::remove(path.c_str());
}

TEST_F(AudioTestGPU, ReadWithExplicitGPUPlace) {
  std::string path = tmp_path("place_gpu");
  int64_t frames = 512;
  Array original = make_sine(frames, 1, 440.0f, 44100.0f);
  audio::write(path, original, 44100);

  // Read directly to GPU
  auto [gpu_data, info] = audio::read(path, GPUPlace(0));
  EXPECT_EQ(info.frames, frames);

  // Verify data is on GPU by transferring back
  Array cpu_data = gpu_data.to(CPUPlace());
  Array diff_arr = abs(sub(cpu_data, original));
  Array max_diff = max(diff_arr);
  EXPECT_LT(max_diff.item<float>(), 1e-6f);

  std::remove(path.c_str());
}

// ========== GPU-specific tests (no CPU counterpart) ==========

TEST_F(AudioTestGPU, FFTRoundTrip) {
  if (!ins::is_compiled_with_fftw3())
    GTEST_SKIP() << "FFTW3 not available";
  int64_t frames = 4096;
  Array cpu_signal = make_sine(frames, 1, 440.0f, 44100.0f);

  Array signal_1d = cpu_signal.reshape({frames});
  Array gpu_signal = signal_1d.to(GPUPlace(0));
  Array gpu_spectrum = fft::rfft(gpu_signal);
  Array gpu_reconstructed = fft::irfft(gpu_spectrum, frames);

  Array cpu_result = gpu_reconstructed.to(CPUPlace()).reshape({frames});
  Array diff_arr = abs(sub(cpu_result, signal_1d));
  Array max_diff = max(diff_arr);
  EXPECT_LT(max_diff.item<float>(), 1e-3f);
}

#endif // INSIGHT_USE_SNDFILE
