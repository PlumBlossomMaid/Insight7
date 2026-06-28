// tests/cuda/test_csv.cpp
#include "insight/insight.h"
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <sstream>

using namespace ins;

class CsvTestGPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu", "iluvatar"});
    try {
      set_device(GPUPlace(0));
    } catch (...) {
      GTEST_SKIP() << "GPU not available";
    }
  }

  void SetUp() override { filename_ = "test_cuda_support.csv"; }

  std::string filename_;
};

static bool is_kernel_registered(const std::string &op_name, DeviceKind device,
                                 DType dtype) {
  return ops().has_kernel(op_name, device, dtype);
}

TEST_F(CsvTestGPU, ExportCUDASupportCSVMatchesRegistry) {
  export_support_csv(filename_, DeviceKind::GPU);

  std::ifstream file(filename_);
  ASSERT_TRUE(file.is_open()) << "Failed to open: " << filename_;

  std::string header;
  std::getline(file, header);
  std::vector<std::string> headers;
  std::stringstream ss(header);
  std::string item;
  while (std::getline(ss, item, ',')) {
    headers.push_back(item);
  }

  std::map<std::string, int> dtype_col;
  for (size_t i = 1; i < headers.size(); ++i) {
    dtype_col[headers[i]] = static_cast<int>(i);
  }

  std::map<std::string, std::vector<std::string>> csv_data;
  std::string line;
  while (std::getline(file, line)) {
    std::stringstream ss_line(line);
    std::vector<std::string> row;
    while (std::getline(ss_line, item, ',')) {
      row.push_back(item);
    }
    if (row.size() >= 1) {
      csv_data[row[0]] = row;
    }
  }
  file.close();

  auto operators = ops().get_operator_names();

  for (const auto &op_name : operators) {
    ASSERT_TRUE(csv_data.count(op_name))
        << "Operator missing in CSV: " << op_name;

    const auto &row = csv_data[op_name];

    for (const auto &pair : dtype_col) {
      const std::string &dtype_name = pair.first;
      int col = pair.second;

      DType dtype = dtype_from_name(dtype_name);
      bool expected = is_kernel_registered(op_name, DeviceKind::GPU, dtype);
      bool actual = (row[col] == "1");

      EXPECT_EQ(actual, expected)
          << "Mismatch for " << op_name << " on " << dtype_name;
    }
  }

  for (const auto &pair : csv_data) {
    const std::string &op_name = pair.first;
    EXPECT_TRUE(std::find(operators.begin(), operators.end(), op_name) !=
                operators.end())
        << "Extra operator in CSV: " << op_name;
  }
}
