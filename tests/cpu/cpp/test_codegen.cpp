#include "gtest/gtest.h"

#include <string>
#include <vector>

#include "insight/generated/kernel_plan.h"
#include "insight/generated/source_manifest.h"

TEST(CodegenTest, EmitsOneBackendNeutralPlanPerOperator) {
  ASSERT_GT(ins::generated::kernel_plan_count, 0u);
  std::vector<std::string> names;
  for (unsigned i = 0; i < ins::generated::kernel_plan_count; ++i) {
    const auto &plan = ins::generated::kernel_plans[i];
    ASSERT_NE(plan.name, nullptr);
    ASSERT_NE(plan.kind, nullptr);
    ASSERT_NE(plan.dtypes, nullptr);
    ASSERT_NE(plan.host, nullptr);
    ASSERT_NE(plan.device, nullptr);
    names.emplace_back(plan.name);
  }
  for (size_t i = 1; i < names.size(); ++i)
    EXPECT_LT(names[i - 1], names[i]);
}

TEST(CodegenTest, SourceManifestMatchesKernelPlan) {
  ASSERT_EQ(ins::generated::source_manifest_count,
            ins::generated::kernel_plan_count);
  for (unsigned i = 0; i < ins::generated::source_manifest_count; ++i) {
    EXPECT_STREQ(ins::generated::source_manifest[i].name,
                 ins::generated::kernel_plans[i].name);
    EXPECT_STREQ(ins::generated::source_manifest[i].host_adapter,
                 ins::generated::kernel_plans[i].host);
    EXPECT_STREQ(ins::generated::source_manifest[i].device_adapter,
                 ins::generated::kernel_plans[i].device);
  }
}

TEST(CodegenTest, UsesSharedHostAndDeviceAdapters) {
  for (unsigned i = 0; i < ins::generated::kernel_plan_count; ++i) {
    const auto &plan = ins::generated::kernel_plans[i];
    EXPECT_NE(std::string(plan.host).find("strategy:") == 0 ||
                  std::string(plan.host).find("expression:") == 0,
              false);
    EXPECT_NE(std::string(plan.device).find("strategy:") == 0 ||
                  std::string(plan.device).find("expression:") == 0,
              false);
  }
}
