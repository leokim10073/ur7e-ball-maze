// The reference shaper: clamp, ignore non-finite, low-pass, hard rate limit.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "maze_arm_control/reference.hpp"
#include "test_support.hpp"

using maze_arm_control::ReferenceShaper;

namespace
{
constexpr double kDt = 0.002;
}

TEST(Reference, ClampsToMaxTilt)
{
  ReferenceShaper ref(test_support::reference_params());
  ref.reset({0, 0});
  EXPECT_TRUE(ref.set_command({0.5, -0.5}));
  EXPECT_DOUBLE_EQ(ref.command()[0], 0.087);
  EXPECT_DOUBLE_EQ(ref.command()[1], -0.087);
  for (int i = 0; i < 2000; ++i) {ref.step(kDt);}
  EXPECT_NEAR(ref.value()[0], 0.087, 1e-6);
  EXPECT_NEAR(ref.value()[1], -0.087, 1e-6);
}

TEST(Reference, IgnoresNonFinite)
{
  ReferenceShaper ref(test_support::reference_params());
  ref.reset({0.01, 0.02});
  EXPECT_FALSE(ref.set_command({std::nan(""), 0.0}));
  EXPECT_FALSE(ref.set_command({0.0, std::numeric_limits<double>::infinity()}));
  EXPECT_EQ(ref.command(), Eigen::Vector2d(0.01, 0.02));
  ref.step(kDt);
  EXPECT_TRUE(ref.value().allFinite());
}

TEST(Reference, RateNeverExceedsLimit)
{
  ReferenceShaper ref(test_support::reference_params());
  ref.reset({0, 0});
  ref.set_command({0.087, -0.087});
  double max_rate = 0.0;
  for (int i = 0; i < 1000; ++i) {
    ref.step(kDt);
    max_rate = std::max(max_rate, ref.rate().cwiseAbs().maxCoeff());
  }
  EXPECT_LE(max_rate, 0.5 + 1e-9);
  EXPECT_GT(max_rate, 0.49);  // a full-range step does hit it
}

TEST(Reference, StartsFromMeasuredTilt)
{
  ReferenceShaper ref(test_support::reference_params());
  ref.reset({0.03, -0.02});
  ref.step(kDt);
  EXPECT_NEAR(ref.value()[0], 0.03, 1e-12);
  EXPECT_NEAR(ref.value()[1], -0.02, 1e-12);
  EXPECT_EQ(ref.rate(), Eigen::Vector2d::Zero());
}

TEST(Reference, SmoothsSixtyHertzStaircase)
{
  // A 0.5 Hz, 5 deg sine sent at 60 Hz (zero-order hold): unfiltered, the rate would jump by
  // ~(step/dt) every 1/60 s. Filtered, consecutive rates differ by a small amount.
  ReferenceShaper ref(test_support::reference_params());
  ref.reset({0, 0});
  const double amp = 5.0 * M_PI / 180.0;
  double max_jump = 0.0;
  Eigen::Vector2d prev_rate = Eigen::Vector2d::Zero();
  for (int k = 0; k < 2000; ++k) {
    const double t = k * kDt;
    ref.set_command({amp * std::sin(M_PI * std::floor(t * 60.0) / 60.0), 0.0});
    ref.step(kDt);
    if (k > 0) {max_jump = std::max(max_jump, (ref.rate() - prev_rate).cwiseAbs().maxCoeff());}
    prev_rate = ref.rate();
  }
  // one 60 Hz step of this sine is at most amp*pi/60 = 4.6 mrad. Unfiltered the rate would jump
  // by step/dt = 2.3 rad/s (to the 0.5 rad/s limit); through the low-pass by step*(1-exp(-dt/tau))/dt,
  // 0.29 rad/s at tau = 15 ms
  EXPECT_LT(max_jump, 0.3);
}
