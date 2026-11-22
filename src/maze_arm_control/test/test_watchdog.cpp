#include <gtest/gtest.h>

#include "maze_arm_control/watchdog.hpp"

using maze_arm_control::Watchdog;

TEST(Watchdog, NotOkBeforeAnything)
{
  Watchdog w(0.2, 0.02);
  EXPECT_FALSE(w.ok(0.0));
  w.on_state(0.0);
  EXPECT_FALSE(w.ok(0.0));  // no command yet
  EXPECT_NE(w.reason(0.0).find("tilt_cmd"), std::string::npos);
}

TEST(Watchdog, CommandTimeout)
{
  Watchdog w(0.2, 0.02);
  w.on_command(1.0);
  for (double t = 1.0; t < 1.19; t += 0.01) {
    w.on_state(t);
    EXPECT_TRUE(w.ok(t)) << t;
  }
  w.on_state(1.21);
  EXPECT_FALSE(w.ok(1.21));
  w.on_command(1.22);
  EXPECT_TRUE(w.ok(1.22));
}

TEST(Watchdog, StateTimeout)
{
  Watchdog w(0.2, 0.02);
  w.on_command(1.0);
  w.on_state(1.0);
  EXPECT_TRUE(w.ok(1.019));
  EXPECT_FALSE(w.ok(1.021));
  EXPECT_NE(w.reason(1.021).find("joint states"), std::string::npos);
}

TEST(Watchdog, ClearForgetsEverything)
{
  Watchdog w(0.2, 0.02);
  w.on_command(1.0);
  w.on_state(1.0);
  w.clear();
  EXPECT_FALSE(w.ok(1.0));
}
