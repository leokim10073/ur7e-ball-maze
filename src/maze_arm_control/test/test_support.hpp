// Test helpers: the kinematics and parameters from the shipped config/tilt_control.yaml.
#pragma once

#include <yaml-cpp/yaml.h>

#include <string>

#include "maze_arm_control/kinematics.hpp"
#include "maze_arm_control/reference.hpp"
#include "maze_arm_control/tilt_controller.hpp"

namespace test_support
{

inline YAML::Node params()
{
  return YAML::LoadFile(MAZE_ARM_CONTROL_CONFIG)["tilt_control"]["ros__parameters"];
}

template<int N>
inline Eigen::Matrix<double, N, 1> vec(const YAML::Node & n)
{
  Eigen::Matrix<double, N, 1> v;
  for (int i = 0; i < N; ++i) {v[i] = n[i].as<double>();}
  return v;
}

inline maze_arm_control::BoardMount sim_mount()
{
  const auto m = params()["mount"];
  return {vec<3>(m["grasp_xyz"]), vec<3>(m["grasp_rpy"]), vec<3>(m["handle_center"])};
}

inline maze_arm_control::DhParams sim_dh()
{
  const auto k = params()["kinematics"];
  maze_arm_control::DhParams dh;
  for (int i = 0; i < 6; ++i) {
    dh.d[i] = k["d"][i].as<double>();
    dh.a[i] = k["a"][i].as<double>();
    dh.alpha[i] = k["alpha"][i].as<double>();
  }
  return dh;
}

inline maze_arm_control::ArmKinematics sim_kinematics() {return {sim_dh(), sim_mount()};}

inline maze_arm_control::Joints home_joints() {return vec<6>(params()["home_joints"]);}

inline maze_arm_control::ReferenceParams reference_params()
{
  const auto p = params();
  return {p["limits"]["max_tilt"].as<double>(), p["limits"]["max_rate"].as<double>(),
    p["reference"]["smoothing_tau"].as<double>()};
}

inline maze_arm_control::ControllerParams controller_params()
{
  const auto p = params();
  maze_arm_control::ControllerParams c;
  c.k_pos = p["gains"]["k_pos"].as<double>();
  c.k_rot = p["gains"]["k_rot"].as<double>();
  c.sigma_min = p["dls"]["sigma_min"].as<double>();
  c.lambda_max = p["dls"]["lambda_max"].as<double>();
  c.max_linear_speed = p["twist_cap"]["linear"].as<double>();
  c.max_angular_speed = p["twist_cap"]["angular"].as<double>();
  c.max_joint_speed = vec<6>(p["joint_limits"]["max_speed"]);
  c.max_joint_accel = vec<6>(p["joint_limits"]["max_accel"]);
  return c;
}

}  // namespace test_support
