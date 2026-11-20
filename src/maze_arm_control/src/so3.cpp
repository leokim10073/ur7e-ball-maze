#include "maze_arm_control/so3.hpp"

#include <algorithm>
#include <cmath>

namespace maze_arm_control
{

Eigen::Vector3d log_so3(const Eigen::Matrix3d & R)
{
  const Eigen::AngleAxisd aa(R);  // angle in [0, pi], stable near 0 and pi
  return aa.angle() * aa.axis();
}

Eigen::Matrix3d tilt_rotation(double tilt_x, double tilt_y)
{
  return (Eigen::AngleAxisd(tilt_x, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(-tilt_y, Eigen::Vector3d::UnitX())).toRotationMatrix();
}

Eigen::Vector2d tilt_of(const Eigen::Matrix3d & R_home, const Eigen::Matrix3d & R)
{
  const Eigen::Vector3d rv = log_so3(R_home.transpose() * R);  // board frame
  return {rv.y(), -rv.x()};
}

double yaw_of(const Eigen::Matrix3d & R_home, const Eigen::Matrix3d & R)
{
  return log_so3(R_home.transpose() * R).z();
}

}  // namespace maze_arm_control
