#include "maze_arm_control/kinematics.hpp"

#include <cmath>
#include <stdexcept>

namespace maze_arm_control
{

Eigen::Matrix3d rpy_matrix(double roll, double pitch, double yaw)
{
  return (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
      .toRotationMatrix();
}

Eigen::Isometry3d transform(const Eigen::Vector3d& xyz, const Eigen::Matrix3d& R)
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = R;
  T.translation() = xyz;
  return T;
}

namespace
{
Eigen::Isometry3d rot_z(double a)
{
  return transform(Eigen::Vector3d::Zero(), Eigen::AngleAxisd(a, Eigen::Vector3d::UnitZ()).toRotationMatrix());
}

// Fixed frames of ur_description's ur_macro.xacro.
const Eigen::Isometry3d BASE_LINK_TO_INERTIA = rot_z(M_PI);  // base_link is REP-103, the chain is not
const Eigen::Isometry3d WRIST_3_TO_TOOL0 = transform(Eigen::Vector3d::Zero(), rpy_matrix(0, -M_PI / 2, -M_PI / 2)) *
                                           transform(Eigen::Vector3d::Zero(), rpy_matrix(M_PI / 2, 0, M_PI / 2));
}  // namespace

std::array<Eigen::Isometry3d, 7> joint_frames(const UrKinematics& kin, const Vector6d& q)
{
  std::array<Eigen::Isometry3d, 7> frames;
  Eigen::Isometry3d T = BASE_LINK_TO_INERTIA;
  for (int i = 0; i < 6; ++i)
  {
    T = T * kin.joint_origin[i];
    frames[i] = T;
    T = T * rot_z(q[i]);
  }
  frames[6] = T * WRIST_3_TO_TOOL0;
  return frames;
}

Eigen::Isometry3d fk_tool0(const UrKinematics& kin, const Vector6d& q)
{
  return joint_frames(kin, q)[6];
}

Eigen::Isometry3d board_offset(const Eigen::Vector3d& grasp_xyz, const Eigen::Vector3d& grasp_rpy,
                               const Eigen::Vector3d& handle_center)
{
  const Eigen::Isometry3d T_tool0_handle = transform(grasp_xyz, rpy_matrix(grasp_rpy[0], grasp_rpy[1], grasp_rpy[2]));
  const Eigen::Isometry3d T_handle_board = transform(-handle_center);  // board axes = handle axes
  return T_tool0_handle * T_handle_board;
}

Matrix6d board_jacobian(const UrKinematics& kin, const Vector6d& q, const Eigen::Isometry3d& T_tool0_board)
{
  const auto frames = joint_frames(kin, q);
  const Eigen::Vector3d p = (frames[6] * T_tool0_board).translation();
  Matrix6d J;
  for (int i = 0; i < 6; ++i)
  {
    const Eigen::Vector3d z = frames[i].linear().col(2);
    const Eigen::Vector3d o = frames[i].translation();
    J.block<3, 1>(0, i) = z.cross(p - o);
    J.block<3, 1>(3, i) = z;
  }
  return J;
}

Vector6d joints_from_state(const std::vector<std::string>& names, const std::vector<double>& positions)
{
  Vector6d q;
  std::string missing;
  for (int i = 0; i < 6; ++i)
  {
    bool found = false;
    for (size_t k = 0; k < names.size() && k < positions.size(); ++k)
    {
      if (names[k] == UR_JOINTS[i])
      {
        q[i] = positions[k];
        found = true;
        break;
      }
    }
    if (!found)
    {
      missing += (missing.empty() ? "" : ", ") + UR_JOINTS[i];
    }
  }
  if (!missing.empty())
  {
    throw std::runtime_error("joint state is missing " + missing);
  }
  return q;
}

}  // namespace maze_arm_control
