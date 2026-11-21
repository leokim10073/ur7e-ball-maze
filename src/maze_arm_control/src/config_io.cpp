// YAML loaders for the arm config and the UR kinematics calibration file.
#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

#include "maze_arm_control/kinematics.hpp"
#include "maze_arm_control/tilt_controller.hpp"

namespace maze_arm_control
{

namespace
{
YAML::Node load_file(const std::string& path)
{
  try
  {
    return YAML::LoadFile(path);
  }
  catch (const YAML::Exception& e)
  {
    throw std::runtime_error(path + ": " + e.what());
  }
}

double positive(const YAML::Node& root, const std::string& path, const std::string& key)
{
  if (!root[key])
  {
    throw std::runtime_error(path + ": missing key '" + key + "'");
  }
  const double v = root[key].as<double>();
  if (!(v > 0.0))
  {
    throw std::runtime_error(path + ": '" + key + "' must be > 0, got " + std::to_string(v));
  }
  return v;
}
}  // namespace

ArmConfig load_arm_config(const std::string& path)
{
  const YAML::Node root = load_file(path);
  return ArmConfig{ positive(root, path, "max_tilt"),  positive(root, path, "max_tilt_rate"),
                    positive(root, path, "max_joint_speed"), positive(root, path, "k_pos"),
                    positive(root, path, "k_rot"),     positive(root, path, "damping"),
                    positive(root, path, "command_timeout") };
}

UrKinematics load_ur_kinematics(const std::string& path)
{
  const YAML::Node k = load_file(path)["kinematics"];
  if (!k)
  {
    throw std::runtime_error(path + ": missing key 'kinematics'");
  }
  UrKinematics kin;
  const char* links[6] = { "shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3" };
  for (int i = 0; i < 6; ++i)
  {
    const YAML::Node j = k[links[i]];
    for (const char* key : { "x", "y", "z", "roll", "pitch", "yaw" })
    {
      if (!j || !j[key])
      {
        throw std::runtime_error(path + ": missing kinematics." + links[i] + "." + key);
      }
    }
    kin.joint_origin[i] = transform(Eigen::Vector3d(j["x"].as<double>(), j["y"].as<double>(), j["z"].as<double>()),
                                    rpy_matrix(j["roll"].as<double>(), j["pitch"].as<double>(), j["yaw"].as<double>()));
  }
  return kin;
}

}  // namespace maze_arm_control
