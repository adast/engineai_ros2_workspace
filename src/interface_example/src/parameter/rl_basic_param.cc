#include "parameter/rl_basic_param.h"

#include <iostream>

namespace example {

RlBasicParam::RlBasicParam(const std::string& config_file) {
  LoadFromYaml(config_file);

  // Initialize derived parameters
  num_actions = active_joint_names.size();

  // Initialize observation scale vector with correct size
  observation_scale = Eigen::VectorXd::Zero(num_observations);

  // Order: ang_vel, gravity, commands, jpos, jvel, actions
  observation_scale <<
      Eigen::VectorXd::Constant(3, observation_scale_angular_vel),       // base angular velocity
      Eigen::VectorXd::Constant(3, 1.0),                                 // projected gravity
      Eigen::VectorXd::Constant(3, 1.0),                                 // commands
      Eigen::VectorXd::Constant(num_actions, observation_scale_dof_pos), // joint positions
      Eigen::VectorXd::Constant(num_actions, observation_scale_dof_vel), // joint velocities
      Eigen::VectorXd::Ones(num_actions);                                // last action
}

void RlBasicParam::LoadFromYaml(const std::string& config_file) {
  try {
    YAML::Node config = YAML::LoadFile(config_file);
    // Load MLP net parameters
    policy_file = config["policy_file"].as<std::string>();
    num_observations = config["num_observations"].as<int>();
    active_joint_names = config["active_joint_names"].as<std::vector<std::string>>();
    active_joint_idx = LoadIntVectorFromYaml(config["active_joint_idx"]);
    num_include_obs_steps = config["num_include_obs_steps"].as<int>();

    // Load observation parameters
    observation_scale_angular_vel = config["observation_scale_angular_vel"].as<double>();
    observation_scale_dof_pos = config["observation_scale_dof_pos"].as<double>();
    observation_scale_dof_vel = config["observation_scale_dof_vel"].as<double>();
    observation_clip = config["observation_clip"].as<double>();

    transition_time = config["transition_time"].as<double>();
    // Load joint control parameters
    action_clip = config["action_clip"].as<double>();
    default_joint_q = LoadVectorArrayFromYaml(config["default_joint_q"]);
    joint_kp = LoadVectorArrayFromYaml(config["joint_kp"]);
    joint_kd = LoadVectorArrayFromYaml(config["joint_kd"]);
    action_scale = LoadVectorArrayFromYaml(config["action_scale"]);
    control_dt = config["control_dt"].as<double>();
    imu_tilt_termination_threshold_rad = config["imu_tilt_termination_threshold_rad"].as<double>(-1.0);
    // Load command scale
    auto command_scale_node = config["command_scale"];
    command_scale = Eigen::Vector3d(command_scale_node[0].as<double>(), command_scale_node[1].as<double>(),
                                    command_scale_node[2].as<double>());

  } catch (const YAML::Exception& e) {
    std::cerr << "Error loading YAML file: " << e.what() << std::endl;
    throw;
  }
  std::cout << "LoadFromYaml done" << std::endl;
}

Eigen::VectorXd RlBasicParam::LoadVectorFromYaml(const YAML::Node& node) {
  std::vector<double> vec;
  for (const auto& item : node) {
    vec.push_back(item.as<double>());
  }
  return Eigen::Map<Eigen::VectorXd>(vec.data(), vec.size());
}

Eigen::VectorXi RlBasicParam::LoadIntVectorFromYaml(const YAML::Node& node) {
  std::vector<int> vec;
  for (const auto& item : node) {
    vec.push_back(item.as<int>());
  }
  return Eigen::Map<Eigen::VectorXi>(vec.data(), vec.size());
}
std::vector<Eigen::VectorXd> RlBasicParam::LoadVectorArrayFromYaml(const YAML::Node& node) {
  std::vector<Eigen::VectorXd> result;
  for (const auto& item : node) {
    result.push_back(LoadVectorFromYaml(item));
  }
  return result;
}

}  // namespace example