#include "parameter/rl_mimic_param.h"

#include <iostream>
#include <stdexcept>

namespace example {

RlMimicParam::RlMimicParam(const std::string& config_file) {
  LoadFromYaml(config_file);

  // Initialize derived parameters
  num_actions = active_joint_names.size();
  if (num_observations != 5 * num_actions + 9) {
    throw std::runtime_error(
        "num_observations=" + std::to_string(num_observations) + " does not match expected " +
        std::to_string(5 * num_actions + 9) + " (5 * num_actions + 9)");
  }

  // Observation order:
  //   command        (2 * num_actions): ref joint_pos + ref joint_vel
  //   anchor_ori_b   (6):               first 2 cols of relative-orientation rotation matrix
  //   base_ang_vel   (3):               angular velocity in body frame
  //   joint_pos      (num_actions):     q - q_default
  //   joint_vel      (num_actions):     qd
  //   actions        (num_actions):     last action
  observation_scale = Eigen::VectorXd::Zero(num_observations);
  observation_scale << Eigen::VectorXd::Ones(2 * num_actions),                              // command
      Eigen::VectorXd::Ones(6),                                                              // motion_anchor_ori_b
      Eigen::VectorXd::Constant(3, observation_scale_angular_vel),                          // base_ang_vel
      Eigen::VectorXd::Constant(num_actions, observation_scale_dof_pos),                    // joint_pos
      Eigen::VectorXd::Constant(num_actions, observation_scale_dof_vel),                    // joint_vel
      Eigen::VectorXd::Ones(num_actions);                                                    // actions
}

void RlMimicParam::LoadFromYaml(const std::string& config_file) {
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

    // Load motion parameters
    motion_file = config["motion_file"].as<std::string>();
    motion_yaw_alignment = config["motion_yaw_alignment"].as<bool>(true);
    motion_start_frame = config["motion_start_frame"].as<int>(0);
    motion_end_frame = config["motion_end_frame"].as<int>(-1);

    // Load safety parameters
    anchor_ori_termination_threshold_rad =
        config["anchor_ori_termination_threshold_rad"].as<double>(0.6);

  } catch (const YAML::Exception& e) {
    std::cerr << "Error loading YAML file: " << e.what() << std::endl;
    throw;
  }
  std::cout << "RlMimicParam::LoadFromYaml done" << std::endl;
}

Eigen::VectorXd RlMimicParam::LoadVectorFromYaml(const YAML::Node& node) {
  std::vector<double> vec;
  for (const auto& item : node) {
    vec.push_back(item.as<double>());
  }
  return Eigen::Map<Eigen::VectorXd>(vec.data(), vec.size());
}

Eigen::VectorXi RlMimicParam::LoadIntVectorFromYaml(const YAML::Node& node) {
  std::vector<int> vec;
  for (const auto& item : node) {
    vec.push_back(item.as<int>());
  }
  return Eigen::Map<Eigen::VectorXi>(vec.data(), vec.size());
}

std::vector<Eigen::VectorXd> RlMimicParam::LoadVectorArrayFromYaml(const YAML::Node& node) {
  std::vector<Eigen::VectorXd> result;
  for (const auto& item : node) {
    result.push_back(LoadVectorFromYaml(item));
  }
  return result;
}

}  // namespace example
