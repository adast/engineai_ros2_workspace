#pragma once

#include <yaml-cpp/yaml.h>
#include <Eigen/Core>
#include <string>
#include <vector>

namespace example {

class RlBasicParam {
 public:
  explicit RlBasicParam(const std::string& config_file);
  ~RlBasicParam() = default;

  // MLP net parameters
  std::string policy_file;
  int num_observations;
  std::vector<std::string> active_joint_names;
  int num_actions;
  int num_include_obs_steps;
  // Observation parameters
  double observation_scale_angular_vel;
  double observation_scale_dof_pos;
  double observation_scale_dof_vel;
  double observation_clip;
  Eigen::VectorXd observation_scale;
  Eigen::VectorXi active_joint_idx;
  double transition_time;
  // Joint control parameters
  double action_clip;
  std::vector<Eigen::VectorXd> default_joint_q;
  std::vector<Eigen::VectorXd> joint_kp;
  std::vector<Eigen::VectorXd> joint_kd;
  std::vector<Eigen::VectorXd> action_scale;
  double control_dt;
  Eigen::Vector3d command_scale;
  double imu_tilt_termination_threshold_rad;

 private:
  void LoadFromYaml(const std::string& config_file);
  Eigen::VectorXd LoadVectorFromYaml(const YAML::Node& node);
  std::vector<Eigen::VectorXd> LoadVectorArrayFromYaml(const YAML::Node& node);
  Eigen::VectorXi LoadIntVectorFromYaml(const YAML::Node& node);
};

}  // namespace example
