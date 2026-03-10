#include <fstream>
#include <memory>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "components/message_handler.hpp"
#include "math/concatenate_vector.h"
#include "math/mnn_model.h"
#include "parameter/rl_mimic_param.h"

using namespace std::chrono_literals;

namespace example {

// ---------------------------------------------------------------------------
// Motion data loaded from the exported mocap CSV file.
// CSV columns (55 total):
//   [0-2]   anchor_pos_w (x, y, z)
//   [3-6]   anchor_quat_w (w, x, y, z)
//   [7-30]  joint_pos for 24 active joints (in active_joint_names order)
//   [31-54] joint_vel for 24 active joints (in active_joint_names order)
// ---------------------------------------------------------------------------
class MotionLoader {
 public:
  bool Load(const std::string& csv_file) {
    std::ifstream file(csv_file);
    if (!file.is_open()) return false;

    std::string line;
    // Skip header line
    std::getline(file, line);

    while (std::getline(file, line)) {
      if (line.empty()) continue;
      std::stringstream ss(line);
      std::string token;
      std::vector<double> vals;
      while (std::getline(ss, token, ',')) {
        vals.push_back(std::stod(token));
      }
      if (static_cast<int>(vals.size()) < kCsvNumCols) continue;

      anchor_quat_w_.emplace_back(vals[3], vals[4], vals[5], vals[6]);  // (w, x, y, z)

      const int num_joints_half = (static_cast<int>(vals.size()) - kJointStartCol) / 2;
      Eigen::VectorXd jpos(num_joints_half), jvel(num_joints_half);
      for (int i = 0; i < num_joints_half; ++i) {
        jpos[i] = vals[kJointStartCol + i];
        jvel[i] = vals[kJointStartCol + num_joints_half + i];
      }
      joint_pos_.push_back(jpos);
      joint_vel_.push_back(jvel);
    }

    num_frames_ = static_cast<int>(anchor_quat_w_.size());
    return num_frames_ > 0;
  }

  int NumFrames() const { return num_frames_; }
  const Eigen::Quaterniond& AnchorQuatW(int idx) const { return anchor_quat_w_[idx]; }
  const Eigen::VectorXd& JointPos(int idx) const { return joint_pos_[idx]; }
  const Eigen::VectorXd& JointVel(int idx) const { return joint_vel_[idx]; }

 private:
  static constexpr int kJointStartCol = 7;   // first joint_pos column index
  static constexpr int kCsvNumCols = 55;     // expected minimum number of columns

  int num_frames_ = 0;
  std::vector<Eigen::Quaterniond> anchor_quat_w_;
  std::vector<Eigen::VectorXd> joint_pos_;
  std::vector<Eigen::VectorXd> joint_vel_;
};

// ---------------------------------------------------------------------------
// ROS2 node that runs the RlMimic motion-tracking policy.
//
// Observation (129 values, num_include_obs_steps = 1):
//   [0:48]   command:            ref_joint_pos[24] + ref_joint_vel[24]
//   [48:54]  motion_anchor_ori_b: first 2 cols of R_rel (robot_anchor.inv * motion_anchor)
//   [54:57]  base_ang_vel:       IMU angular velocity in body frame
//   [57:81]  joint_pos:          q_real - q_default  (active joints)
//   [81:105] joint_vel:          qd_real             (active joints)
//   [105:129] actions:           last network action
// ---------------------------------------------------------------------------
class RlMimicRunner : public rclcpp::Node {
 public:
  explicit RlMimicRunner(const std::string& config_file_dir) : Node("rl_mimic_runner") {
    std::string config_file = config_file_dir + "/rl_params.yaml";
    param_ = std::make_shared<RlMimicParam>(config_file);
    config_file_dir_ = config_file_dir;
    joint_command_ = std::make_shared<interface_protocol::msg::JointCommand>();
  }

  bool Initialize() {
    try {
      // Initialize message handler
      message_handler_ = std::make_shared<MessageHandler>(shared_from_this());
      message_handler_->Initialize();

      // Wait for joint_bridge state
      while (!message_handler_->GetLatestMotionState() ||
             message_handler_->GetLatestMotionState()->current_motion_task != "joint_bridge") {
        rclcpp::spin_some(shared_from_this());
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Waiting for joint bridge state...");
      }
      RCLCPP_INFO(get_logger(), "Already in joint bridge state");

      // Get initial joint positions
      auto initial_state = message_handler_->GetLatestJointState();
      if (!initial_state) {
        RCLCPP_ERROR(get_logger(), "Failed to get initial joint state");
        return false;
      }
      initial_joint_q_ =
          Eigen::Map<const Eigen::VectorXd>(initial_state->position.data(), initial_state->position.size());
      RCLCPP_INFO_STREAM(get_logger(), "Initial joint positions: " << initial_joint_q_.transpose());

      // Concatenate joint parameters from yaml
      active_joint_idx_ = param_->active_joint_idx;
      default_joint_q_ = math::ConcatenateVectors(param_->default_joint_q);
      joint_kp_ = math::ConcatenateVectors(param_->joint_kp);
      joint_kd_ = math::ConcatenateVectors(param_->joint_kd);
      action_scale_ = math::ConcatenateVectors(param_->action_scale);

      // Load motion reference data
      std::string motion_file = config_file_dir_ + "/" + param_->motion_file;
      if (!motion_loader_.Load(motion_file)) {
        RCLCPP_ERROR(get_logger(), "Failed to load motion file: %s", motion_file.c_str());
        return false;
      }
      RCLCPP_INFO(get_logger(), "Loaded motion file with %d frames", motion_loader_.NumFrames());

      // Initialize MNN model
      mlp_net_ = std::make_unique<math::MnnModel>(config_file_dir_ + "/" + param_->policy_file);
      mlp_net_action_.setZero(param_->num_actions);

      // Initialize control variables
      time_ = 0.0;
      motion_idx_ = 0;
      is_first_time_ = true;

      RCLCPP_INFO(get_logger(), "Starting control loop");
      control_timer_ = create_wall_timer(std::chrono::duration<double>(param_->control_dt),
                                         std::bind(&RlMimicRunner::ControlCallback, this));
      return true;
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Failed to initialize: %s", e.what());
      return false;
    }
  }

 private:
  void ControlCallback() {
    if (message_handler_->GetLatestMotionState()->current_motion_task != "joint_bridge") {
      time_ = 0.0;
      motion_idx_ = 0;
      is_first_time_ = true;
      return;
    }
    auto joint_state = message_handler_->GetLatestJointState();
    if (!joint_state) return;

    UpdateState(joint_state);
    CalculateObservation();
    CalculateMotorCommand();
    SendMotorCommand();

    time_ += param_->control_dt;
    motion_idx_ = (motion_idx_ + 1) % motion_loader_.NumFrames();
  }

  void UpdateState(const interface_protocol::msg::JointState::SharedPtr& joint_state) {
    q_real_ = Eigen::Map<const Eigen::VectorXd>(joint_state->position.data(), joint_state->position.size());
    qd_real_ = Eigen::Map<const Eigen::VectorXd>(joint_state->velocity.data(), joint_state->velocity.size());
  }

  void CalculateObservation() {
    if (is_first_time_) {
      is_first_time_ = false;
      mlp_net_action_.setZero(param_->num_actions);
    }

    auto imu = message_handler_->GetLatestImu();

    // Robot anchor orientation in world frame (from IMU)
    Eigen::Quaterniond q_robot(imu->quaternion.w, imu->quaternion.x, imu->quaternion.y, imu->quaternion.z);
    q_robot.normalize();

    // Angular velocity in body frame (from IMU)
    Eigen::Vector3d w_real(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z);

    // Reference motion data at current time step
    Eigen::Quaterniond q_motion = motion_loader_.AnchorQuatW(motion_idx_);
    q_motion.normalize();
    const Eigen::VectorXd& ref_joint_pos = motion_loader_.JointPos(motion_idx_);
    const Eigen::VectorXd& ref_joint_vel = motion_loader_.JointVel(motion_idx_);

    // motion_anchor_ori_b:
    //   q_rel = q_robot.inv * q_motion  (orientation of motion anchor in robot anchor frame)
    //   Represent as first 2 columns of the rotation matrix, row-interleaved:
    //   [R(0,0), R(0,1), R(1,0), R(1,1), R(2,0), R(2,1)] = 6 values
    Eigen::Quaterniond q_rel = q_robot.inverse() * q_motion;
    q_rel.normalize();
    Eigen::Matrix3d R_rel = q_rel.toRotationMatrix();
    Eigen::VectorXd anchor_ori_b(6);
    for (int row = 0; row < 3; ++row) {
      anchor_ori_b[row * 2 + 0] = R_rel(row, 0);
      anchor_ori_b[row * 2 + 1] = R_rel(row, 1);
    }

    // Build the 129-dim observation vector
    //   command        (48): ref_joint_pos[24] + ref_joint_vel[24]
    //   anchor_ori_b   ( 6): rotation matrix cols 0-1 of q_robot.inv * q_motion
    //   base_ang_vel   ( 3): IMU angular velocity in body frame
    //   joint_pos      (24): q_real - q_default  (active joints)
    //   joint_vel      (24): qd_real             (active joints)
    //   actions        (24): last network output
    mlp_net_observation_ = Eigen::VectorXd::Zero(param_->num_observations);
    mlp_net_observation_ << ref_joint_pos,                                               // 24
        ref_joint_vel,                                                                    // 24
        anchor_ori_b,                                                                     //  6
        w_real,                                                                           //  3
        (q_real_ - default_joint_q_)(active_joint_idx_),                                 // 24
        qd_real_(active_joint_idx_),                                                      // 24
        mlp_net_action_;                                                                  // 24

    // Scale and clip
    mlp_net_observation_.array() *= param_->observation_scale.array();
    mlp_net_observation_ =
        mlp_net_observation_.cwiseMax(-param_->observation_clip).cwiseMin(param_->observation_clip);
  }

  void CalculateMotorCommand() {
    // Feed the observation directly to the network (num_include_obs_steps = 1)
    mlp_net_action_ = mlp_net_->Inference(mlp_net_observation_.cast<float>()).cast<double>();
    mlp_net_action_ = mlp_net_action_.cwiseMax(-param_->action_clip).cwiseMin(param_->action_clip);

    q_des_ = default_joint_q_;
    q_des_(active_joint_idx_) += mlp_net_action_.cwiseProduct(action_scale_);

    if (time_ < param_->transition_time) {
      double ratio = time_ / param_->transition_time;
      q_des_ = ratio * q_des_ + (1.0 - ratio) * initial_joint_q_;
    }
  }

  void SendMotorCommand() {
    joint_command_->position = std::vector<double>(q_des_.data(), q_des_.data() + q_des_.size());
    joint_command_->velocity = std::vector<double>(q_des_.size(), 0.0);
    joint_command_->feed_forward_torque = std::vector<double>(q_des_.size(), 0.0);
    joint_command_->torque = std::vector<double>(q_des_.size(), 0.0);
    joint_command_->stiffness = std::vector<double>(joint_kp_.data(), joint_kp_.data() + joint_kp_.size());
    joint_command_->damping = std::vector<double>(joint_kd_.data(), joint_kd_.data() + joint_kd_.size());
    joint_command_->parallel_parser_type = interface_protocol::msg::ParallelParserType::RL_PARSER;
    message_handler_->PublishJointCommand(*joint_command_);
  }

  // Parameters
  std::shared_ptr<RlMimicParam> param_;

  // Message handling
  std::shared_ptr<MessageHandler> message_handler_;

  // MNN model
  std::unique_ptr<math::MnnModel> mlp_net_;
  Eigen::VectorXd mlp_net_observation_;
  Eigen::VectorXd mlp_net_action_;

  // Motion reference data
  MotionLoader motion_loader_;
  int motion_idx_;

  // State variables
  double time_;
  bool is_first_time_;
  Eigen::VectorXd q_real_;
  Eigen::VectorXd qd_real_;
  Eigen::VectorXd q_des_;
  Eigen::VectorXi active_joint_idx_;
  Eigen::VectorXd initial_joint_q_;
  Eigen::VectorXd default_joint_q_;
  Eigen::VectorXd joint_kp_;
  Eigen::VectorXd joint_kd_;
  Eigen::VectorXd action_scale_;

  // ROS timer
  rclcpp::TimerBase::SharedPtr control_timer_;
  std::string config_file_dir_;
  interface_protocol::msg::JointCommand::SharedPtr joint_command_;
};

}  // namespace example

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  if (argc < 2) {
    RCLCPP_ERROR(rclcpp::get_logger("rl_mimic_example"), "Usage: rl_mimic_example <config_file_dir>");
    return 1;
  }

  auto node = std::make_shared<example::RlMimicRunner>(argv[1]);
  if (!node->Initialize()) {
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
