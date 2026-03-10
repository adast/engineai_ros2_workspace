#include "ros_interface.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <rclcpp/logging.hpp>
#include <stdexcept>
#include <string>

#include "config_loader.h"
#include "rclcpp/rclcpp.hpp"

// Constants
const int kDofFloatingBase = 6;        // Number of DoF for floating base
const int kNumFloatingBaseJoints = 7;  // Number of joints for floating base (quaternion + xyz)

namespace mujoco {

RosInterface::RosInterface(const rclcpp::Node::SharedPtr& node, std::shared_ptr<ConfigLoader> config_loader)
    : node_(node), config_loader_(config_loader), model_(nullptr), data_(nullptr), is_floating_base_(false) {}

RosInterface::~RosInterface() {}

bool RosInterface::Initialize() {
  // Create publishers
  joint_state_pub_ =
      node_->create_publisher<interface_protocol::msg::JointState>(config_loader_->GetJointStateTopic(), 10);

  imu_pub_ = node_->create_publisher<interface_protocol::msg::ImuInfo>(config_loader_->GetImuTopic(), 10);
  
  // Create publisher for motion state
  motion_state_pub_ = node_->create_publisher<interface_protocol::msg::MotionState>("/motion/motion_state", 10);

  // Create subscriber with more compatible QoS settings
  using std::placeholders::_1;

  // 创建更兼容的QoS设置
  auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();

  joint_cmd_sub_ = node_->create_subscription<interface_protocol::msg::JointCommand>(
      config_loader_->GetJointCommandTopic(), qos, std::bind(&RosInterface::JointCommandCallback, this, _1));

  // Get number of joints from config loader
  num_total_joints_ = config_loader_->GetNumTotalJoints();

  // Initialize commanded values with zeros
  joint_command_.position.resize(num_total_joints_, 0.0);
  joint_command_.velocity.resize(num_total_joints_, 0.0);
  joint_command_.torque.resize(num_total_joints_, 0.0);
  joint_command_.feed_forward_torque.resize(num_total_joints_, 0.0);
  joint_command_.stiffness.resize(num_total_joints_, 0.0);
  joint_command_.damping.resize(num_total_joints_, 0.0);

  // Create timer for publishing motion state every 1 second
  motion_state_timer_ = node_->create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&RosInterface::MotionStateTimerCallback, this));

  RCLCPP_INFO(node_->get_logger(), "MuJoCo ROS interface initialized successfully");
  return true;
}

interface_protocol::msg::JointCommand RosInterface::GetCommandedSafe() {
  std::lock_guard<std::mutex> lock(mtx_);
  return joint_command_;
}

void RosInterface::JointCommandCallback(const interface_protocol::msg::JointCommand::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(mtx_);

  // Update commanded values
  joint_command_ = *msg;

  // Ensure all vectors are properly sized
  if (joint_command_.position.size() > num_total_joints_) {
    joint_command_.position.resize(num_total_joints_);
  }
  if (joint_command_.velocity.size() > num_total_joints_) {
    joint_command_.velocity.resize(num_total_joints_);
  }
  if (joint_command_.torque.size() > num_total_joints_) {
    joint_command_.torque.resize(num_total_joints_);
  }
  if (joint_command_.feed_forward_torque.size() > num_total_joints_) {
    joint_command_.feed_forward_torque.resize(num_total_joints_);
  }
  if (joint_command_.stiffness.size() > num_total_joints_) {
    joint_command_.stiffness.resize(num_total_joints_);
  }
  if (joint_command_.damping.size() > num_total_joints_) {
    joint_command_.damping.resize(num_total_joints_);
  }
}

void RosInterface::UpdateSimState(const mjModel* m, mjData* d) {
  is_floating_base_ = (m->nv != m->nu);

  // Create messages
  auto joint_state_msg = std::make_unique<interface_protocol::msg::JointState>();
  auto imu_msg = std::make_unique<interface_protocol::msg::ImuInfo>();

  // Set timestamp
  joint_state_msg->header.stamp = node_->now();
  imu_msg->header.stamp = node_->now();

  // Set joint states
  joint_state_msg->position.resize(num_total_joints_);
  joint_state_msg->velocity.resize(num_total_joints_);
  joint_state_msg->torque.resize(num_total_joints_);

  if (is_floating_base_) {
    // Skip the floating base joints
    for (int i = 0; i < num_total_joints_; ++i) {
      joint_state_msg->position[i] = d->qpos[i + kNumFloatingBaseJoints];
      joint_state_msg->velocity[i] = d->qvel[i + kDofFloatingBase];
      joint_state_msg->torque[i] = d->actuator_force[i];
    }
  } else {
    for (int i = 0; i < num_total_joints_; ++i) {
      joint_state_msg->position[i] = d->qpos[i];
      joint_state_msg->velocity[i] = d->qvel[i];
      joint_state_msg->torque[i] = d->actuator_force[i];
    }
  }

  // IMU quaternion (framequat at imu site)
  const double qw = d->sensordata[imu_quat_adr_ + 0];
  const double qx = d->sensordata[imu_quat_adr_ + 1];
  const double qy = d->sensordata[imu_quat_adr_ + 2];
  const double qz = d->sensordata[imu_quat_adr_ + 3];
  imu_msg->quaternion.w = qw;
  imu_msg->quaternion.x = qx;
  imu_msg->quaternion.y = qy;
  imu_msg->quaternion.z = qz;

  // RPY derived from quaternion (ZYX convention); used by plotjuggler visualization
  imu_msg->rpy.x = std::atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy));
  imu_msg->rpy.y = std::asin(std::clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0));
  imu_msg->rpy.z = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));

  // Linear acceleration (accelerometer, body frame)
  imu_msg->linear_acceleration.x = d->sensordata[imu_accel_adr_ + 0];
  imu_msg->linear_acceleration.y = d->sensordata[imu_accel_adr_ + 1];
  imu_msg->linear_acceleration.z = d->sensordata[imu_accel_adr_ + 2];

  // Angular velocity (gyro, body frame)
  imu_msg->angular_velocity.x = d->sensordata[imu_gyro_adr_ + 0];
  imu_msg->angular_velocity.y = d->sensordata[imu_gyro_adr_ + 1];
  imu_msg->angular_velocity.z = d->sensordata[imu_gyro_adr_ + 2];

  // Publish messages
  joint_state_pub_->publish(std::move(joint_state_msg));
  imu_pub_->publish(std::move(imu_msg));
}

void RosInterface::SetModelAndData(mjModel* model, mjData* data) {
  model_ = model;
  data_ = data;

  auto lookup = [&](const char* name) {
    int id = mj_name2id(model, mjOBJ_SENSOR, name);
    if (id == -1) throw std::runtime_error(std::string("IMU sensor not found in model: ") + name);
    return model->sensor_adr[id];
  };
  imu_quat_adr_  = lookup("imu_quaternion");
  imu_accel_adr_ = lookup("imu_linear_acceleration");
  imu_gyro_adr_  = lookup("imu_angular_velocity");
}

void RosInterface::MotionStateTimerCallback() {
  // Create a motion state message
  auto motion_state_msg = std::make_unique<interface_protocol::msg::MotionState>();
  
  // Set the current_motion_task field to "joint_bridge"
  motion_state_msg->current_motion_task = "joint_bridge";
  
  // Publish the message
  motion_state_pub_->publish(std::move(motion_state_msg));
}

}  // namespace mujoco
