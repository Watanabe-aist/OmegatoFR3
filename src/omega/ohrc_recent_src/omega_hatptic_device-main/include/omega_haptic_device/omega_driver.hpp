#ifndef OMEGA_DRIVER_HPP
#define OMEGA_DRIVER_HPP

#include <limits.h>
#include <pthread.h>
#include <rclcpp/rclcpp.hpp>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

#include <algorithm>
#include <functional>
#include <mutex>
#include <string>

#include <Eigen/Geometry>
#include <tf2_eigen/tf2_eigen.hpp>

#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/bool.hpp>

#include "dhdc.h"
#include "drdc.h"
#include "ohrc_msgs/msg/state.hpp"

class OmegaDriver : public rclcpp::Node {
protected:
  rclcpp::Publisher<ohrc_msgs::msg::State>::SharedPtr pub_pose_;
  rclcpp::Publisher<ohrc_msgs::msg::State>::SharedPtr pub_pose2_;

  // ===== 右Omega指への力フィードバック用 =====
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr sub_gripper_force_cmd_;

  double gripper_force_cmd_ = 0.0;
  double gripper_force_gain_ = 0.6;
  double gripper_force_limit_ = 2.0;
  double gripper_force_sign_ = 1.0;

  void gripperForceCallback(const std_msgs::msg::Float64::SharedPtr msg);

  // Experiment 2: right-Omega finger force-feedback gate
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr
    sub_force_feedback_enable_;
  bool force_feedback_enabled_ = true;
  void forceFeedbackEnableCallback(
    const std_msgs::msg::Bool::SharedPtr msg
  );

  // ===== 追加：Omega XYZ方向力覚フィードバック用 =====
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr sub_force_cmd_;

  double force_cmd_x_ = 0.0;
  double force_cmd_y_ = 0.0;
  double force_cmd_z_ = 0.0;

  double force_cmd_gain_ = 1.0;
  double force_cmd_limit_ = 0.5;
  double force_cmd_sign_ = 1.0;

  void forceCmdCallback(const geometry_msgs::msg::WrenchStamped::SharedPtr msg);

  std::string device_name_;

  double freq_;
  std::string frame_;
  bool realtime_ = true;

  const int id_;

  int override_button_ = -1;
  bool fixed_orientation_ = false;

  double wv_b_ = 0.0;

  double nullPose[DHD_MAX_DOF] = {
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.03
  };

  int initDevice();

  virtual bool sendCommandToOmega(const ohrc_msgs::msg::State &omega) = 0;

  virtual bool initializeOmegaParam() { return true; }

public:
  explicit OmegaDriver(int id);

  ~OmegaDriver();

  int control();
};

#endif // OMEGA_DRIVER_HPP