#ifndef OMEGA_DRIVER_HPP
#define OMEGA_DRIVER_HPP

#include <limits.h>
#include <pthread.h>
#include <rclcpp/rclcpp.hpp>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <Eigen/Geometry>
#include <mutex>
#include <algorithm>
#include <functional>

#include <std_msgs/msg/float64.hpp>

#include "dhdc.h"
#include "drdc.h"
#include "ohrc_msgs/msg/state.hpp"


class OmegaDriver : public rclcpp::Node {
protected:
  rclcpp::Publisher<ohrc_msgs::msg::State>::SharedPtr pub_pose_;
  rclcpp::Publisher<ohrc_msgs::msg::State>::SharedPtr pub_pose2_;

  // ===== 追加：右Omega指への力フィードバック用 =====
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr sub_gripper_force_cmd_;

  double gripper_force_cmd_ = 0.0;
  double gripper_force_gain_ = 0.6;
  double gripper_force_limit_ = 2.0;
  double gripper_force_sign_ = 1.0;

  void gripperForceCallback(const std_msgs::msg::Float64::SharedPtr msg);

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

#endif  // OMEGA_DRIVER_HPP
