#include "omega_haptic_device/omega_driver.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<OmegaDriver>(0);
  node->control();   // 通信ループ開始
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

