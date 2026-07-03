#include "omega_haptic_device/omega_driver.hpp"

#include <memory>

#include "rclcpp/rclcpp.hpp"

class OmegaRight : public OmegaDriver {
public:
  // 右Omegaが id=0 で認識されている想定
  OmegaRight() : OmegaDriver(0) {}

protected:
  bool initializeOmegaParam() override
  {
    // Omegaへの力出力を有効化
    if (dhdEnableForce(DHD_ON, id_) < 0) {
      RCLCPP_ERROR(
        this->get_logger(),
        "failed to enable force output for right Omega (%s)",
        dhdErrorGetLastStr()
      );
      return false;
    }

    RCLCPP_INFO(this->get_logger(), "Right Omega force output enabled");
    return true;
  }

  bool sendCommandToOmega(const ohrc_msgs::msg::State &omega) override
  {
    // /right/force_cmd から受け取ったXYZ力を右Omegaへ出力する
    // 手首トルクは安全のため0
    // gripper.force は従来通り右Omega指への反力として使う
    int rc = dhdSetForceAndTorqueAndGripperForce(
      omega.wrench.force.x,
      omega.wrench.force.y,
      omega.wrench.force.z,
      0.0,
      0.0,
      0.0,
      omega.gripper.force,
      id_
    );

    if (rc < 0) {
      RCLCPP_ERROR(
        this->get_logger(),
        "failed to send force command to right Omega (%s)",
        dhdErrorGetLastStr()
      );
      return false;
    }

    return true;
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<OmegaRight>();
  int ret = node->control();

  rclcpp::shutdown();
  return ret;
}
