#include "omega_haptic_device/omega_driver.hpp"
#include "rclcpp/rclcpp.hpp"

class OmegaRight : public OmegaDriver {
public:
  // 右Omegaがid=0で認識されている想定
  OmegaRight() : OmegaDriver(0) {}

protected:
  bool initializeOmegaParam() override {
    // Omegaへの力出力を有効化
    if (dhdEnableForce(DHD_ON, id_) < 0) {
      RCLCPP_ERROR(
        this->get_logger(),
        "failed to enable force output (%s)",
        dhdErrorGetLastStr()
      );
      return false;
    }

    RCLCPP_INFO(this->get_logger(), "Right Omega force output enabled");
    return true;
  }

  bool sendCommandToOmega(const ohrc_msgs::msg::State &omega) override {
    // 今回は右Omegaの指だけに反力を返す
    // 並進力・手首トルクは安全のため0
    int rc = dhdSetForceAndTorqueAndGripperForce(
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      omega.gripper.force,
      id_
    );

    if (rc < 0) {
      RCLCPP_ERROR(
        this->get_logger(),
        "failed to send gripper force to right Omega (%s)",
        dhdErrorGetLastStr()
      );
      return false;
    }

    return true;
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  auto node = std::make_shared<OmegaRight>();
  int ret = node->control();

  rclcpp::shutdown();
  return ret;
}
