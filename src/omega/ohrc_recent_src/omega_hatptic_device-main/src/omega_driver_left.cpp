
#include "omega_haptic_device/omega_driver.hpp"

#include "rclcpp/rclcpp.hpp"

class OmegaLeft : public OmegaDriver {
public:
  // 注意：
  // 現在の元コードでは左Omegaも OmegaDriver(0) になっている。
  // もし左Omegaが動かない場合は，ここを OmegaDriver(1) に変更して試す。
  OmegaLeft() : OmegaDriver(0) {}

protected:
  bool initializeOmegaParam() override
  {
    // Omegaへの力出力を有効化
    if (dhdEnableForce(DHD_ON, id_) < 0) {
      RCLCPP_ERROR(
        this->get_logger(),
        "failed to enable force output for left Omega (%s)",
        dhdErrorGetLastStr()
      );
      return false;
    }

    RCLCPP_INFO(this->get_logger(), "Left Omega force output enabled");
    return true;
  }

  bool sendCommandToOmega(const ohrc_msgs::msg::State &omega) override
  {
    // Python側 /left/force_cmd から受け取ったXYZ力を左Omegaへ出力する
    // 手首トルクとグリッパ力は安全のため0
    int rc = dhdSetForceAndTorqueAndGripperForce(
      omega.wrench.force.x,
      omega.wrench.force.y,
      omega.wrench.force.z,
      0.0,
      0.0,
      0.0,
      0.0,
      id_
    );

    if (rc < 0) {
      RCLCPP_ERROR(
        this->get_logger(),
        "failed to send force command to left Omega (%s)",
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

  auto node = std::make_shared<OmegaLeft>();
  int ret = node->control();

  rclcpp::shutdown();
  return ret;
}
