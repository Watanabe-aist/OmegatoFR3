#include "omega_haptic_device/omega_driver.hpp"
#include "rclcpp/rclcpp.hpp"

class OmegaLeft : public OmegaDriver {
public:
  // この環境では左も起動順で id=0 として開く
  // OmegaDriver(1) は no device found になるので使わない
  OmegaLeft() : OmegaDriver(0) {}

protected:
  bool initializeOmegaParam() override {
    // 左Omegaも力覚出力をONにする
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

  bool sendCommandToOmega(const ohrc_msgs::msg::State &omega) override {
    // 左FR3の手先力を左Omega本体へ返す
    // 今のPython側ではZだけ非ゼロなので、実質Fzだけ返る
    //
    // グリッパ把持力は右Omegaだけに返す設計なので、
    // 左Omegaのgripper forceは0.0にする
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
        "failed to send force to left Omega (%s)",
        dhdErrorGetLastStr()
      );
      return false;
    }

    return true;
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  auto node = std::make_shared<OmegaLeft>();
  int ret = node->control();

  rclcpp::shutdown();
  return ret;
}