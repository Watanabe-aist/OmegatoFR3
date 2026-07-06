#include "omega_haptic_device/omega_driver.hpp"
#include "rclcpp/rclcpp.hpp"

#include <string>

static const std::string EE_Z_FEEDBACK_SIDE = "right";
// "right" : 右OmegaだけEE Z力覚ON
// "left"  : 左OmegaだけEE Z力覚ON
// "both"  : 左右両方EE Z力覚ON
// "none"  : 左右両方EE Z力覚OFF

static bool enable_ee_z_feedback_for(const std::string &side) {
  if (EE_Z_FEEDBACK_SIDE == "both") {
    return true;
  }
  if (EE_Z_FEEDBACK_SIDE == side) {
    return true;
  }
  return false;
}

class OmegaRight : public OmegaDriver {
public:
  // 右Omegaがid=0で認識されている想定
  OmegaRight() : OmegaDriver(0) {}

protected:
  bool initializeOmegaParam() override {
    if (dhdEnableForce(DHD_ON, id_) < 0) {
      RCLCPP_ERROR(
        this->get_logger(),
        "failed to enable force output (%s)",
        dhdErrorGetLastStr()
      );
      return false;
    }

    RCLCPP_INFO(this->get_logger(), "Right Omega force output enabled");
    RCLCPP_INFO(
      this->get_logger(),
      "EE Z feedback side = %s",
      EE_Z_FEEDBACK_SIDE.c_str()
    );

    return true;
  }

  bool sendCommandToOmega(const ohrc_msgs::msg::State &omega) override {
    // 右FR3手先Z方向力を右Omega本体Z方向へ返す
    double fz_cmd = omega.wrench.force.z;

    if (!enable_ee_z_feedback_for("right")) {
      fz_cmd = 0.0;
    }

    // 安全のため、今はZ方向だけ返す
    // Fx, Fy, torqueは0
    // gripper.forceは今まで通り右グリッパ把持反力
    int rc = dhdSetForceAndTorqueAndGripperForce(
      0.0,
      0.0,
      fz_cmd,
      0.0,
      0.0,
      0.0,
      omega.gripper.force,
      id_
    );

    if (rc < 0) {
      RCLCPP_ERROR(
        this->get_logger(),
        "failed to send force to right Omega (%s)",
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