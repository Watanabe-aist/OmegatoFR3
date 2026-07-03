#include "omega_haptic_device/omega_driver.hpp"
#include "rclcpp/rclcpp.hpp"

// OmegaDriverを継承して純粋仮想関数を実装（ダミー）
class TestOmegaDriver : public OmegaDriver {
public:
  TestOmegaDriver(int id) : OmegaDriver(id) {}

protected:
  bool sendCommandToOmega(const ohrc_msgs::msg::State &omega) override {
    (void)omega;
    RCLCPP_INFO(
      rclcpp::get_logger("TestOmegaDriver"),
      "sendCommandToOmega called"
    );
    return true;
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  auto node = std::make_shared<TestOmegaDriver>(0);

  node->control();

  rclcpp::shutdown();
  return 0;
}
