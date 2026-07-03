#include "omega_haptic_device/omega_driver.hpp"
#include "rclcpp/rclcpp.hpp"

class OmegaLeft : public OmegaDriver {
public:
  OmegaLeft() : OmegaDriver(0) {}

protected:
  bool sendCommandToOmega(const ohrc_msgs::msg::State &omega) override {
    (void)omega;
    return true;
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<OmegaLeft>();
  node->control();
  rclcpp::shutdown();
  return 0;
}
