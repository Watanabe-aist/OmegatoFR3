#include <geometry_msgs/Wrench.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>

#include "omega_haptic_device/omega_driver.hpp"

class OmegaForceWaveVariableController : public OmegaDriver {
  ros::Subscriber subForce, subGripperForce;
  std::mutex mtx1, mtx2;

  geometry_msgs::Vector3 _inputForceWV;
  double _inputGripperForce;

  void cbForceWV(const geometry_msgs::Vector3::ConstPtr& msg) {
    std::lock_guard<std::mutex> lock(mtx1);
    _inputForceWV = *msg;
  }

  void cbGripperForce(const std_msgs::Float32::ConstPtr& msg) {
    std::lock_guard<std::mutex> lock(mtx2);
    _inputGripperForce = msg->data;
  }

  bool sendCommandToOmega(const ohrc_msgs::State& omega) override;

public:
  OmegaForceWaveVariableController(int i);
  ~OmegaForceWaveVariableController(){};
};

OmegaForceWaveVariableController::OmegaForceWaveVariableController(int id) : OmegaDriver(id) {
  // for (int i = 0; i < nDevice; i++)
  //   omageCmd[i].reset(new OmegaCommand(n, deviceName[i]));
  subForce = n.subscribe<geometry_msgs::Vector3>(deviceName + "/wv_s2m", 2, &OmegaForceWaveVariableController::cbForceWV, this, ros::TransportHints().tcpNoDelay(true));
  subGripperForce = n.subscribe<std_msgs::Float32>(deviceName + "/cmd_gripper_force", 2, &OmegaForceWaveVariableController::cbGripperForce, this, ros::TransportHints().tcpNoDelay(true));

  // pubWave = n.advertise<std_msgs::Float32MultiArray>(deviceName + "/wave_variable", 2);
}

bool OmegaForceWaveVariableController::sendCommandToOmega(const ohrc_msgs::State& omega) {
  geometry_msgs::Vector3 inputForceWV;
  double inputGripperForce = 0.0;

  {
    std::lock_guard<std::mutex> lock(mtx1);
    inputForceWV = _inputForceWV;
  }
  {
    std::lock_guard<std::mutex> lock(mtx2);
    inputGripperForce = _inputGripperForce;
  }

  geometry_msgs::Vector3 um = omega.wv_m2s;
  geometry_msgs::Vector3 vm = inputForceWV;

  if (dhdSetForceAndTorqueAndGripperForce(sqrt(wv_b / 2.0) * (um.x + vm.x), sqrt(wv_b / 2.0) * (um.y + vm.y), sqrt(wv_b / 2.0) * (um.z + vm.z), 0., 0., 0., inputGripperForce, id) < DHD_NO_ERROR) {
    printf("error: cannot set force (%s)\n", dhdErrorGetLastStr());
    return false;
  }

  return true;
}
void* cyclic_Task(void* arg) {
  int id = *((int*)arg);
  OmegaForceWaveVariableController OmegaForceWaveVariableController(id);
  if (OmegaForceWaveVariableController.control() < 0)
    ROS_ERROR("Filed to something");

  return nullptr;
}
//////////////////////////////////////////////
int main(int argc, char** argv) {
  ros::init(argc, argv, "omega_force_driver");

  int nDevice = dhdGetDeviceCount();
  ROS_INFO_STREAM("Start Omega initialization: " << nDevice << " device(s) detected");

  std::vector<pthread_t> thread_cyclic_loop(nDevice);
  for (int i = 0; i < nDevice; i++) {
    int device = i;
    pthread_create(&thread_cyclic_loop[i], NULL, cyclic_Task, &device);
    // dhdStartThread(cyclic_Task, &device, DHD_THREAD_PRIORITY_HIGH);
  }
  // loop in the real-time thread

  for (int i = 0; i < nDevice; i++)
    pthread_join(thread_cyclic_loop[i], nullptr);

  return 0;
}