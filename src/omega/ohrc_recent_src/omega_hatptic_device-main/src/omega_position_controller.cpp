#include <geometry_msgs/Pose.h>

#include "omega_haptic_device/omega_driver.hpp"

class OmegaPositionController : public OmegaDriver {
  ros::Subscriber subPose;
  std::mutex mtx3;

  geometry_msgs::Pose _inputPose;
  double _inputGripperForce = 0.0;

  void cbPose(const geometry_msgs::Pose::ConstPtr& msg);

  bool initializeOmegaParam() override;
  bool sendCommandToOmega(const ohrc_msgs::State& omega) override;

public:
  OmegaPositionController(int id);
  ~OmegaPositionController(){};
};

bool OmegaPositionController::initializeOmegaParam() {
  if (drdStart() < 0) {
    ROS_ERROR("error: regulation thread failed to start (%s)\n", dhdErrorGetLastStr());
    dhdSleep(2.0);
    return false;
  }

  // if (drdSetPosTrackParam(150.0, 150.0, 150.0) < 0) {
  // ROS_ERROR("error: set track param (%s)\n", dhdErrorGetLastStr());
  // dhdSleep(2.0);
  // return false;
  // }

  double amax, vmax, jerk;
  drdGetPosTrackParam(&amax, &vmax, &jerk);
  ROS_INFO_STREAM(" Tracking Filter: acc max: " << amax << ", vel max: " << vmax << ", jerk max: " << jerk);

  override_button = 0;
  fixedOrientation = true;

  return true;
}

OmegaPositionController::OmegaPositionController(int id) : OmegaDriver(id) {
  subPose = n.subscribe<geometry_msgs::Pose>(deviceName + "/cmd_pose", 2, &OmegaPositionController::cbPose, this);
}

void OmegaPositionController::cbPose(const geometry_msgs::Pose::ConstPtr& msg) {
  std::lock_guard<std::mutex> lock(mtx3);
  _inputPose = *msg;

  override_button = 1;
}

bool OmegaPositionController::sendCommandToOmega(const ohrc_msgs::State& omega) {
  geometry_msgs::Pose inputPose;

  {
    std::lock_guard<std::mutex> lock(mtx3);
    inputPose = _inputPose;
  }

  double targetPose[DHD_MAX_DOF] = { 0.0 };
  targetPose[0] = inputPose.position.x;
  targetPose[1] = inputPose.position.y;
  targetPose[3] = inputPose.position.z;
  targetPose[6] = nullPose[6];

  if (drdTrack(targetPose, id)) {
    printf("error: cannot set regulation pose (%s)\n", dhdErrorGetLastStr());
    return false;
  }
  return true;
}

void* cyclic_Task(void* arg) {
  int id = *((int*)arg);
  OmegaPositionController OmegaPositionController(id);
  if (OmegaPositionController.control() < 0)
    ROS_ERROR("Filed to something");

  return nullptr;
}
//////////////////////////////////////////////
int main(int argc, char** argv) {
  ros::init(argc, argv, "omega_position_driver");

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