#include "omega_haptic_device/omega_force_driver.hpp"

OmegaForceController::OmegaForceController(int id) : OmegaDriver(id) {
  // for (int i = 0; i < nDevice; i++)
  //   omageCmd[i].reset(new OmegaCommand(n, deviceName[i]));
  subForce = n.subscribe<geometry_msgs::Wrench>(deviceName + "/cmd_force", 2, &OmegaForceController::cbForce, this, ros::TransportHints().tcpNoDelay(true));
  subGripperForce = n.subscribe<std_msgs::Float32>(deviceName + "/cmd_gripper_force", 2, &OmegaForceController::cbGripperForce, this, ros::TransportHints().tcpNoDelay(true));

  // subFollowerEnergy = n.subscribe<std_msgs::Float32>("/passivity_observer/follower/energy", 2, &OmegaForceController::cbEnergy, this, ros::TransportHints().tcpNoDelay(true));

  // pubEnergy = n.advertise<std_msgs::Float32>("/passivity_observer/leader/energy", 2);
}

bool OmegaForceController::sendCommandToOmega(const ohrc_msgs::State& omega) {
  geometry_msgs::Wrench inputForce;
  double inputGripperForce = 0.0;

  {
    std::lock_guard<std::mutex> lock(mtx);
    inputForce = _inputForce;
    inputGripperForce = _inputGripperForce;
  }

  this->modifyOmegaCommand(omega, inputForce, inputGripperForce);

#if 0
  double followerEnergy = 0.0;
  {
    std::lock_guard<std::mutex> lock(mtx3);
    followerEnergy = _followerEnergy;
  }
  double dt = 1.0 / freq;

  static Eigen::Vector3d E_out = Eigen::Vector3d::Zero();
  static Eigen::Vector3d E_PC = Eigen::Vector3d::Zero();

  Eigen::Vector3d f, v;
  tf2::fromMsg(inputForce.force, f);
  tf2::fromMsg(omega.twist.linear, v);

  // int i = 2;
  for (int i = 0; i < 3; i++) {
    E_out(i) += (-f(i) * v(i)) * dt;

    double E_PO = 0.0;
    double E_in = 0.0;  // followerEnergy;

    if (omega.gripper.button)
      E_PO = E_in + E_out(i) + E_PC(i);
    else {
      E_out(i) = 0.0;
      E_PC(i) = 0.0;
    }

    // std_msgs::Float32MultiArray e;
    // e.data.push_back(E_out(i));
    // pubEnergy.publish(e);

    // std::cout << E_PO << std::endl;

    // std::cout << omega.twist.linear.z << ", " << inputForce.force.z << ", " << E << std::endl;
    // double alpha = 0.0;

    double alpha = 0.0;
    if (E_PO < 0.0 && abs(v(i)) > 1.0e-4)
      alpha = -E_PO / (dt * v(i) * v(i));

    // double fz = inputForce.force.z - omega.twist.linear.z * alpha;
    // if (alpha > 0.0) {
    //   std::cout << "E_PO: " << E_PO << ", E_out: " << E_out << std::endl;
    //   std::cout << "(original force) " << inputForce.force.z << "- (modification) " << omega.twist.linear.z * alpha << "= (total) " << fz << std::endl;
    // }

    // inputForce.force.z = fz;
    // f(i) = f(i) - v(i) * alpha;

    E_PC(i) += alpha * (v(i) * v(i)) * dt;
  }
  tf2::toMsg(f, inputForce.force);
#endif
  if (dhdSetForceAndTorqueAndGripperForce(inputForce.force.x, inputForce.force.y, inputForce.force.z, inputForce.torque.x, inputForce.torque.y, inputForce.torque.z, inputGripperForce, id) < DHD_NO_ERROR) {
    printf("error: cannot set force (%s)\n", dhdErrorGetLastStr());
    return false;
  }

  return true;
}
void* cyclic_Task(void* arg) {
  int id = *((int*)arg);
  OmegaForceController OmegaForceController(id);
  if (OmegaForceController.control() < 0)
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