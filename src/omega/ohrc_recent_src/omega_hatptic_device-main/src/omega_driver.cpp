#include "omega_haptic_device/omega_driver.hpp"

OmegaDriver::OmegaDriver(int id)
: Node("omega_driver"), id_(id)
{
  this->declare_parameter("frequency", 1000.0);
  this->declare_parameter("frame", "omega_link");
  this->declare_parameter("realtime", true);
  this->declare_parameter("wv_b", 1.0);

  // ===== 追加：右Omega指への力フィードバック設定 =====
  this->declare_parameter("gripper_force_gain", 0.3);
  this->declare_parameter("gripper_force_limit", 2.0);
  this->declare_parameter("gripper_force_sign", 1.0);

  freq_     = this->get_parameter("frequency").as_double();
  frame_    = this->get_parameter("frame").as_string();
  realtime_ = this->get_parameter("realtime").as_bool();
  wv_b_     = this->get_parameter("wv_b").as_double();

  gripper_force_gain_  = this->get_parameter("gripper_force_gain").as_double();
  gripper_force_limit_ = this->get_parameter("gripper_force_limit").as_double();
  gripper_force_sign_  = this->get_parameter("gripper_force_sign").as_double();

  if (initDevice() < 0)
    rclcpp::shutdown();

  pub_pose_ = this->create_publisher<ohrc_msgs::msg::State>(
    device_name_ + "/state",
    1
  );

  // ===== 追加：右Omegaだけ /dual_grip/grip_force を購読 =====
  if (device_name_ == "right") {
    sub_gripper_force_cmd_ = this->create_subscription<std_msgs::msg::Float64>(
      "/dual_grip/grip_force",
      1,
      std::bind(&OmegaDriver::gripperForceCallback, this, std::placeholders::_1)
    );

    RCLCPP_INFO(
      this->get_logger(),
      "subscribe gripper force feedback: /dual_grip/grip_force"
    );
  }

  RCLCPP_INFO(this->get_logger(), "OmegaDriver initialized (id=%d)", id_);
}

OmegaDriver::~OmegaDriver() {
  dhdClose(id_);
  munlockall();
}

// ===== 追加：把持力を右Omega指反力に変換 =====
void OmegaDriver::gripperForceCallback(
  const std_msgs::msg::Float64::SharedPtr msg
) {
  double f_grip = msg->data;

  double f_cmd = gripper_force_sign_ * gripper_force_gain_ * f_grip;

  if (f_cmd > gripper_force_limit_) {
    f_cmd = gripper_force_limit_;
  }

  if (f_cmd < -gripper_force_limit_) {
    f_cmd = -gripper_force_limit_;
  }

  gripper_force_cmd_ = f_cmd;
}

int OmegaDriver::initDevice() {
  if (drdOpenID(id_) < 0) {
    RCLCPP_ERROR(
      this->get_logger(),
      "error: cannot open device (%s)",
      dhdErrorGetLastStr()
    );
    dhdSleep(2.0);
    return -1;
  }

  if (!drdIsSupported(id_)) {
    RCLCPP_ERROR(this->get_logger(), "unsupported device");
    dhdSleep(2.0);
    drdClose(id_);
    return -1;
  }

  RCLCPP_INFO(
    this->get_logger(),
    "%s haptic device detected",
    dhdGetSystemName(id_)
  );

  int systemType = dhdGetSystemType(id_);
  if (systemType == DHD_DEVICE_OMEGA331_LEFT)
    device_name_ = "left";
  else if (systemType == DHD_DEVICE_OMEGA331)
    device_name_ = "right";

  if (!drdIsInitialized(id_) && drdAutoInit(id_) < 0) {
    RCLCPP_ERROR(
      this->get_logger(),
      "auto-initialization failed (%s)",
      dhdErrorGetLastStr()
    );
    dhdSleep(2.0);
    return -1;
  } else if (drdStart(id_) < 0) {
    RCLCPP_ERROR(
      this->get_logger(),
      "regulation thread failed to start (%s)",
      dhdErrorGetLastStr()
    );
    dhdSleep(2.0);
    return -1;
  }

  // もし中央に行った後に落ちる問題を直したいなら、
  // 下2行のうち drdMoveTo はコメントアウトでもOK
  drdMoveTo(nullPose, true, id_);
  drdStop(true, id_);

  if (override_button_ == -1)
    dhdEmulateButton(DHD_ON, id_);

  return 1;
}

int OmegaDriver::control() {
  if (realtime_) {
    cpu_set_t cpuset;
    pthread_t thread = pthread_self();

    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);

    struct sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_FIFO);

    if (pthread_setschedparam(thread, SCHED_FIFO, &param) < 0) {
      RCLCPP_ERROR(this->get_logger(), "sched_setscheduler: %s", strerror(errno));
      return -1;
    }

    if (mlockall(MCL_CURRENT | MCL_FUTURE) < 0) {
      RCLCPP_ERROR(this->get_logger(), "mlockall: %s", strerror(errno));
      return -1;
    }
  }

  int done = 0;
  ohrc_msgs::msg::State omega;
  omega.header.frame_id = frame_;

  double px, py, pz;
  double vx, vy, vz;
  double ox, oy, oz;
  double gp, gv;
  double fx, fy, fz;
  double tx, ty, tz;
  double fg;
  double oa, ob, og;
  double Rot[3][3];

  bool gb;
  Eigen::Matrix3d R;

  rclcpp::Rate r(freq_);

  if (!initializeOmegaParam()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to change parameters of Omega device");
    return -1;
  }

  RCLCPP_INFO(this->get_logger(), "Start Omega communication loop");

  while (!done && rclcpp::ok()) {
    // ===== 追加：subscriber callbackを処理 =====
    // これがないと /dual_grip/grip_force を受け取れない
    rclcpp::spin_some(this->get_node_base_interface());

    int rc = 0;

    rc += drdGetPositionAndOrientation(
      &px, &py, &pz,
      &oa, &ob, &og,
      &gp,
      Rot,
      id_
    );

    rc += drdGetVelocity(
      &vx, &vy, &vz,
      &ox, &oy, &oz,
      &gv,
      id_
    );

    rc += dhdGetForceAndTorqueAndGripperForce(
      &fx, &fy, &fz,
      &tx, &ty, &tz,
      &fg,
      id_
    );

    if (rc < 0) {
      RCLCPP_ERROR(this->get_logger(), "Failed to get data from Omega device(s)");
      return -1;
    }

    if (override_button_ == -1)
      gb = dhdGetButton(0, id_);
    else
      gb = override_button_;

    for (int j = 0; j < 3; j++) {
      for (int i = 0; i < 3; i++) {
        R(j, i) = Rot[j][i];
      }
    }

    omega.header.stamp = this->now();

    omega.pose.position.x = px;
    omega.pose.position.y = py;
    omega.pose.position.z = pz;

    if (!fixed_orientation_)
      omega.pose.orientation = tf2::toMsg(Eigen::Quaterniond(R));
    else
      omega.pose.orientation = tf2::toMsg(Eigen::Quaterniond::Identity());

    omega.twist.linear.x = vx;
    omega.twist.linear.y = vy;
    omega.twist.linear.z = vz;

    if (!fixed_orientation_) {
      omega.twist.angular.x = ox;
      omega.twist.angular.y = oy;
      omega.twist.angular.z = oz;
    }

    omega.wrench.force.x = fx;
    omega.wrench.force.y = fy;
    omega.wrench.force.z = fz;

    omega.wrench.torque.x = tx;
    omega.wrench.torque.y = ty;
    omega.wrench.torque.z = tz;

    // ここは「状態としてpublishする実測値」
    omega.gripper.angle = gp;
    omega.gripper.anglar_vel = gv;
    omega.gripper.force = fg;
    omega.gripper.button = gb;

    omega.wave.x = (omega.wrench.force.x + wv_b_ * omega.twist.linear.x) / sqrt(2.0 * wv_b_);
    omega.wave.y = (omega.wrench.force.y + wv_b_ * omega.twist.linear.y) / sqrt(2.0 * wv_b_);
    omega.wave.z = (omega.wrench.force.z + wv_b_ * omega.twist.linear.z) / sqrt(2.0 * wv_b_);

    // 実測状態をpublish
    pub_pose_->publish(omega);

    // ===== 追加：送信コマンド用にgripper.forceを上書き =====
    // publishされる /right/state の gripper.force は実測fg
    // sendCommandToOmega() に渡す gripper.force は指令値
    if (device_name_ == "right") {
      omega.gripper.force = gripper_force_cmd_;
    } else {
      omega.gripper.force = 0.0;
    }

    if (!sendCommandToOmega(omega)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to send control command to Omega device");
      return -1;
    }

    r.sleep();
  }

  return 0;
}