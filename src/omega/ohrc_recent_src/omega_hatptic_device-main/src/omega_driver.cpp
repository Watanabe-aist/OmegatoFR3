
#include "omega_haptic_device/omega_driver.hpp"

#include <cerrno>
#include <cmath>
#include <cstring>

OmegaDriver::OmegaDriver(int id)
: Node("omega_driver"), id_(id)
{
  this->declare_parameter("frequency", 1000.0);
  this->declare_parameter("frame", "omega_link");
  this->declare_parameter("realtime", true);
  this->declare_parameter("wv_b", 1.0);

  // ===== 右Omega指への力フィードバック設定 =====
  this->declare_parameter("gripper_force_gain", 0.3);
  this->declare_parameter("gripper_force_limit", 2.0);
  this->declare_parameter("gripper_force_sign", 1.0);

  // ===== 追加：Omega XYZ方向力覚フィードバック設定 =====
  this->declare_parameter("force_cmd_gain", 1.0);
  this->declare_parameter("force_cmd_limit", 0.5);
  this->declare_parameter("force_cmd_sign", 1.0);

  freq_ = this->get_parameter("frequency").as_double();
  frame_ = this->get_parameter("frame").as_string();
  realtime_ = this->get_parameter("realtime").as_bool();
  wv_b_ = this->get_parameter("wv_b").as_double();

  gripper_force_gain_ = this->get_parameter("gripper_force_gain").as_double();
  gripper_force_limit_ = this->get_parameter("gripper_force_limit").as_double();
  gripper_force_sign_ = this->get_parameter("gripper_force_sign").as_double();

  force_cmd_gain_ = this->get_parameter("force_cmd_gain").as_double();
  force_cmd_limit_ = this->get_parameter("force_cmd_limit").as_double();
  force_cmd_sign_ = this->get_parameter("force_cmd_sign").as_double();

  if (initDevice() < 0) {
    rclcpp::shutdown();
    return;
  }

  pub_pose_ = this->create_publisher<ohrc_msgs::msg::State>(
    device_name_ + "/state",
    1
  );

  // ===== 右Omegaだけ /dual_grip/grip_force を購読 =====
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

  // ===== 追加：左右OmegaそれぞれのXYZ力指令を購読 =====
  // device_name_ が "right" なら /right/force_cmd
  // device_name_ が "left"  なら /left/force_cmd
  const std::string force_cmd_topic = "/" + device_name_ + "/force_cmd";

  sub_force_cmd_ = this->create_subscription<geometry_msgs::msg::WrenchStamped>(
    force_cmd_topic,
    1,
    std::bind(&OmegaDriver::forceCmdCallback, this, std::placeholders::_1)
  );

  RCLCPP_INFO(
    this->get_logger(),
    "subscribe Omega XYZ force command: %s",
    force_cmd_topic.c_str()
  );

  RCLCPP_INFO(this->get_logger(), "OmegaDriver initialized (id=%d)", id_);
}

OmegaDriver::~OmegaDriver()
{
  // 安全のため終了時に力を0にしてから閉じる
  dhdSetForceAndTorqueAndGripperForce(
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0,
    id_
  );

  dhdClose(id_);
  munlockall();
}

// ===== 把持力を右Omega指反力に変換 =====
void OmegaDriver::gripperForceCallback(
  const std_msgs::msg::Float64::SharedPtr msg
)
{
  double f_grip = msg->data;

  double f_cmd = gripper_force_sign_ * gripper_force_gain_ * f_grip;

  f_cmd = std::clamp(
    f_cmd,
    -gripper_force_limit_,
    gripper_force_limit_
  );

  gripper_force_cmd_ = f_cmd;
}

// ===== 追加：XYZ方向の力指令を保存 =====
void OmegaDriver::forceCmdCallback(
  const geometry_msgs::msg::WrenchStamped::SharedPtr msg
)
{
  double fx = force_cmd_sign_ * force_cmd_gain_ * msg->wrench.force.x;
  double fy = force_cmd_sign_ * force_cmd_gain_ * msg->wrench.force.y;
  double fz = force_cmd_sign_ * force_cmd_gain_ * msg->wrench.force.z;

  force_cmd_x_ = std::clamp(
    fx,
    -force_cmd_limit_,
    force_cmd_limit_
  );

  force_cmd_y_ = std::clamp(
    fy,
    -force_cmd_limit_,
    force_cmd_limit_
  );

  force_cmd_z_ = std::clamp(
    fz,
    -force_cmd_limit_,
    force_cmd_limit_
  );
}

int OmegaDriver::initDevice()
{
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

  if (systemType == DHD_DEVICE_OMEGA331_LEFT) {
    device_name_ = "left";
  } else if (systemType == DHD_DEVICE_OMEGA331) {
    device_name_ = "right";
  } else {
    device_name_ = "omega";
    RCLCPP_WARN(
      this->get_logger(),
      "unknown Omega system type: %d. device_name_ is set to omega",
      systemType
    );
  }

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

  // 初期位置へ移動してからDRD制御を停止
  drdMoveTo(nullPose, true, id_);
  drdStop(true, id_);

  if (override_button_ == -1) {
    dhdEmulateButton(DHD_ON, id_);
  }

  return 1;
}

int OmegaDriver::control()
{
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
    // subscriber callbackを処理
    // これがないと /dual_grip/grip_force や /right/force_cmd を受け取れない
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

    if (override_button_ == -1) {
      gb = dhdGetButton(0, id_);
    } else {
      gb = override_button_;
    }

    for (int j = 0; j < 3; j++) {
      for (int i = 0; i < 3; i++) {
        R(j, i) = Rot[j][i];
      }
    }

    omega.header.stamp = this->now();

    omega.pose.position.x = px;
    omega.pose.position.y = py;
    omega.pose.position.z = pz;

    if (!fixed_orientation_) {
      omega.pose.orientation = tf2::toMsg(Eigen::Quaterniond(R));
    } else {
      omega.pose.orientation = tf2::toMsg(Eigen::Quaterniond::Identity());
    }

    omega.twist.linear.x = vx;
    omega.twist.linear.y = vy;
    omega.twist.linear.z = vz;

    if (!fixed_orientation_) {
      omega.twist.angular.x = ox;
      omega.twist.angular.y = oy;
      omega.twist.angular.z = oz;
    } else {
      omega.twist.angular.x = 0.0;
      omega.twist.angular.y = 0.0;
      omega.twist.angular.z = 0.0;
    }

    // ===== ここは「状態としてpublishする実測値」 =====
    omega.wrench.force.x = fx;
    omega.wrench.force.y = fy;
    omega.wrench.force.z = fz;

    omega.wrench.torque.x = tx;
    omega.wrench.torque.y = ty;
    omega.wrench.torque.z = tz;

    omega.gripper.angle = gp;
    omega.gripper.anglar_vel = gv;
    omega.gripper.force = fg;
    omega.gripper.button = gb;

    omega.wave.x =
      (omega.wrench.force.x + wv_b_ * omega.twist.linear.x) /
      std::sqrt(2.0 * wv_b_);

    omega.wave.y =
      (omega.wrench.force.y + wv_b_ * omega.twist.linear.y) /
      std::sqrt(2.0 * wv_b_);

    omega.wave.z =
      (omega.wrench.force.z + wv_b_ * omega.twist.linear.z) /
      std::sqrt(2.0 * wv_b_);

    // 実測状態をpublish
    pub_pose_->publish(omega);

    // ===== ここから下はOmegaへ送る指令値 =====
    // publishされる /right/state, /left/state は実測値
    // sendCommandToOmega() に渡す omega.wrench.force は指令値

    omega.wrench.force.x = force_cmd_x_;
    omega.wrench.force.y = force_cmd_y_;
    omega.wrench.force.z = force_cmd_z_;

    omega.wrench.torque.x = 0.0;
    omega.wrench.torque.y = 0.0;
    omega.wrench.torque.z = 0.0;

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
