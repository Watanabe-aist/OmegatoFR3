import argparse
import sys

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, WrenchStamped
from std_msgs.msg import Float64

from pylibfranka import Robot, Torques


# ===========================
# 把持力計算用
# ===========================
FORCE_SIGN = -1.0
CONTACT_THRESHOLD = 1.0  # N


# ===========================
# 双腕base間オフセット
# ===========================
# 左Franka baseを共通world原点としたときの、右Franka baseの位置 [m]
# ロボットが平行・同じ向きで、
# 右Frankaが左Frankaから -Y 方向に1.0 m離れている場合
RIGHT_BASE_OFFSET_IN_LEFT_WORLD = np.array([0.0, -0.50, 0.0], dtype=float)


# ===========================
# FR3ベース取付姿勢
# ===========================
#
# logical name / IP対応は従来のまま:
#
# left  -> 192.168.2.121 -> +45 deg
# right -> 192.168.2.122 -> -45 deg
#
# O_F_ext_hat_K は各FR3 base座標なので、
# ee_wrench_worldへ出す前にcommon worldへ回転する。
LEFT_FR3_BASE_ROLL_RAD = np.deg2rad(+45.0)
RIGHT_FR3_BASE_ROLL_RAD = np.deg2rad(-45.0)


def rot_x(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ], dtype=float)


LEFT_FR3_BASE_ROT_WORLD = rot_x(
    LEFT_FR3_BASE_ROLL_RAD
)

RIGHT_FR3_BASE_ROT_WORLD = rot_x(
    RIGHT_FR3_BASE_ROLL_RAD
)


def get_fr3_base_rot_world(side):
    if side == "left":
        return LEFT_FR3_BASE_ROT_WORLD

    if side == "right":
        return RIGHT_FR3_BASE_ROT_WORLD

    return np.eye(3)



# もし右Frankaが左Frankaから +Y 方向に1.0 m離れているなら、上をこれに変える：
# RIGHT_BASE_OFFSET_IN_LEFT_WORLD = np.array([0.0, 1.0, 0.0], dtype=float)


class FrankaOmegaReal(Node):
    def __init__(self, name):
        super().__init__(f"franka_omega_real_{name}")

        self.name = name

        if self.name == "left":
            self.ip = "192.168.2.121"
        elif self.name == "right":
            self.ip = "192.168.2.122"
        else:
            raise ValueError("name must be left or right")

        self.q_goal = None
        self.prev_cmd = None

        self.max_joint_vel = 3.5
        self.dt = 1.0 / 1000.0

        self.q_goal_sub = self.create_subscription(
            JointState,
            f"/omega_fr3/{self.name}/q_goal",
            self.q_goal_callback,
            10,
        )

        self.obs_franka_state_pub = self.create_publisher(
            JointState,
            f"/franka/{self.name}/obs_franka_state",
            10,
        )

        # ===========================
        # 追加：
        # 外乱関節トルク publish
        # record_z_force_bias_graph.py の関節トルク推定版が読むtopic
        # ===========================
        self.obs_franka_torque_pub = self.create_publisher(
            JointState,
            f"/franka/{self.name}/obs_franka_torque",
            10,
        )

        self.cmd_franka_pos_pub = self.create_publisher(
            JointState,
            f"/factr_teleop/{self.name}/cmd_franka_pos",
            10,
        )

        # 自分の手先位置をpublish
        self.ee_pose_pub = self.create_publisher(
            PoseStamped,
            f"/franka/{self.name}/ee_pose",
            10,
        )

        # 自分の外力をpublish
        self.ee_wrench_pub = self.create_publisher(
            WrenchStamped,
            f"/franka/{self.name}/ee_wrench_world",
            10,
        )

        self.state_pub_count = 0

        # 外乱関節トルクの属性確認用
        self.external_torque_attr_name = None
        self.warned_no_external_torque = False

        # ===========================
        # 右側プロセスだけが把持力を計算する
        # ===========================
        self.p_left = None
        self.p_right = None
        self.f_left = None
        self.f_right = None
        self.grip_log_count = 0

        if self.name == "right":
            self.left_pose_sub = self.create_subscription(
                PoseStamped,
                "/franka/left/ee_pose",
                self.left_pose_callback,
                10,
            )

            self.right_pose_sub = self.create_subscription(
                PoseStamped,
                "/franka/right/ee_pose",
                self.right_pose_callback,
                10,
            )

            self.left_wrench_sub = self.create_subscription(
                WrenchStamped,
                "/franka/left/ee_wrench_world",
                self.left_wrench_callback,
                10,
            )

            self.right_wrench_sub = self.create_subscription(
                WrenchStamped,
                "/franka/right/ee_wrench_world",
                self.right_wrench_callback,
                10,
            )

            self.grip_force_pub = self.create_publisher(
                Float64,
                "/dual_grip/grip_force",
                10,
            )

            self.grip_force_timer = self.create_timer(
                0.005,
                self.compute_and_publish_grip_force,
            )

            self.get_logger().info("right node will compute /dual_grip/grip_force")

        self.get_logger().info(f"FrankaOmegaReal started for {self.name}")
        self.get_logger().info(f"subscribe: /omega_fr3/{self.name}/q_goal")
        self.get_logger().info(f"robot ip : {self.ip}")
        self.get_logger().info(f"publish obs_franka_state : /franka/{self.name}/obs_franka_state")
        self.get_logger().info(f"publish obs_franka_torque: /franka/{self.name}/obs_franka_torque")
        self.get_logger().info(f"publish ee_pose : /franka/{self.name}/ee_pose")
        self.get_logger().info(f"publish ee_wrench_world: /franka/{self.name}/ee_wrench_world")

    # ======================================================
    # callbacks / common publish
    # ======================================================
    def q_goal_callback(self, msg):
        if len(msg.position) < 7:
            self.get_logger().warn("q_goal has less than 7 joints")
            return

        self.q_goal = np.array(msg.position[:7], dtype=float)

    def rate_limit(self, cmd):
        if self.prev_cmd is None:
            self.prev_cmd = cmd.copy()
            return cmd

        max_step = self.max_joint_vel * self.dt
        step = np.clip(cmd - self.prev_cmd, -max_step, max_step)
        limited = self.prev_cmd + step
        self.prev_cmd = limited.copy()

        return limited

    def publish_joint_state(self, pub, q):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = q.tolist()
        pub.publish(msg)

    # ======================================================
    # 追加：
    # robot_stateから外乱関節トルクを取り出す
    # ======================================================
    def get_external_joint_torque(self, robot_state):
        """
        pylibfranka RobotState から外乱関節トルクを取り出す。

        優先:
          tau_ext_hat_filtered

        それが無い場合:
          tau_ext_hat

        どちらも無い場合:
          None を返す
        """

        candidate_attrs = [
            "tau_ext_hat_filtered",
            "tau_ext_hat",
        ]

        for attr in candidate_attrs:
            if not hasattr(robot_state, attr):
                continue

            try:
                tau = np.array(getattr(robot_state, attr), dtype=float).reshape(-1)
            except Exception:
                continue

            if tau.shape[0] >= 7:
                if self.external_torque_attr_name is None:
                    self.external_torque_attr_name = attr
                    self.get_logger().info(
                        f"using robot_state.{attr} for /franka/{self.name}/obs_franka_torque"
                    )

                return tau[:7]

        if not self.warned_no_external_torque:
            available_tau_like_attrs = [
                a for a in dir(robot_state)
                if "tau" in a.lower() or "torque" in a.lower()
            ]

            self.get_logger().warn(
                "robot_state does not have tau_ext_hat_filtered or tau_ext_hat. "
                f"Available torque-like attributes: {available_tau_like_attrs}"
            )
            self.warned_no_external_torque = True

        return None

    def publish_external_joint_torque(self, robot_state, stamp):
        """
        /franka/{left/right}/obs_franka_torque をpublishする。
        FACTR側と合わせて JointState.position に7関節分を入れる。
        """

        tau_ext = self.get_external_joint_torque(robot_state)

        if tau_ext is None:
            return

        msg = JointState()
        msg.header.stamp = stamp
        msg.name = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        msg.position = list(map(float, tau_ext[:7]))

        self.obs_franka_torque_pub.publish(msg)

    def publish_ee_pose_wrench_and_torque(self, robot_state):
        self.state_pub_count += 1

        # 1000Hzで出すと重いので、100Hz程度に落とす
        if self.state_pub_count % 10 != 0:
            return

        now = self.get_clock().now().to_msg()

        # ---------------------------
        # EE pose publish
        # ---------------------------
        # O_T_EE: 各Franka baseから見たEE姿勢
        # libfrankaはcolumn-majorなので位置は 12,13,14
        T = np.array(robot_state.O_T_EE, dtype=float)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = now

        # 注意：
        # ここでは元コードに合わせて "world" にしているが、
        # 実際には left/right それぞれのFranka base基準の値。
        # 把持力計算時に右手先位置へbase間オフセットを加えて補正する。
        pose_msg.header.frame_id = "world"

        pose_msg.pose.position.x = float(T[12])
        pose_msg.pose.position.y = float(T[13])
        pose_msg.pose.position.z = float(T[14])

        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0

        self.ee_pose_pub.publish(pose_msg)

        # ---------------------------
        # EE wrench publish
        # ---------------------------
        # O_F_ext_hat_K は各FR3 base座標。
        # ±45deg mountingを反映してcommon worldへ回す。
        wrench_base = np.array(
            robot_state.O_F_ext_hat_K,
            dtype=float,
        )

        R_world_base = get_fr3_base_rot_world(
            self.name
        )

        force_world = (
            R_world_base
            @ wrench_base[:3]
        )

        torque_world = (
            R_world_base
            @ wrench_base[3:6]
        )

        wrench_msg = WrenchStamped()
        wrench_msg.header.stamp = now
        wrench_msg.header.frame_id = "world"

        wrench_msg.wrench.force.x = float(force_world[0])
        wrench_msg.wrench.force.y = float(force_world[1])
        wrench_msg.wrench.force.z = float(force_world[2])

        wrench_msg.wrench.torque.x = float(torque_world[0])
        wrench_msg.wrench.torque.y = float(torque_world[1])
        wrench_msg.wrench.torque.z = float(torque_world[2])

        self.ee_wrench_pub.publish(wrench_msg)
        # ---------------------------
        # 追加：
        # external joint torque publish
        # ---------------------------
        self.publish_external_joint_torque(robot_state, now)

    # ======================================================
    # 右ノードだけが使うcallback
    # ======================================================
    def left_pose_callback(self, msg):
        self.p_left = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=float,
        )

    def right_pose_callback(self, msg):
        self.p_right = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=float,
        )

    def left_wrench_callback(self, msg):
        self.f_left = np.array(
            [
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z,
            ],
            dtype=float,
        )

    def right_wrench_callback(self, msg):
        self.f_right = np.array(
            [
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z,
            ],
            dtype=float,
        )

    def compute_and_publish_grip_force(self):
        if self.name != "right":
            return

        if self.p_left is None:
            return
        if self.p_right is None:
            return
        if self.f_left is None:
            return
        if self.f_right is None:
            return

        # ==================================================
        # 左Franka baseを共通world座標系として扱う
        #
        # self.p_left:
        #   左Franka baseから見た左手先位置
        #
        # self.p_right:
        #   右Franka baseから見た右手先位置
        #
        # 右Franka baseは左Franka baseから1.0m離れているため、
        # 右手先位置にbase間オフセットを加えて、
        # 左Franka基準の共通world座標へ変換する。
        # ==================================================
        p_left_world = (
            LEFT_FR3_BASE_ROT_WORLD
            @ self.p_left
        )

        p_right_world = (
            RIGHT_BASE_OFFSET_IN_LEFT_WORLD
            + RIGHT_FR3_BASE_ROT_WORLD
            @ self.p_right
        )
        diff = p_right_world - p_left_world
        dist = np.linalg.norm(diff)

        if dist < 1e-6:
            return

        # 左手先から右手先への単位ベクトル
        # これが「ワールド座標で見た左右手先を結ぶ方向」
        e = diff / dist

        # 左手の内側方向: +e
        # 右手の内側方向: -e
        raw_left = float(np.dot(self.f_left, e))
        raw_right = float(np.dot(self.f_right, -e))

        # Frankaの外力推定の符号に合わせて反転
        f_left_in = FORCE_SIGN * raw_left
        f_right_in = FORCE_SIGN * raw_right

        # 内向きに押している成分だけ残す
        f_left_in = max(0.0, f_left_in)
        f_right_in = max(0.0, f_right_in)

        # 左右の内向き力の平均を把持力とする
        f_grip = 0.5 * (f_left_in + f_right_in)

        # 小さいノイズは接触なしとして0にする
        if f_grip < CONTACT_THRESHOLD:
            f_grip = 0.0

        msg = Float64()
        msg.data = float(f_grip)
        self.grip_force_pub.publish(msg)

        self.grip_log_count += 1

        if self.grip_log_count % 20 == 0:
            self.get_logger().info(
                f"pL_world=[{p_left_world[0]:.3f}, {p_left_world[1]:.3f}, {p_left_world[2]:.3f}], "
                f"pR_world=[{p_right_world[0]:.3f}, {p_right_world[1]:.3f}, {p_right_world[2]:.3f}], "
                f"dist={dist:.3f} m, "
                f"e=[{e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f}], "
                f"raw_L={raw_left:.2f}, raw_R={raw_right:.2f}, "
                f"in_L={f_left_in:.2f}, in_R={f_right_in:.2f}, "
                f"F_grip={f_grip:.2f} N"
            )

    # ======================================================
    # main control loop
    # ======================================================
    def run(self):
        joint_stiffness = np.array(
            [90, 90, 90, 30, 30, 30, 30],
            dtype=float,
        )
        joint_damping = np.array(
            [2.5 * np.sqrt(k) for k in joint_stiffness],
            dtype=float,
        )

        robot = None

        try:
            robot = Robot(self.ip)

            robot.set_collision_behavior(
                [6000.0] * 7,
                [6000.0] * 7,
                [6000.0] * 6,
                [6000.0] * 6,
            )

            initial_state = robot.read_once()
            current_q = np.array(initial_state.q)

            self.get_logger().info(f"initial q: {current_q}")

            active_control = robot.start_torque_control()
            model = robot.load_model()

            init_error = None
            prev_dq = np.zeros(7)

            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.0)

                robot_state, _ = active_control.readOnce()

                coriolis = np.array(model.coriolis(robot_state))

                q = np.array(robot_state.q)
                dq = np.array(robot_state.dq)

                # 現在関節角度をpublish
                self.publish_joint_state(self.obs_franka_state_pub, q)

                # 自分のEE位置・手先外力・外乱関節トルクをpublish
                self.publish_ee_pose_wrench_and_torque(robot_state)

                if self.q_goal is None:
                    continue

                q_goal = self.rate_limit(self.q_goal.copy())

                self.publish_joint_state(self.cmd_franka_pos_pub, q_goal)

                if init_error is None:
                    init_error = q - q_goal
                    self.get_logger().info("init_error set")

                position_error = q - q_goal - init_error

                tau_task = np.zeros(7)

                for i in range(7):
                    dq[i] = prev_dq[i] * 0.01 + dq[i] * 0.99
                    prev_dq[i] = dq[i]

                    tau_task[i] = (
                        -joint_stiffness[i] * position_error[i]
                        - joint_damping[i] * dq[i]
                    )

                tau_d = tau_task + coriolis

                torque_command = Torques(tau_d.tolist())
                torque_command.motion_finished = False

                active_control.writeOnce(torque_command)

        except Exception as e:
            print(f"\nError occurred: {e}")

            if robot is not None:
                robot.stop()

            return -1

        return 0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        type=str,
        default="right",
        help="Robot name: left or right",
    )

    args = parser.parse_args()

    rclpy.init()

    node = FrankaOmegaReal(args.name)

    ret = node.run()

    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()

    return ret


if __name__ == "__main__":
    sys.exit(main())