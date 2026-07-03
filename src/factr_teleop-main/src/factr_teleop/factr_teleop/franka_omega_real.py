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
# 右Frankaが左Frankaから +X 方向に1.0 m離れている場合
RIGHT_BASE_OFFSET_IN_LEFT_WORLD = np.array([0.0, -1.0, 0.0], dtype=float)

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
        self.get_logger().info(f"publish ee_pose        : /franka/{self.name}/ee_pose")
        self.get_logger().info(f"publish ee_wrench_world: /franka/{self.name}/ee_wrench_world")

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

    def publish_ee_pose_and_wrench(self, robot_state):
        self.state_pub_count += 1

        # 1000Hzで出すと重いので、100Hz程度に落とす
        if self.state_pub_count % 10 != 0:
            return

        now = self.get_clock().now().to_msg()

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

        # Franka APIの手先外力推定
        wrench = np.array(robot_state.O_F_ext_hat_K, dtype=float)

        wrench_msg = WrenchStamped()
        wrench_msg.header.stamp = now

        # 左右ロボットが平行・同じ向きなら、
        # 力ベクトルの向きは同じ座標軸として扱える。
        wrench_msg.header.frame_id = "world"

        wrench_msg.wrench.force.x = float(wrench[0])
        wrench_msg.wrench.force.y = float(wrench[1])
        wrench_msg.wrench.force.z = float(wrench[2])

        wrench_msg.wrench.torque.x = float(wrench[3])
        wrench_msg.wrench.torque.y = float(wrench[4])
        wrench_msg.wrench.torque.z = float(wrench[5])

        self.ee_wrench_pub.publish(wrench_msg)

    # ===========================
    # 右ノードだけが使うcallback
    # ===========================
    def left_pose_callback(self, msg):
        self.p_left = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=float)

    def right_pose_callback(self, msg):
        self.p_right = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=float)

    def left_wrench_callback(self, msg):
        self.f_left = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
        ], dtype=float)

    def right_wrench_callback(self, msg):
        self.f_right = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
        ], dtype=float)

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
        p_left_world = self.p_left
        p_right_world = self.p_right + RIGHT_BASE_OFFSET_IN_LEFT_WORLD

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

    def run(self):
        joint_stiffness = np.array([90, 90, 90, 30, 30, 30, 30], dtype=float)
        joint_damping = np.array([2.5 * np.sqrt(k) for k in joint_stiffness])

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

                self.publish_joint_state(self.obs_franka_state_pub, q)

                # 自分のEE位置と外力をpublish
                self.publish_ee_pose_and_wrench(robot_state)

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
                        -joint_damping[i] * dq[i]
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