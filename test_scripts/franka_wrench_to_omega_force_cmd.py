
#!/usr/bin/env python3

import argparse
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped


# ===========================
# Topic設定
# ===========================

RIGHT_EE_WRENCH_TOPIC = "/franka/right/ee_wrench_world"
LEFT_EE_WRENCH_TOPIC = "/franka/left/ee_wrench_world"

RIGHT_OMEGA_FORCE_CMD_TOPIC = "/right/force_cmd"
LEFT_OMEGA_FORCE_CMD_TOPIC = "/left/force_cmd"


# ===========================
# 安全設定
# ===========================

PUBLISH_HZ = 200.0

# 最初はZ方向だけ返す
ENABLE_XY_FORCE_FEEDBACK = False

# FR3外力 -> Omega反力のゲイン
# 最初はかなり小さめ
FORCE_GAIN_XY = 0.1
FORCE_GAIN_Z = 0.1

# Omegaに出す最大力[N]
# 最初は0.3〜0.5Nくらいで確認
OMEGA_FORCE_LIMIT_N = 1.0

# 小さいノイズは0扱い
FORCE_DEADBAND_N = 0.2

# ローパスフィルタ
LPF_ALPHA = 0.05

# 起動直後の外力オフセット推定サンプル数
BIAS_SAMPLE_COUNT = 200

# 符号
# Omegaで逆向きに感じたら，ここを +1.0 / -1.0 で変える
FORCE_SIGN_X = 1.0
FORCE_SIGN_Y = 1.0
FORCE_SIGN_Z = 1.0


class ForceBiasEstimator:
    def __init__(self, sample_count: int):
        self.sample_count = sample_count
        self.samples = []
        self.bias = np.zeros(3)
        self.ready = False

    def update(self, force: np.ndarray):
        if self.ready:
            return

        self.samples.append(force.copy())

        if len(self.samples) >= self.sample_count:
            self.bias = np.mean(np.array(self.samples), axis=0)
            self.ready = True

    def remove_bias(self, force: np.ndarray) -> np.ndarray:
        if not self.ready:
            return np.zeros(3)

        return force - self.bias


class FrankaWrenchToOmegaForceCmd(Node):
    def __init__(self, mode: str, constant_force: np.ndarray):
        super().__init__("franka_wrench_to_omega_force_cmd")

        if mode not in ["wrench", "constant", "zero"]:
            raise ValueError("--mode must be wrench, constant, or zero")

        self.mode = mode
        self.constant_force = constant_force.astype(float)

        self.right_force_raw = np.zeros(3)
        self.left_force_raw = np.zeros(3)

        self.right_force_lpf = np.zeros(3)
        self.left_force_lpf = np.zeros(3)

        self.right_received = False
        self.left_received = False

        self.right_bias = ForceBiasEstimator(BIAS_SAMPLE_COUNT)
        self.left_bias = ForceBiasEstimator(BIAS_SAMPLE_COUNT)

        self.right_pub = self.create_publisher(
            WrenchStamped,
            RIGHT_OMEGA_FORCE_CMD_TOPIC,
            10,
        )

        self.left_pub = self.create_publisher(
            WrenchStamped,
            LEFT_OMEGA_FORCE_CMD_TOPIC,
            10,
        )

        self.create_subscription(
            WrenchStamped,
            RIGHT_EE_WRENCH_TOPIC,
            self.right_wrench_callback,
            10,
        )

        self.create_subscription(
            WrenchStamped,
            LEFT_EE_WRENCH_TOPIC,
            self.left_wrench_callback,
            10,
        )

        self.timer = self.create_timer(
            1.0 / PUBLISH_HZ,
            self.timer_callback,
        )

        self.loop_count = 0

        self.get_logger().info("Franka wrench -> Omega force command node started")
        self.get_logger().info(f"mode = {self.mode}")
        self.get_logger().info(f"right output topic = {RIGHT_OMEGA_FORCE_CMD_TOPIC}")
        self.get_logger().info(f"left output topic  = {LEFT_OMEGA_FORCE_CMD_TOPIC}")

        if self.mode == "constant":
            self.get_logger().warn(
                f"CONSTANT FORCE MODE: force={self.constant_force.tolist()} N"
            )

        if self.mode == "wrench":
            self.get_logger().info(
                f"Bias estimation samples = {BIAS_SAMPLE_COUNT}. "
                "Keep FR3 unloaded during startup."
            )

    def right_wrench_callback(self, msg: WrenchStamped):
        self.right_force_raw = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
        ], dtype=float)

        self.right_received = True

    def left_wrench_callback(self, msg: WrenchStamped):
        self.left_force_raw = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
        ], dtype=float)

        self.left_received = True

    def apply_deadband(self, force: np.ndarray) -> np.ndarray:
        out = force.copy()

        for i in range(3):
            if abs(out[i]) < FORCE_DEADBAND_N:
                out[i] = 0.0

        return out

    def convert_franka_force_to_omega_force(self, force_franka: np.ndarray) -> np.ndarray:
        """
        FR3手先外力をOmegaへ返す力に変換する。

        最初は安全のためZ方向中心。
        XYは ENABLE_XY_FORCE_FEEDBACK=True にしたときだけ返す。
        """

        force_franka = self.apply_deadband(force_franka)

        force_omega = np.zeros(3)

        if ENABLE_XY_FORCE_FEEDBACK:
            force_omega[0] = FORCE_SIGN_X * FORCE_GAIN_XY * force_franka[0]
            force_omega[1] = FORCE_SIGN_Y * FORCE_GAIN_XY * force_franka[1]

        force_omega[2] = FORCE_SIGN_Z * FORCE_GAIN_Z * force_franka[2]

        force_omega = np.clip(
            force_omega,
            -OMEGA_FORCE_LIMIT_N,
            OMEGA_FORCE_LIMIT_N,
        )

        return force_omega

    def make_wrench_msg(self, force_cmd: np.ndarray) -> WrenchStamped:
        msg = WrenchStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "omega_force_cmd"

        msg.wrench.force.x = float(force_cmd[0])
        msg.wrench.force.y = float(force_cmd[1])
        msg.wrench.force.z = float(force_cmd[2])

        msg.wrench.torque.x = 0.0
        msg.wrench.torque.y = 0.0
        msg.wrench.torque.z = 0.0

        return msg

    def compute_cmd_force(self, side: str) -> np.ndarray:
        if self.mode == "zero":
            return np.zeros(3)

        if self.mode == "constant":
            return np.clip(
                self.constant_force,
                -OMEGA_FORCE_LIMIT_N,
                OMEGA_FORCE_LIMIT_N,
            )

        if side == "right":
            if not self.right_received:
                return np.zeros(3)

            self.right_bias.update(self.right_force_raw)

            if not self.right_bias.ready:
                return np.zeros(3)

            force_without_bias = self.right_bias.remove_bias(self.right_force_raw)

            force_cmd_raw = self.convert_franka_force_to_omega_force(
                force_without_bias
            )

            self.right_force_lpf = (
                (1.0 - LPF_ALPHA) * self.right_force_lpf
                + LPF_ALPHA * force_cmd_raw
            )

            return self.right_force_lpf

        if side == "left":
            if not self.left_received:
                return np.zeros(3)

            self.left_bias.update(self.left_force_raw)

            if not self.left_bias.ready:
                return np.zeros(3)

            force_without_bias = self.left_bias.remove_bias(self.left_force_raw)

            force_cmd_raw = self.convert_franka_force_to_omega_force(
                force_without_bias
            )

            self.left_force_lpf = (
                (1.0 - LPF_ALPHA) * self.left_force_lpf
                + LPF_ALPHA * force_cmd_raw
            )

            return self.left_force_lpf

        return np.zeros(3)

    def timer_callback(self):
        self.loop_count += 1

        right_cmd = self.compute_cmd_force("right")
        left_cmd = self.compute_cmd_force("left")

        self.right_pub.publish(self.make_wrench_msg(right_cmd))
        self.left_pub.publish(self.make_wrench_msg(left_cmd))

        if self.loop_count % int(PUBLISH_HZ) == 0:
            self.get_logger().info(
                "force_cmd "
                f"right={right_cmd.round(3).tolist()} "
                f"left={left_cmd.round(3).tolist()} "
                f"mode={self.mode}"
            )



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="zero",
        choices=["zero", "constant", "wrench"],
        help=(
            "zero: publish zero force, "
            "constant: publish fixed test force, "
            "wrench: convert FR3 wrench to Omega force"
        ),
    )

    parser.add_argument("--fx", type=float, default=0.0)
    parser.add_argument("--fy", type=float, default=0.0)
    parser.add_argument("--fz", type=float, default=0.0)

    args = parser.parse_args()

    rclpy.init()

    constant_force = np.array([args.fx, args.fy, args.fz], dtype=float)

    node = FrankaWrenchToOmegaForceCmd(
        mode=args.mode,
        constant_force=constant_force,
    )

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        try:
            if rclpy.ok():
                zero = np.zeros(3)
                node.right_pub.publish(node.make_wrench_msg(zero))
                node.left_pub.publish(node.make_wrench_msg(zero))
                rclpy.spin_once(node, timeout_sec=0.05)
        except Exception as e:
            try:
                node.get_logger().warn(f"failed to publish zero force on shutdown: {e}")
            except Exception:
                pass

        try:
            node.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
