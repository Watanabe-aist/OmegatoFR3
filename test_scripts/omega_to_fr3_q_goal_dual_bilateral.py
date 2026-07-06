#!/usr/bin/env python3
"""
omega_to_fr3_q_goal_dual_integrated_factr_style.py

目的:
  既存の omega_to_fr3_q_goal_dual.py の q_goal 制御に，
  FR3外力 -> Omega力覚返しを同じROSノード内で統合する。
  FACTRを参考に，Omega速度に対するダンピングと任意の重力/バイアス補償を加える。

使い方:
  1. このファイルを ~/franka_ros2_ws/test_scripts/ に置く
  2. 既存の omega_to_fr3_q_goal_dual.py と
     franka_wrench_to_omega_force_cmd.py は起動しない
  3. 以下を実行:
       python3 ~/franka_ros2_ws/test_scripts/omega_to_fr3_q_goal_dual_integrated_factr_style.py --name dual --force-mode wrench

注意:
  - Omega driver right/left は今まで通り別ターミナルで起動する。
  - franka_omega_real.py right/left も今まで通り別ターミナルで起動する。
  - force_cmd はこのノードが /right/force_cmd, /left/force_cmd に出す。
"""

import argparse
import numpy as np

import rclpy
from geometry_msgs.msg import WrenchStamped

# 既存の巨大なq_goal制御コードをそのまま利用する。
# このファイルを test_scripts/ に置けば，隣の omega_to_fr3_q_goal_dual.py をimportできる。
from omega_to_fr3_q_goal_dual import (  # noqa: E402
    OmegaToFR3QGoal,
    RIGHT_EE_WRENCH_TOPIC,
    LEFT_EE_WRENCH_TOPIC,
)


# ==========================================================
# Omega force command topic
# ==========================================================
RIGHT_OMEGA_FORCE_CMD_TOPIC = "/right/force_cmd"
LEFT_OMEGA_FORCE_CMD_TOPIC = "/left/force_cmd"


class ForceBiasEstimator:
    def __init__(self, sample_count: int):
        self.sample_count = int(sample_count)
        self.samples = []
        self.bias = np.zeros(3, dtype=float)
        self.ready = self.sample_count <= 0

    def update(self, force: np.ndarray) -> None:
        if self.ready:
            return

        self.samples.append(force.copy())

        if len(self.samples) >= self.sample_count:
            self.bias = np.mean(np.array(self.samples), axis=0)
            self.ready = True

    def remove_bias(self, force: np.ndarray) -> np.ndarray:
        if not self.ready:
            return np.zeros(3, dtype=float)

        return force - self.bias


class IntegratedBilateralOmegaToFR3FactrStyle(OmegaToFR3QGoal):
    """
    q_goal制御とFR3外力->Omega力覚返しを同じノードにまとめたFACTR参考版。

    ポイント:
      - 既存の OmegaToFR3QGoal.loop() を呼んだ直後に force_cmd をpublishする。
      - そのため，q_goalとforce_cmdは同じPythonプロセス，同じROSノード，同じ制御周期側で動く。
      - もう franka_wrench_to_omega_force_cmd.py は起動しない。
      - FACTRの torque_feedback と同じ考えで，力覚に速度ダンピングを入れる。
    """

    def __init__(
        self,
        mode: str,
        force_mode: str,
        constant_force: np.ndarray,
        enable_xy_force: bool,
        force_gain_xy: float,
        force_gain_z: float,
        force_limit: float,
        force_deadband: float,
        lpf_alpha: float,
        bias_sample_count: int,
        force_sign: np.ndarray,
        publish_every_n_loops: int,
        force_damping_xy: float,
        force_damping_z: float,
        omega_gravity_comp: np.ndarray,
    ):
        self.force_mode = force_mode
        self.constant_force = constant_force.astype(float)
        self.enable_xy_force = bool(enable_xy_force)
        self.force_gain_xy = float(force_gain_xy)
        self.force_gain_z = float(force_gain_z)
        self.force_limit = float(force_limit)
        self.force_deadband = float(force_deadband)
        self.lpf_alpha = float(lpf_alpha)
        self.force_sign = force_sign.astype(float)
        self.publish_every_n_loops = max(1, int(publish_every_n_loops))
        self.force_damping_xy = float(force_damping_xy)
        self.force_damping_z = float(force_damping_z)
        self.omega_gravity_comp = omega_gravity_comp.astype(float)

        self.right_force_raw_for_feedback = np.zeros(3, dtype=float)
        self.left_force_raw_for_feedback = np.zeros(3, dtype=float)
        self.right_force_lpf = np.zeros(3, dtype=float)
        self.left_force_lpf = np.zeros(3, dtype=float)
        self.right_force_received_for_feedback = False
        self.left_force_received_for_feedback = False

        self.right_bias = ForceBiasEstimator(bias_sample_count)
        self.left_bias = ForceBiasEstimator(bias_sample_count)

        # ここで既存のq_goalノードを初期化する。
        # 既存ノード側がtimerを作る場合，self.loopはこのクラスのoverride版が呼ばれる。
        super().__init__(mode)

        if self.force_mode not in ["zero", "constant", "wrench"]:
            raise ValueError("--force-mode must be zero, constant, or wrench")

        self.right_force_cmd_pub = self.create_publisher(
            WrenchStamped,
            RIGHT_OMEGA_FORCE_CMD_TOPIC,
            10,
        )
        self.left_force_cmd_pub = self.create_publisher(
            WrenchStamped,
            LEFT_OMEGA_FORCE_CMD_TOPIC,
            10,
        )

        # 既存コードも把持アシスト用に同じwrenchを購読しているが，
        # 力覚返し用には独立したraw値として保持する。
        self.create_subscription(
            WrenchStamped,
            RIGHT_EE_WRENCH_TOPIC,
            self.right_force_feedback_callback,
            10,
        )
        self.create_subscription(
            WrenchStamped,
            LEFT_EE_WRENCH_TOPIC,
            self.left_force_feedback_callback,
            10,
        )

        self.get_logger().warn(
            "INTEGRATED BILATERAL MODE: q_goal and force_cmd are in one node. "
            "Do NOT run franka_wrench_to_omega_force_cmd.py."
        )
        self.get_logger().info(f"force_mode = {self.force_mode}")
        self.get_logger().info(f"right force_cmd topic = {RIGHT_OMEGA_FORCE_CMD_TOPIC}")
        self.get_logger().info(f"left  force_cmd topic = {LEFT_OMEGA_FORCE_CMD_TOPIC}")
        self.get_logger().info(
            "force params: "
            f"enable_xy={self.enable_xy_force}, "
            f"gain_xy={self.force_gain_xy}, "
            f"gain_z={self.force_gain_z}, "
            f"limit={self.force_limit}, "
            f"deadband={self.force_deadband}, "
            f"lpf_alpha={self.lpf_alpha}, "
            f"bias_samples={bias_sample_count}, "
            f"publish_every_n_loops={self.publish_every_n_loops}, "
            f"damping_xy={self.force_damping_xy}, "
            f"damping_z={self.force_damping_z}, "
            f"omega_gravity_comp={self.omega_gravity_comp.round(3).tolist()}"
        )

        if self.force_mode == "wrench" and bias_sample_count > 0:
            self.get_logger().warn(
                "Keep both FR3 end-effectors unloaded during startup until bias estimation finishes."
            )

    # ------------------------------------------------------
    # FR3 wrench callbacks for force feedback
    # ------------------------------------------------------
    def right_force_feedback_callback(self, msg: WrenchStamped) -> None:
        self.right_force_raw_for_feedback = np.array(
            [
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z,
            ],
            dtype=float,
        )
        self.right_force_received_for_feedback = True

    def left_force_feedback_callback(self, msg: WrenchStamped) -> None:
        self.left_force_raw_for_feedback = np.array(
            [
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z,
            ],
            dtype=float,
        )
        self.left_force_received_for_feedback = True

    # ------------------------------------------------------
    # Omega state utilities
    # ------------------------------------------------------
    def get_omega_linear_velocity(self, side: str) -> np.ndarray:
        """
        Omega実機の手先速度を取り出す。
        既存のOmegaToFR3QGoalが /right/state, /left/state を購読して
        self.right_omega / self.left_omega に保持している前提。
        取得できない場合はゼロを返す。
        """
        try:
            if side == "right":
                omega_state = self.right_omega
            else:
                omega_state = self.left_omega

            return np.array(
                [
                    omega_state.twist.linear.x,
                    omega_state.twist.linear.y,
                    omega_state.twist.linear.z,
                ],
                dtype=float,
            )
        except Exception:
            return np.zeros(3, dtype=float)

    def apply_factr_style_damping_and_comp(self, side: str, force_cmd: np.ndarray) -> np.ndarray:
        """
        FACTRの torque_feedback では，外力フィードバックに
        速度ダンピングを足して暴れを抑えている。
        Omega版ではCartesian力に対して同じことを行う。

            F_cmd <- F_cmd - B * v_omega + F_const

        F_const は必要な場合だけ使う任意のOmega側定数補償。
        通常はゼロでよい。
        """
        vel = self.get_omega_linear_velocity(side)

        damping = np.array(
            [
                self.force_damping_xy,
                self.force_damping_xy,
                self.force_damping_z,
            ],
            dtype=float,
        )

        out = force_cmd - damping * vel + self.omega_gravity_comp
        out = np.clip(out, -self.force_limit, self.force_limit)
        return out

    # ------------------------------------------------------
    # Force command calculation
    # ------------------------------------------------------
    def apply_deadband(self, force: np.ndarray) -> np.ndarray:
        out = force.copy()

        for i in range(3):
            if abs(out[i]) < self.force_deadband:
                out[i] = 0.0

        return out

    def convert_franka_force_to_omega_force(self, force_franka: np.ndarray) -> np.ndarray:
        """
        FR3外力をOmegaに返す力に変換する。

        デフォルトではZ方向のみ返す。
        XYを返したい場合は --enable-xy-force を付ける。
        """
        force_franka = self.apply_deadband(force_franka)

        force_omega = np.zeros(3, dtype=float)

        if self.enable_xy_force:
            force_omega[0] = self.force_sign[0] * self.force_gain_xy * force_franka[0]
            force_omega[1] = self.force_sign[1] * self.force_gain_xy * force_franka[1]

        force_omega[2] = self.force_sign[2] * self.force_gain_z * force_franka[2]

        force_omega = np.clip(
            force_omega,
            -self.force_limit,
            self.force_limit,
        )

        return force_omega

    def compute_force_cmd(self, side: str) -> np.ndarray:
        if self.force_mode == "zero":
            return np.zeros(3, dtype=float)

        if self.force_mode == "constant":
            raw_constant = np.clip(
                self.constant_force,
                -self.force_limit,
                self.force_limit,
            )
            return self.apply_factr_style_damping_and_comp(side, raw_constant)

        if side == "right":
            if not self.right_force_received_for_feedback:
                return np.zeros(3, dtype=float)

            self.right_bias.update(self.right_force_raw_for_feedback)

            if not self.right_bias.ready:
                return np.zeros(3, dtype=float)

            force_without_bias = self.right_bias.remove_bias(
                self.right_force_raw_for_feedback
            )
            raw_cmd = self.convert_franka_force_to_omega_force(force_without_bias)
            raw_cmd = self.apply_factr_style_damping_and_comp("right", raw_cmd)
            self.right_force_lpf = (
                (1.0 - self.lpf_alpha) * self.right_force_lpf
                + self.lpf_alpha * raw_cmd
            )

            return self.right_force_lpf

        if side == "left":
            if not self.left_force_received_for_feedback:
                return np.zeros(3, dtype=float)

            self.left_bias.update(self.left_force_raw_for_feedback)

            if not self.left_bias.ready:
                return np.zeros(3, dtype=float)

            force_without_bias = self.left_bias.remove_bias(
                self.left_force_raw_for_feedback
            )
            raw_cmd = self.convert_franka_force_to_omega_force(force_without_bias)
            raw_cmd = self.apply_factr_style_damping_and_comp("left", raw_cmd)
            self.left_force_lpf = (
                (1.0 - self.lpf_alpha) * self.left_force_lpf
                + self.lpf_alpha * raw_cmd
            )

            return self.left_force_lpf

        return np.zeros(3, dtype=float)

    def make_wrench_msg(self, force_cmd: np.ndarray) -> WrenchStamped:
        msg = WrenchStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "omega_force_cmd_integrated"
        msg.wrench.force.x = float(force_cmd[0])
        msg.wrench.force.y = float(force_cmd[1])
        msg.wrench.force.z = float(force_cmd[2])
        msg.wrench.torque.x = 0.0
        msg.wrench.torque.y = 0.0
        msg.wrench.torque.z = 0.0
        return msg

    def publish_force_feedback_once(self) -> None:
        # q_goal側のloop_countと同期して間引く。
        # publish_every_n_loops=1 ならq_goal loopごとにforce_cmdもpublishする。
        if getattr(self, "loop_count", 0) % self.publish_every_n_loops != 0:
            return

        right_cmd = self.compute_force_cmd("right")
        left_cmd = self.compute_force_cmd("left")

        if self.mode in ["right", "dual"]:
            self.right_force_cmd_pub.publish(self.make_wrench_msg(right_cmd))

        if self.mode in ["left", "dual"]:
            self.left_force_cmd_pub.publish(self.make_wrench_msg(left_cmd))

        # loop_countは既存q_goal側が持っている想定。
        if getattr(self, "loop_count", 0) % 1000 == 0:
            self.get_logger().info(
                "integrated force_cmd "
                f"right={right_cmd.round(3).tolist()} "
                f"left={left_cmd.round(3).tolist()} "
                f"force_mode={self.force_mode}"
            )

    # ------------------------------------------------------
    # Integrated loop
    # ------------------------------------------------------
    def loop(self) -> None:
        """
        既存のq_goal処理を実行した後，同じ周期でforce_cmdを出す。
        """
        super().loop()
        self.publish_force_feedback_once()

    def publish_zero_force_once(self) -> None:
        zero = np.zeros(3, dtype=float)

        try:
            if self.mode in ["right", "dual"]:
                self.right_force_cmd_pub.publish(self.make_wrench_msg(zero))

            if self.mode in ["left", "dual"]:
                self.left_force_cmd_pub.publish(self.make_wrench_msg(zero))
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        type=str,
        default="dual",
        choices=["right", "left", "dual"],
        help="right, left, or dual. For bilateral operation, use dual.",
    )
    parser.add_argument(
        "--force-mode",
        type=str,
        default="wrench",
        choices=["zero", "constant", "wrench"],
        help="zero: publish zero force, constant: fixed test force, wrench: FR3 wrench feedback",
    )
    parser.add_argument("--fx", type=float, default=0.0)
    parser.add_argument("--fy", type=float, default=0.0)
    parser.add_argument("--fz", type=float, default=0.0)

    parser.add_argument("--enable-xy-force", action="store_true")
    parser.add_argument("--force-gain-xy", type=float, default=0.08)
    parser.add_argument("--force-gain-z", type=float, default=0.08)
    parser.add_argument("--force-limit", type=float, default=0.8)
    parser.add_argument("--force-deadband", type=float, default=0.2)
    parser.add_argument("--lpf-alpha", type=float, default=0.20)
    parser.add_argument("--bias-samples", type=int, default=200)
    parser.add_argument("--force-damping-xy", type=float, default=0.0)
    parser.add_argument("--force-damping-z", type=float, default=0.15)
    parser.add_argument("--omega-gravity-comp-x", type=float, default=0.0)
    parser.add_argument("--omega-gravity-comp-y", type=float, default=0.0)
    parser.add_argument("--omega-gravity-comp-z", type=float, default=0.0)
    parser.add_argument("--force-sign-x", type=float, default=1.0)
    parser.add_argument("--force-sign-y", type=float, default=1.0)
    parser.add_argument("--force-sign-z", type=float, default=1.0)
    parser.add_argument(
        "--publish-every-n-loops",
        type=int,
        default=1,
        help="1 means publish force_cmd every q_goal loop. 2 means half rate.",
    )

    args = parser.parse_args()

    constant_force = np.array([args.fx, args.fy, args.fz], dtype=float)
    force_sign = np.array(
        [
            args.force_sign_x,
            args.force_sign_y,
            args.force_sign_z,
        ],
        dtype=float,
    )
    omega_gravity_comp = np.array(
        [
            args.omega_gravity_comp_x,
            args.omega_gravity_comp_y,
            args.omega_gravity_comp_z,
        ],
        dtype=float,
    )

    rclpy.init()

    node = IntegratedBilateralOmegaToFR3FactrStyle(
        mode=args.name,
        force_mode=args.force_mode,
        constant_force=constant_force,
        enable_xy_force=args.enable_xy_force,
        force_gain_xy=args.force_gain_xy,
        force_gain_z=args.force_gain_z,
        force_limit=args.force_limit,
        force_deadband=args.force_deadband,
        lpf_alpha=args.lpf_alpha,
        bias_sample_count=args.bias_samples,
        force_sign=force_sign,
        publish_every_n_loops=args.publish_every_n_loops,
        force_damping_xy=args.force_damping_xy,
        force_damping_z=args.force_damping_z,
        omega_gravity_comp=omega_gravity_comp,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.publish_zero_force_once()
                rclpy.spin_once(node, timeout_sec=0.05)
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
