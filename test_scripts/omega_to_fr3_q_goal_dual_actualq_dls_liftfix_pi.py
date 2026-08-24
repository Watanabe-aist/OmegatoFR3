#!/usr/bin/env python3
"""
Actual-Q Normal DLS + bilateral force feedback
+ lifting-entry Cartesian target continuity.

変更点
------
1. TimeSeries IKは使わない。
2. 毎周期、通常DLS IKを最新の実FR3 qから解く。
3. lifting開始時だけ、
   「現在の実EE pose」ではなく
   「直前の通常モードで使っていた target_pos / target_rot」
   をlifting初期基準として引き継ぐ。

目的
----
通常把持中:
    target_pos = Omega操作 + grip position control を含む現在の目標

lifting ON:
    そのtargetをそのままlift基準にする

これにより lifting ON の瞬間に
    q_goal -> q_actual
へ戻って腕間が開く動作を防ぐ。

起動例
------
python3 omega_to_fr3_q_goal_dual_actualq_dls_liftfix_pi.py \
  --name dual \
  --force-mode wrench \
  --force-gain-z 0.12
"""

from __future__ import annotations

import numpy as np

from sensor_msgs.msg import JointState

import pinocchio as pin

import omega_to_fr3_q_goal_dual as base


# ==========================================================
# Experiment 1:
# 中間リンク操作はOFF
# ==========================================================
base.ENABLE_RIGHT_MIDPOINT_CONTROL = False
base.ENABLE_LEFT_MIDPOINT_CONTROL = False


# ==========================================================
# 実FR3 joint state topic
# ==========================================================
RIGHT_FR3_STATE_TOPIC = "/franka/right/obs_franka_state"
LEFT_FR3_STATE_TOPIC = "/franka/left/obs_franka_state"


# ==========================================================
# PI grasp assist
# ==========================================================
# P: force不足に対して即時に閉じる量 [m/N]
# 8 N不足 -> 16 mm total closing command
PI_GRASP_ASSIST_KP = 0.002

# I: 残ったforce誤差をゆっくり詰める [m/(N*s)]
# 8 N不足 -> 8 mm/s total closing command increase
PI_GRASP_ASSIST_KI = 0.001

# P項だけで一気に入れてよい最大量 [m]
# total hand-distance reduction; each arm gets half
PI_GRASP_ASSIST_P_MAX = 0.030

# I項が蓄積してよい最大量 [m]
PI_GRASP_ASSIST_I_MAX = 0.070

# P + I の最終総補正量上限 [m]
# 最初のPI試験では100 mmに制限する
PI_GRASP_ASSIST_TOTAL_MAX = 0.100


class ActualQArmIK(base.ArmIK):
    """
    base.ArmIK.solve_ik() の通常逐次DLSを使用。

    毎回solve開始前だけ
        self.q <- measured actual q
    とする。
    """

    def __init__(
        self,
        node,
        arm_name,
        q_goal_pub,
        enable_midpoint_control=False,
    ):
        super().__init__(
            node=node,
            arm_name=arm_name,
            q_goal_pub=q_goal_pub,
            enable_midpoint_control=enable_midpoint_control,
        )

        self.measured_q_latest = None
        self.measured_q_initialized = False

        self.node.get_logger().warn(
            f"{self.arm_name}: "
            "ACTUAL-Q NORMAL DLS mode enabled"
        )

    def set_measured_q(
        self,
        q_measured,
        omega_state=None,
    ):
        q7 = np.asarray(
            q_measured,
            dtype=float,
        ).reshape(-1)

        if q7.size < 7:
            return False

        q7 = q7[:7].copy()

        if not np.all(np.isfinite(q7)):
            return False

        first = self.measured_q_latest is None

        self.measured_q_latest = q7.copy()

        if not first:
            return True

        # ==================================================
        # 初回だけ実QをFR3基準姿勢にする
        # ==================================================
        self.sync_to_measured_q()

        pin.forwardKinematics(
            self.model,
            self.data,
            self.q,
        )
        pin.updateFramePlacements(
            self.model,
            self.data,
        )

        ee = self.data.oMf[
            self.frame_id
        ]

        self.fr3_initial_pos = (
            ee.translation.copy()
        )
        self.fr3_initial_rot = (
            ee.rotation.copy()
        )

        self.target_pos = (
            self.fr3_initial_pos.copy()
        )
        self.target_rot = (
            self.fr3_initial_rot.copy()
        )

        self.prev_published_q = q7.copy()

        if (
            self.enable_midpoint_control
            and self.mid_frame_id is not None
        ):
            mid = self.data.oMf[
                self.mid_frame_id
            ]
            mid_pos = (
                mid.translation.copy()
            )

            self.mid_initial_pos = (
                mid_pos.copy()
            )
            self.mid_target_pos = (
                mid_pos.copy()
            )
            self.mid_base_pos = (
                mid_pos.copy()
            )
            self.mid_activation_pos = (
                mid_pos.copy()
            )
            self.mid_reference_pos = (
                mid_pos.copy()
            )

            self.mid_reference_delta_fr3 = (
                np.zeros(3)
            )
            self.mid_reference_gripper_delta = 0.0
            self.mid_move_amount = 0.0

        # Omegaが先に初期化されていた場合、
        # 実Q基準を作った瞬間をOmega原点にもする。
        if (
            omega_state is not None
            and omega_state.initialized
        ):
            omega_state.initial_pos = (
                omega_state.pos.copy()
            )
            omega_state.initial_rot = (
                omega_state.rot.copy()
            )

        self.measured_q_initialized = True

        self.node.get_logger().warn(
            f"{self.arm_name}: "
            "actual-q reference initialized, "
            f"q_actual={np.round(q7, 4).tolist()}, "
            f"ee_pos={np.round(self.fr3_initial_pos, 4).tolist()}"
        )

        return True

    def sync_to_measured_q(self):
        """
        Pinocchioの現在状態 self.q を最新実Qへ同期。
        """
        if self.measured_q_latest is None:
            return False

        self.q[:7] = (
            self.measured_q_latest.copy()
        )

        if self.q.size > 7:
            self.q[7] = 0.02

        if self.q.size > 8:
            self.q[8] = 0.02

        return True

    def solve_ik(self):
        """
        最新実Qから通常DLSを解く。
        """
        if not self.sync_to_measured_q():
            return (
                self.q,
                None,
                None,
                None,
                None,
                None,
            )

        return super().solve_ik()


# ==========================================================
# base nodeがActualQArmIKを生成するように差し替える
# ==========================================================
base.ArmIK = ActualQArmIK


class ActualQOmegaToFR3QGoal(
    base.OmegaToFR3QGoal
):
    """
    実Q購読 + lifting開始時target連続化。
    """

    def __init__(
        self,
        mode,
    ):
        super().__init__(mode)

        if self.mode in [
            "right",
            "dual",
        ]:
            self.create_subscription(
                JointState,
                RIGHT_FR3_STATE_TOPIC,
                self.right_fr3_state_callback,
                10,
            )

        if self.mode in [
            "left",
            "dual",
        ]:
            self.create_subscription(
                JointState,
                LEFT_FR3_STATE_TOPIC,
                self.left_fr3_state_callback,
                10,
            )

        self.get_logger().warn(
            "ACTUAL-Q NORMAL DLS NODE: "
            "TimeSeries IK is NOT used."
        )

        self.get_logger().warn(
            "LIFTING ENTRY FIX: "
            "pre-lifting Cartesian targets are preserved."
        )

        # PI assist internal state.
        self.grasp_assist_integral_offset = 0.0

        self.get_logger().warn(
            "GRASP ASSIST PI ENABLED: "
            f"Kp={PI_GRASP_ASSIST_KP:.4f} m/N, "
            f"Ki={PI_GRASP_ASSIST_KI:.4f} m/(N*s), "
            f"Pmax={PI_GRASP_ASSIST_P_MAX*1000.0:.1f} mm, "
            f"Imax={PI_GRASP_ASSIST_I_MAX*1000.0:.1f} mm, "
            f"total_max={PI_GRASP_ASSIST_TOTAL_MAX*1000.0:.1f} mm"
        )

    # ======================================================
    # 実Q受信
    # ======================================================
    def right_fr3_state_callback(
        self,
        msg,
    ):
        if self.right_arm is None:
            return

        if len(msg.position) < 7:
            return

        self.right_arm.set_measured_q(
            msg.position[:7],
            omega_state=self.right_omega,
        )

    def left_fr3_state_callback(
        self,
        msg,
    ):
        if self.left_arm is None:
            return

        if len(msg.position) < 7:
            return

        self.left_arm.set_measured_q(
            msg.position[:7],
            omega_state=self.left_omega,
        )

    def actual_q_ready(self):
        if self.mode in [
            "right",
            "dual",
        ]:
            if (
                self.right_arm is None
                or not self.right_arm.measured_q_initialized
            ):
                return False

        if self.mode in [
            "left",
            "dual",
        ]:
            if (
                self.left_arm is None
                or not self.left_arm.measured_q_initialized
            ):
                return False

        return True

    # ======================================================
    # PI grasp assist
    # ======================================================
    def reset_grasp_assist(self):
        """
        Base側のforce target / offset resetに加えて、
        PIの積分状態も0へ戻す。
        """
        super().reset_grasp_assist()
        self.grasp_assist_integral_offset = 0.0

    def initialize_grasp_assist_target(
        self,
        right_pos=None,
        left_pos=None,
    ):
        """
        lifting開始時の把持力をtarget基準として初期化。
        積分項は毎回0から開始する。
        """
        ok = super().initialize_grasp_assist_target(
            right_pos=right_pos,
            left_pos=left_pos,
        )

        if ok:
            self.grasp_assist_integral_offset = 0.0
            self.grasp_assist_offset = 0.0

        return ok

    def apply_grasp_assist(
        self,
        right_target_pos,
        left_target_pos,
    ):
        """
        One-sided PI grasp assist.

        force error:
            e = F_target - F_filtered

        P term:
            x_P = Kp * max(e, 0)

        I term:
            x_I <- x_I + Ki * e * dt

        total:
            x_assist = x_P + x_I

        P項はforce不足時だけ即座に閉じ方向へ作用する。
        forceがtargetを上回ったときにP項で急に腕を開かない。

        I項はforce過大時にはゆっくり減少できるため、
        積分飽和を戻せる。
        """
        if not base.ENABLE_GRASP_ASSIST:
            return (
                right_target_pos,
                left_target_pos,
            )

        if self.grasp_force_target is None:
            initialized = (
                self.initialize_grasp_assist_target(
                    right_pos=right_target_pos,
                    left_pos=left_target_pos,
                )
            )

            if not initialized:
                return (
                    right_target_pos,
                    left_target_pos,
                )

        if not self.grip_force_received:
            return (
                right_target_pos,
                left_target_pos,
            )

        grip_force_raw = max(
            0.0,
            float(self.grip_force_input_n),
        )

        # --------------------------------------------------
        # Physical grasp axis in common WORLD
        # --------------------------------------------------
        grip_axis_world = (
            self.get_grasp_axis_from_local_positions(
                right_pos_local=right_target_pos,
                left_pos_local=left_target_pos,
            )
        )

        # WORLD vector -> each FR3 local/base frame
        left_grip_axis_local = (
            base.common_world_vector_to_local(
                "left",
                grip_axis_world,
            )
        )

        right_grip_axis_local = (
            base.common_world_vector_to_local(
                "right",
                grip_axis_world,
            )
        )

        # --------------------------------------------------
        # Same LPF as existing controller
        # --------------------------------------------------
        self.grasp_force_filtered = (
            (1.0 - base.GRASP_FORCE_LPF_ALPHA)
            * self.grasp_force_filtered
            + base.GRASP_FORCE_LPF_ALPHA
            * grip_force_raw
        )

        force_error = (
            self.grasp_force_target
            - self.grasp_force_filtered
        )

        if (
            abs(force_error)
            < base.GRASP_FORCE_DEADBAND_N
        ):
            force_error = 0.0

        # --------------------------------------------------
        # P term
        # Immediate closing only when force is insufficient.
        # Never instantaneously open because of negative error.
        # --------------------------------------------------
        proportional_offset = (
            PI_GRASP_ASSIST_KP
            * max(force_error, 0.0)
        )

        proportional_offset = float(
            np.clip(
                proportional_offset,
                0.0,
                PI_GRASP_ASSIST_P_MAX,
            )
        )

        # --------------------------------------------------
        # I term
        # Positive error -> close more
        # Negative error -> slowly unwind
        # --------------------------------------------------
        integral_step = (
            PI_GRASP_ASSIST_KI
            * force_error
            * base.CONTROL_DT
        )

        # Retain the existing per-cycle safety limiter.
        integral_step = float(
            np.clip(
                integral_step,
                -base.GRASP_ASSIST_MAX_STEP,
                base.GRASP_ASSIST_MAX_STEP,
            )
        )

        self.grasp_assist_integral_offset += (
            integral_step
        )

        self.grasp_assist_integral_offset = float(
            np.clip(
                self.grasp_assist_integral_offset,
                0.0,
                PI_GRASP_ASSIST_I_MAX,
            )
        )

        # --------------------------------------------------
        # Total PI assist command
        # --------------------------------------------------
        self.grasp_assist_offset = float(
            np.clip(
                proportional_offset
                + self.grasp_assist_integral_offset,
                0.0,
                PI_GRASP_ASSIST_TOTAL_MAX,
            )
        )

        # Total requested hand-distance reduction is
        # self.grasp_assist_offset.
        # Each arm receives half.
        left_target_pos_assisted = (
            left_target_pos
            + 0.5
            * self.grasp_assist_offset
            * left_grip_axis_local
        )

        right_target_pos_assisted = (
            right_target_pos
            - 0.5
            * self.grasp_assist_offset
            * right_grip_axis_local
        )

        if self.loop_count % 500 == 0:
            self.get_logger().info(
                "GRASP ASSIST PI: "
                f"source={base.GRIP_FORCE_TOPIC}, "
                f"F_raw={grip_force_raw:.3f} N, "
                f"F_filtered={self.grasp_force_filtered:.3f} N, "
                f"F_target={self.grasp_force_target:.3f} N, "
                f"err={force_error:.3f} N, "
                f"P={proportional_offset*1000.0:.2f} mm, "
                f"I={self.grasp_assist_integral_offset*1000.0:.2f} mm, "
                f"total={self.grasp_assist_offset*1000.0:.2f} mm, "
                f"axis={grip_axis_world.round(3).tolist()}"
            )

        return (
            right_target_pos_assisted,
            left_target_pos_assisted,
        )

    # ======================================================
    # lifting開始時の連続化
    # ======================================================
    def activate_lifting_mode(self):
        """
        元コードのactivate_lifting_mode()は
        get_current_ee_pose()からlift初期位置を作る。

        実Q DLSでは、
          通常モード直前:
              target_pos != actual EE pose
        になることがある。

        その状態でactual EE poseをlift基準にすると、
        lifting ON直後のtargetがactual poseへ戻り、
        q_goalがactual q付近へ飛んで腕が開く。

        対策:
          super().activate_lifting_mode()を実行した後、
          lift基準だけを「ON直前のtarget_pos/target_rot」
          へ置き換える。

        把持力目標の初期化はsuper側をそのまま使用する。
        """

        if not self.can_use_lifting_mode():
            return super().activate_lifting_mode()

        # --------------------------------------------------
        # lifting ON直前のCartesian targetを保存
        # --------------------------------------------------
        right_target_pos_before = (
            self.right_arm.target_pos.copy()
        )
        left_target_pos_before = (
            self.left_arm.target_pos.copy()
        )

        right_target_rot_before = (
            self.right_arm.target_rot.copy()
        )
        left_target_rot_before = (
            self.left_arm.target_rot.copy()
        )

        # 診断用: 実EE poseも取得
        right_actual_pos_before, _ = (
            self.right_arm.get_current_ee_pose()
        )
        left_actual_pos_before, _ = (
            self.left_arm.get_current_ee_pose()
        )

        # --------------------------------------------------
        # 既存処理:
        # - lifting flag
        # - Omega基準
        # - grasp assist reset/init
        # - midpoint rebase
        # 等はそのまま使う
        # --------------------------------------------------
        ok = super().activate_lifting_mode()

        if not ok:
            return False

        # --------------------------------------------------
        # ここだけ上書き:
        # lift初期pose = ON直前Cartesian target
        # --------------------------------------------------
        self.lift_right_pos0 = (
            right_target_pos_before.copy()
        )
        self.lift_left_pos0 = (
            left_target_pos_before.copy()
        )

        self.lift_right_rot0 = (
            right_target_rot_before.copy()
        )
        self.lift_left_rot0 = (
            left_target_rot_before.copy()
        )

        right_target_world = (
            self.local_to_common_world(
                "right",
                right_target_pos_before,
            )
        )

        left_target_world = (
            self.local_to_common_world(
                "left",
                left_target_pos_before,
            )
        )

        center_target_world = 0.5 * (
            right_target_world
            + left_target_world
        )

        self.lift_center_pos0 = (
            center_target_world.copy()
        )

        self.lift_right_rel_pos = (
            right_target_world
            - center_target_world
        )

        self.lift_left_rel_pos = (
            left_target_world
            - center_target_world
        )

        # --------------------------------------------------
        # 1周期目のtargetも明示的に直前targetへ保持
        # grasp_assist_offsetはactivate時に0へreset済み。
        # --------------------------------------------------
        self.right_arm.set_lifting_target_from_pose(
            right_target_pos_before,
            right_target_rot_before,
        )

        self.left_arm.set_lifting_target_from_pose(
            left_target_pos_before,
            left_target_rot_before,
        )

        # --------------------------------------------------
        # 診断ログ
        # --------------------------------------------------
        right_actual_world = (
            self.local_to_common_world(
                "right",
                right_actual_pos_before,
            )
        )

        left_actual_world = (
            self.local_to_common_world(
                "left",
                left_actual_pos_before,
            )
        )

        actual_distance = float(
            np.linalg.norm(
                right_actual_world
                - left_actual_world
            )
        )

        target_distance = float(
            np.linalg.norm(
                right_target_world
                - left_target_world
            )
        )

        right_gap_mm = 1000.0 * float(
            np.linalg.norm(
                right_target_pos_before
                - right_actual_pos_before
            )
        )

        left_gap_mm = 1000.0 * float(
            np.linalg.norm(
                left_target_pos_before
                - left_actual_pos_before
            )
        )

        self.get_logger().warn(
            "LIFTING ENTRY TARGET CONTINUITY: "
            f"actual_hand_distance={actual_distance:.4f} m, "
            f"target_hand_distance={target_distance:.4f} m, "
            f"right_target_actual_gap={right_gap_mm:.2f} mm, "
            f"left_target_actual_gap={left_gap_mm:.2f} mm"
        )

        return True

    # ======================================================
    # Main loop
    # ======================================================
    def loop(self):
        # 実Qが来るまではq_goalを出さない
        if not self.actual_q_ready():
            return

        # この周期のFK/Jacobianは実Q基準
        if self.right_arm is not None:
            self.right_arm.sync_to_measured_q()

        if self.left_arm is not None:
            self.left_arm.sync_to_measured_q()

        super().loop()


# ==========================================================
# bilateral側がこのnode classを継承するよう差し替える
# ==========================================================
base.OmegaToFR3QGoal = (
    ActualQOmegaToFR3QGoal
)


# bilateral force feedbackは現在のものをそのまま使用
import omega_to_fr3_q_goal_dual_bilateral as bilateral  # noqa: E402


if __name__ == "__main__":
    bilateral.main()
