#!/usr/bin/env python3
"""
SII_ex1_assist.py

SII 実験1：把持アシスト OFF / ON 比較用「計測専用」ROS 2 logger

重要:
- このノードはFR3/Omegaへ制御指令を一切publishしない。
- 普段の teleoperation/control code と別ターミナルで同時起動する。
- 制御コードを import しない。
- ROS topic を subscribe してCSVへ保存するだけ。

想定する既存topic:
  /dual_grip/grip_force
  /franka/left/ee_pose
  /franka/right/ee_pose
  /franka/left/ee_wrench_world
  /franka/right/ee_wrench_world
  /left/force_cmd
  /right/force_cmd
  /franka/left/obs_franka_state
  /franka/right/obs_franka_state
  /omega_fr3/left/q_goal
  /omega_fr3/right/q_goal
  /right/state

実験条件は制御しない。
--condition off / on は「この試行がどちらの条件か」をCSVへ記録するだけ。

例:
  python3 ~/franka_ros2_ws/test_scripts/SII_ex1_assist.py \
    --condition off \
    --trial 1 \
    --object box1

  python3 ~/franka_ros2_ws/test_scripts/SII_ex1_assist.py \
    --condition on \
    --trial 1 \
    --object box1
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, WrenchStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from ohrc_msgs.msg import State


# ============================================================
# Topic
# ============================================================

GRIP_FORCE_TOPIC = "/dual_grip/grip_force"
GRASP_ASSIST_OFFSET_TOPIC = "/dual_grip/grasp_assist_offset"

LEFT_EE_POSE_TOPIC = "/franka/left/ee_pose"
RIGHT_EE_POSE_TOPIC = "/franka/right/ee_pose"

LEFT_WRENCH_TOPIC = "/franka/left/ee_wrench_world"
RIGHT_WRENCH_TOPIC = "/franka/right/ee_wrench_world"

LEFT_FORCE_CMD_TOPIC = "/left/force_cmd"
RIGHT_FORCE_CMD_TOPIC = "/right/force_cmd"

LEFT_STATE_TOPIC = "/franka/left/obs_franka_state"
RIGHT_STATE_TOPIC = "/franka/right/obs_franka_state"

LEFT_Q_GOAL_TOPIC = "/omega_fr3/left/q_goal"
RIGHT_Q_GOAL_TOPIC = "/omega_fr3/right/q_goal"

RIGHT_OMEGA_STATE_TOPIC = "/right/state"


LOG_ROOT = (
    Path.home()
    / "franka_ros2_ws"
    / "experiment1_logs"
)


URDF_PATH = (
    Path.home()
    / "franka_ros2_ws"
    / "src"
    / "fr3_urdf"
    / "fr3.urdf"
)


def rot_x(theta_rad: float) -> np.ndarray:
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)

    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ], dtype=float)


def get_xyz_from_pose(msg: PoseStamped) -> np.ndarray:
    return np.array([
        msg.pose.position.x,
        msg.pose.position.y,
        msg.pose.position.z,
    ], dtype=float)


def get_force_from_wrench(msg: WrenchStamped) -> np.ndarray:
    return np.array([
        msg.wrench.force.x,
        msg.wrench.force.y,
        msg.wrench.force.z,
    ], dtype=float)


def get_torque_from_wrench(msg: WrenchStamped) -> np.ndarray:
    return np.array([
        msg.wrench.torque.x,
        msg.wrench.torque.y,
        msg.wrench.torque.z,
    ], dtype=float)


def safe_joint_array(msg: JointState) -> np.ndarray:
    q = np.full(7, np.nan, dtype=float)

    n = min(
        7,
        len(msg.position),
    )

    if n > 0:
        q[:n] = np.asarray(
            msg.position[:n],
            dtype=float,
        )

    return q


class Experiment1Logger(Node):
    def __init__(
        self,
        *,
        condition: str,
        trial: int,
        object_name: str,
        subject: str,
        note: str,
        log_hz: float,
        target_margin_n: float,
        left_roll_deg: float,
        right_roll_deg: float,
        right_base_xyz: np.ndarray,
    ):
        super().__init__(
            "sii_ex1_assist_logger"
        )

        # ====================================================
        # 実験情報
        # ====================================================

        self.condition = condition
        self.trial = int(trial)
        self.object_name = object_name
        self.subject = subject
        self.note = note

        self.log_hz = max(
            1.0,
            float(log_hz),
        )

        # これは「評価用」の定義。
        # 制御器へは一切使用しない。
        self.target_margin_n = float(
            target_margin_n
        )

        # ====================================================
        # 双腕の実験幾何
        # ee_poseは各FR3 base座標でpublishされているため、
        # 共通worldへ変換して手先間距離を計測する。
        # ====================================================

        self.left_rot_world = rot_x(
            math.radians(left_roll_deg)
        )

        self.right_rot_world = rot_x(
            math.radians(right_roll_deg)
        )

        self.left_base_world = np.zeros(
            3,
            dtype=float,
        )

        self.right_base_world = np.asarray(
            right_base_xyz,
            dtype=float,
        ).reshape(3)

        # ====================================================
        # FK診断
        # q_goalと実関節qを同じURDFでFKし，
        # lifting開始時からの相対閉じ込み量を比較する。
        # ====================================================

        if not URDF_PATH.exists():
            raise FileNotFoundError(
                f"FR3 URDF not found: {URDF_PATH}"
            )

        self.fk_model = pin.buildModelFromUrdf(
            str(URDF_PATH)
        )
        self.fk_data = self.fk_model.createData()

        self.fk_frame_name = "fr3_hand_tcp"

        if not self.fk_model.existFrame(
            self.fk_frame_name
        ):
            self.fk_frame_name = "fr3_link8"

        if not self.fk_model.existFrame(
            self.fk_frame_name
        ):
            raise ValueError(
                "FK diagnostic EE frame not found"
            )

        self.fk_frame_id = self.fk_model.getFrameId(
            self.fk_frame_name
        )

        # ====================================================
        # 最新値
        # ====================================================

        self.grip_force_n = np.nan
        self.grasp_assist_offset_m = np.nan

        self.left_ee_local = None
        self.right_ee_local = None

        self.left_force_world = np.full(
            3,
            np.nan,
        )
        self.right_force_world = np.full(
            3,
            np.nan,
        )

        self.left_torque_world = np.full(
            3,
            np.nan,
        )
        self.right_torque_world = np.full(
            3,
            np.nan,
        )

        self.left_force_cmd = np.full(
            3,
            np.nan,
        )
        self.right_force_cmd = np.full(
            3,
            np.nan,
        )

        self.left_q = np.full(
            7,
            np.nan,
        )
        self.right_q = np.full(
            7,
            np.nan,
        )

        self.left_q_goal = np.full(
            7,
            np.nan,
        )
        self.right_q_goal = np.full(
            7,
            np.nan,
        )

        self.right_omega_pos = np.full(
            3,
            np.nan,
        )

        self.right_gripper_angle = np.nan
        self.right_gripper_button = False

        # ====================================================
        # lifting評価状態
        # ====================================================

        self.lifting_mode = False
        self.prev_lifting_mode = False

        self.lift_start_elapsed_s = np.nan

        self.lift_start_grip_force_n = np.nan
        self.eval_grip_target_n = np.nan

        self.lift_start_hand_distance_m = np.nan

        # q_goalと実qは絶対基準が異なり得るので，
        # それぞれlifting開始時のFK距離を独立に保存する。
        self.lift_start_q_goal_fk_distance_m = np.nan
        self.lift_start_q_fk_distance_m = np.nan

        self.current_lifting_rows = []

        # ====================================================
        # Subscriber
        # ====================================================

        self.create_subscription(
            Float64,
            GRIP_FORCE_TOPIC,
            self.grip_force_callback,
            10,
        )
        self.create_subscription(
            Float64,
            GRASP_ASSIST_OFFSET_TOPIC,
            self.grasp_assist_offset_callback,
            10,
        )

        self.create_subscription(
            PoseStamped,
            LEFT_EE_POSE_TOPIC,
            self.left_pose_callback,
            10,
        )

        self.create_subscription(
            PoseStamped,
            RIGHT_EE_POSE_TOPIC,
            self.right_pose_callback,
            10,
        )

        self.create_subscription(
            WrenchStamped,
            LEFT_WRENCH_TOPIC,
            self.left_wrench_callback,
            10,
        )

        self.create_subscription(
            WrenchStamped,
            RIGHT_WRENCH_TOPIC,
            self.right_wrench_callback,
            10,
        )

        self.create_subscription(
            WrenchStamped,
            LEFT_FORCE_CMD_TOPIC,
            self.left_force_cmd_callback,
            10,
        )

        self.create_subscription(
            WrenchStamped,
            RIGHT_FORCE_CMD_TOPIC,
            self.right_force_cmd_callback,
            10,
        )

        self.create_subscription(
            JointState,
            LEFT_STATE_TOPIC,
            self.left_state_callback,
            10,
        )

        self.create_subscription(
            JointState,
            RIGHT_STATE_TOPIC,
            self.right_state_callback,
            10,
        )

        self.create_subscription(
            JointState,
            LEFT_Q_GOAL_TOPIC,
            self.left_q_goal_callback,
            10,
        )

        self.create_subscription(
            JointState,
            RIGHT_Q_GOAL_TOPIC,
            self.right_q_goal_callback,
            10,
        )

        self.create_subscription(
            State,
            RIGHT_OMEGA_STATE_TOPIC,
            self.right_omega_callback,
            10,
        )

        # ====================================================
        # CSV
        # ====================================================

        LOG_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

        now = datetime.now()
        stamp = now.strftime(
            "%Y%m%d_%H%M%S"
        )

        safe_object = (
            object_name
            .replace("/", "_")
            .replace(" ", "_")
        )

        safe_subject = (
            subject
            .replace("/", "_")
            .replace(" ", "_")
        )

        filename = (
            f"{stamp}_"
            f"assist_{condition}_"
            f"trial{self.trial:02d}_"
            f"{safe_object}_"
            f"{safe_subject}.csv"
        )

        self.csv_path = (
            LOG_ROOT
            / filename
        )

        self.csv_file = self.csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        )

        self.fieldnames = self.make_fieldnames()

        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=self.fieldnames,
        )

        self.writer.writeheader()
        self.csv_file.flush()

        self.start_monotonic = (
            time.monotonic()
        )

        self.row_count = 0

        self.timer = self.create_timer(
            1.0 / self.log_hz,
            self.log_row,
        )

        # ====================================================
        # 起動表示
        # ====================================================

        self.get_logger().info(
            "=========================================="
        )
        self.get_logger().info(
            "SII Experiment 1 LOGGER ONLY"
        )
        self.get_logger().info(
            "NO control command is published."
        )
        self.get_logger().info(
            f"condition = {self.condition}"
        )
        self.get_logger().info(
            f"trial = {self.trial}"
        )
        self.get_logger().info(
            f"object = {self.object_name}"
        )
        self.get_logger().info(
            f"target margin for evaluation = "
            f"{self.target_margin_n:.3f} N"
        )
        self.get_logger().info(
            f"log = {self.csv_path}"
        )
        self.get_logger().info(
            "=========================================="
        )

    # ========================================================
    # CSV列
    # ========================================================

    def make_fieldnames(self):
        fields = [
            "wall_time",
            "elapsed_s",

            "condition",
            "trial",
            "object",
            "subject",
            "note",

            "lifting_mode",

            "grip_force_N",
            "grasp_assist_offset_m",
            "grasp_assist_offset_mm",
            "lift_start_grip_force_N",
            "eval_grip_target_N",
            "grip_error_to_eval_target_N",
            "below_eval_target",

            "hand_distance_m",
            "lift_start_hand_distance_m",

            # lifting開始時から実際に手先間距離が
            # どれだけ縮んだかという観測量
            "effective_closing_m",
            "effective_closing_mm",

            "assist_target_hand_distance_m",

            "q_goal_fk_hand_distance_m",
            "lift_start_q_goal_fk_distance_m",
            "q_goal_fk_closing_m",
            "q_goal_fk_closing_mm",

            "q_fk_hand_distance_m",
            "lift_start_q_fk_distance_m",
            "q_fk_closing_m",
            "q_fk_closing_mm",

            "ik_unrealized_closing_mm",
            "robot_unrealized_closing_mm",

            "left_ee_world_x_m",
            "left_ee_world_y_m",
            "left_ee_world_z_m",

            "right_ee_world_x_m",
            "right_ee_world_y_m",
            "right_ee_world_z_m",

            "grip_axis_x",
            "grip_axis_y",
            "grip_axis_z",

            "left_fx_N",
            "left_fy_N",
            "left_fz_N",

            "right_fx_N",
            "right_fy_N",
            "right_fz_N",

            "left_tx_Nm",
            "left_ty_Nm",
            "left_tz_Nm",

            "right_tx_Nm",
            "right_ty_Nm",
            "right_tz_Nm",

            "left_force_cmd_x_N",
            "left_force_cmd_y_N",
            "left_force_cmd_z_N",

            "right_force_cmd_x_N",
            "right_force_cmd_y_N",
            "right_force_cmd_z_N",

            "right_omega_x",
            "right_omega_y",
            "right_omega_z",

            "right_gripper_angle",
            "right_gripper_button",
        ]

        for side in [
            "left",
            "right",
        ]:
            for i in range(7):
                fields.append(
                    f"{side}_q{i+1}_rad"
                )

        for side in [
            "left",
            "right",
        ]:
            for i in range(7):
                fields.append(
                    f"{side}_q_goal{i+1}_rad"
                )

        return fields

    # ========================================================
    # callbacks
    # ========================================================

    def grip_force_callback(
        self,
        msg: Float64,
    ):
        self.grip_force_n = float(
            msg.data
        )

    def grasp_assist_offset_callback(
        self,
        msg: Float64,
    ):
        self.grasp_assist_offset_m = float(msg.data)

    def left_pose_callback(
        self,
        msg: PoseStamped,
    ):
        self.left_ee_local = (
            get_xyz_from_pose(msg)
        )

    def right_pose_callback(
        self,
        msg: PoseStamped,
    ):
        self.right_ee_local = (
            get_xyz_from_pose(msg)
        )

    def left_wrench_callback(
        self,
        msg: WrenchStamped,
    ):
        self.left_force_world = (
            get_force_from_wrench(msg)
        )

        self.left_torque_world = (
            get_torque_from_wrench(msg)
        )

    def right_wrench_callback(
        self,
        msg: WrenchStamped,
    ):
        self.right_force_world = (
            get_force_from_wrench(msg)
        )

        self.right_torque_world = (
            get_torque_from_wrench(msg)
        )

    def left_force_cmd_callback(
        self,
        msg: WrenchStamped,
    ):
        self.left_force_cmd = (
            get_force_from_wrench(msg)
        )

    def right_force_cmd_callback(
        self,
        msg: WrenchStamped,
    ):
        self.right_force_cmd = (
            get_force_from_wrench(msg)
        )

    def left_state_callback(
        self,
        msg: JointState,
    ):
        self.left_q = safe_joint_array(
            msg
        )

    def right_state_callback(
        self,
        msg: JointState,
    ):
        self.right_q = safe_joint_array(
            msg
        )

    def left_q_goal_callback(
        self,
        msg: JointState,
    ):
        self.left_q_goal = (
            safe_joint_array(msg)
        )

    def right_q_goal_callback(
        self,
        msg: JointState,
    ):
        self.right_q_goal = (
            safe_joint_array(msg)
        )

    def right_omega_callback(
        self,
        msg: State,
    ):
        try:
            self.right_omega_pos = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ], dtype=float)
        except Exception:
            pass

        try:
            self.right_gripper_angle = float(
                msg.gripper.angle
            )
        except Exception:
            self.right_gripper_angle = np.nan

        try:
            button_now = bool(
                msg.gripper.button
            )
        except Exception:
            button_now = False

        self.right_gripper_button = (
            button_now
        )

        # 現在の制御コードではright gripper buttonの状態が
        # lifting_modeの状態と一致するため、
        # logger側は観測値としてこれを記録する。
        self.lifting_mode = button_now

    # ========================================================
    # 幾何
    # ========================================================

    def get_ee_world_positions(self):
        left_world = None
        right_world = None

        if self.left_ee_local is not None:
            left_world = (
                self.left_base_world
                + self.left_rot_world
                @ self.left_ee_local
            )

        if self.right_ee_local is not None:
            right_world = (
                self.right_base_world
                + self.right_rot_world
                @ self.right_ee_local
            )

        return left_world, right_world

    def get_hand_geometry(self):
        left_world, right_world = (
            self.get_ee_world_positions()
        )

        if (
            left_world is None
            or right_world is None
        ):
            return (
                left_world,
                right_world,
                np.nan,
                np.full(
                    3,
                    np.nan,
                ),
            )

        diff = (
            right_world
            - left_world
        )

        dist = float(
            np.linalg.norm(diff)
        )

        if dist < 1e-12:
            axis = np.full(
                3,
                np.nan,
            )
        else:
            axis = diff / dist

        return (
            left_world,
            right_world,
            dist,
            axis,
        )

    # ========================================================
    # FK診断
    # ========================================================

    def fk_ee_local_from_q7(
        self,
        q7,
    ):
        q7 = np.asarray(
            q7,
            dtype=float,
        ).reshape(-1)

        if q7.size < 7:
            return None

        if not np.all(
            np.isfinite(q7[:7])
        ):
            return None

        q_full = pin.neutral(
            self.fk_model
        )

        q_full[:7] = q7[:7]

        if self.fk_model.nq >= 9:
            q_full[7] = 0.02
            q_full[8] = 0.02

        pin.forwardKinematics(
            self.fk_model,
            self.fk_data,
            q_full,
        )
        pin.updateFramePlacements(
            self.fk_model,
            self.fk_data,
        )

        return (
            self.fk_data
            .oMf[self.fk_frame_id]
            .translation
            .copy()
        )

    def get_fk_hand_distance(
        self,
        left_q7,
        right_q7,
    ):
        left_local = self.fk_ee_local_from_q7(
            left_q7
        )
        right_local = self.fk_ee_local_from_q7(
            right_q7
        )

        if (
            left_local is None
            or right_local is None
        ):
            return np.nan

        left_world = (
            self.left_base_world
            + self.left_rot_world
            @ left_local
        )

        right_world = (
            self.right_base_world
            + self.right_rot_world
            @ right_local
        )

        return float(
            np.linalg.norm(
                right_world
                - left_world
            )
        )

    # ========================================================
    # lifting開始・終了
    # ========================================================

    def start_lifting_section(
        self,
        elapsed_s,
        hand_distance_m,
    ):
        self.lift_start_elapsed_s = (
            elapsed_s
        )

        self.lift_start_grip_force_n = (
            self.grip_force_n
        )

        if math.isfinite(
            self.lift_start_grip_force_n
        ):
            self.eval_grip_target_n = (
                self.lift_start_grip_force_n
                + self.target_margin_n
            )
        else:
            self.eval_grip_target_n = (
                np.nan
            )

        self.lift_start_hand_distance_m = (
            hand_distance_m
        )

        self.lift_start_q_goal_fk_distance_m = (
            self.get_fk_hand_distance(
                self.left_q_goal,
                self.right_q_goal,
            )
        )

        self.lift_start_q_fk_distance_m = (
            self.get_fk_hand_distance(
                self.left_q,
                self.right_q,
            )
        )

        self.current_lifting_rows = []

        self.get_logger().warn(
            "LIFTING START detected: "
            f"F_start="
            f"{self.lift_start_grip_force_n:.3f} N, "
            f"F_eval_target="
            f"{self.eval_grip_target_n:.3f} N, "
            f"distance_start="
            f"{self.lift_start_hand_distance_m:.4f} m"
        )

    def finish_lifting_section(self):
        if not self.current_lifting_rows:
            self.get_logger().warn(
                "LIFTING END detected, "
                "but no valid samples."
            )
            return

        grip = np.array([
            r["grip_force_N"]
            for r in self.current_lifting_rows
        ], dtype=float)

        t = np.array([
            r["elapsed_s"]
            for r in self.current_lifting_rows
        ], dtype=float)

        closing = np.array([
            r["effective_closing_mm"]
            for r in self.current_lifting_rows
        ], dtype=float)

        assist_offset = np.array([
            r["grasp_assist_offset_mm"]
            for r in self.current_lifting_rows
        ], dtype=float)

        valid_grip = np.isfinite(grip)
        valid_closing = np.isfinite(
            closing
        )
        valid_assist_offset = np.isfinite(
            assist_offset
        )

        if len(t) >= 2:
            duration = float(
                t[-1] - t[0]
            )
        else:
            duration = 0.0

        if (
            math.isfinite(
                self.eval_grip_target_n
            )
            and np.any(valid_grip)
        ):
            below = (
                grip
                < self.eval_grip_target_n
            )

            if len(t) >= 2:
                dt = np.diff(
                    t,
                    prepend=t[0],
                )

                dt = np.maximum(
                    dt,
                    0.0,
                )

                below_time = float(
                    np.sum(
                        dt[
                            below
                            & valid_grip
                        ]
                    )
                )
            else:
                below_time = 0.0
        else:
            below_time = np.nan

        if np.any(valid_grip):
            grip_mean = float(
                np.nanmean(grip)
            )
            grip_min = float(
                np.nanmin(grip)
            )
        else:
            grip_mean = np.nan
            grip_min = np.nan

        if np.any(valid_closing):
            closing_max = float(
                np.nanmax(closing)
            )
        else:
            closing_max = np.nan


        if np.any(valid_assist_offset):
            assist_offset_max = float(
                np.nanmax(assist_offset)
            )
        else:
            assist_offset_max = np.nan

        summary = {
            "timestamp": (
                datetime.now().isoformat(
                    timespec="seconds"
                )
            ),
            "condition": self.condition,
            "trial": self.trial,
            "object": self.object_name,
            "subject": self.subject,
            "lifting_duration_s": duration,
            "lift_start_grip_force_N": (
                self.lift_start_grip_force_n
            ),
            "eval_grip_target_N": (
                self.eval_grip_target_n
            ),
            "grip_force_mean_N": (
                grip_mean
            ),
            "grip_force_min_N": (
                grip_min
            ),
            "below_eval_target_time_s": (
                below_time
            ),
            "max_effective_closing_mm": (
                closing_max
            ),
            "max_grasp_assist_offset_mm": (
                assist_offset_max
            ),
            "raw_csv": str(
                self.csv_path
            ),
            "note": self.note,
        }

        summary_path = (
            LOG_ROOT
            / "experiment1_summary.csv"
        )

        fieldnames = list(
            summary.keys()
        )

        new_file = (
            not summary_path.exists()
            or summary_path.stat().st_size == 0
        )

        with summary_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            if new_file:
                writer.writeheader()

            writer.writerow(summary)

        self.get_logger().warn(
            "EXPERIMENT SUMMARY: "
            f"duration={duration:.2f}s, "
            f"mean Fg={grip_mean:.3f}N, "
            f"min Fg={grip_min:.3f}N, "
            f"below target={below_time:.3f}s, "
            f"max effective closing="
            f"{closing_max:.3f}mm, "
            f"max assist cmd="
            f"{assist_offset_max:.3f}mm"
        )

        self.get_logger().warn(
            f"summary saved: "
            f"{summary_path}"
        )

    # ========================================================
    # CSV logging
    # ========================================================

    def log_row(self):
        now_mono = time.monotonic()

        elapsed_s = (
            now_mono
            - self.start_monotonic
        )

        (
            left_world,
            right_world,
            hand_distance_m,
            grip_axis,
        ) = self.get_hand_geometry()

        # lifting立ち上がり
        if (
            self.lifting_mode
            and not self.prev_lifting_mode
        ):
            self.start_lifting_section(
                elapsed_s=elapsed_s,
                hand_distance_m=(
                    hand_distance_m
                ),
            )

        # lifting立ち下がり
        if (
            (not self.lifting_mode)
            and self.prev_lifting_mode
        ):
            self.finish_lifting_section()

        # 評価目標との差
        if (
            math.isfinite(
                self.grip_force_n
            )
            and math.isfinite(
                self.eval_grip_target_n
            )
        ):
            grip_error = (
                self.eval_grip_target_n
                - self.grip_force_n
            )

            below_target = int(
                self.grip_force_n
                < self.eval_grip_target_n
            )
        else:
            grip_error = np.nan
            below_target = 0

        # 実測の手先間距離縮小量
        # 制御内部変数grasp_assist_offsetではない。
        if (
            self.lifting_mode
            and math.isfinite(
                self.lift_start_hand_distance_m
            )
            and math.isfinite(
                hand_distance_m
            )
        ):
            effective_closing_m = (
                self.lift_start_hand_distance_m
                - hand_distance_m
            )
        else:
            effective_closing_m = (
                np.nan
            )

        if math.isfinite(
            effective_closing_m
        ):
            effective_closing_mm = (
                effective_closing_m
                * 1000.0
            )
        else:
            effective_closing_mm = (
                np.nan
            )

        # ====================================================
        # 把持アシスト距離追従の切り分け
        # ====================================================

        q_goal_fk_hand_distance_m = (
            self.get_fk_hand_distance(
                self.left_q_goal,
                self.right_q_goal,
            )
        )

        q_fk_hand_distance_m = (
            self.get_fk_hand_distance(
                self.left_q,
                self.right_q,
            )
        )

        if (
            self.lifting_mode
            and math.isfinite(
                self.lift_start_hand_distance_m
            )
            and math.isfinite(
                self.grasp_assist_offset_m
            )
        ):
            assist_target_hand_distance_m = (
                self.lift_start_hand_distance_m
                - self.grasp_assist_offset_m
            )
        else:
            assist_target_hand_distance_m = np.nan

        if (
            self.lifting_mode
            and math.isfinite(
                self.lift_start_q_goal_fk_distance_m
            )
            and math.isfinite(
                q_goal_fk_hand_distance_m
            )
        ):
            q_goal_fk_closing_m = (
                self.lift_start_q_goal_fk_distance_m
                - q_goal_fk_hand_distance_m
            )
        else:
            q_goal_fk_closing_m = np.nan

        if (
            self.lifting_mode
            and math.isfinite(
                self.lift_start_q_fk_distance_m
            )
            and math.isfinite(
                q_fk_hand_distance_m
            )
        ):
            q_fk_closing_m = (
                self.lift_start_q_fk_distance_m
                - q_fk_hand_distance_m
            )
        else:
            q_fk_closing_m = np.nan

        if (
            math.isfinite(
                self.grasp_assist_offset_m
            )
            and math.isfinite(
                q_goal_fk_closing_m
            )
        ):
            ik_unrealized_closing_mm = (
                self.grasp_assist_offset_m
                - q_goal_fk_closing_m
            ) * 1000.0
        else:
            ik_unrealized_closing_mm = np.nan

        if (
            math.isfinite(
                q_goal_fk_closing_m
            )
            and math.isfinite(
                q_fk_closing_m
            )
        ):
            robot_unrealized_closing_mm = (
                q_goal_fk_closing_m
                - q_fk_closing_m
            ) * 1000.0
        else:
            robot_unrealized_closing_mm = np.nan

        row = {
            "wall_time": (
                datetime.now().isoformat(
                    timespec="milliseconds"
                )
            ),
            "elapsed_s": elapsed_s,

            "condition": self.condition,
            "trial": self.trial,
            "object": self.object_name,
            "subject": self.subject,
            "note": self.note,

            "lifting_mode": int(
                self.lifting_mode
            ),

            "grip_force_N": (
                self.grip_force_n
            ),
            "grasp_assist_offset_m": (
                self.grasp_assist_offset_m
            ),
            "grasp_assist_offset_mm": (
                self.grasp_assist_offset_m * 1000.0
                if math.isfinite(self.grasp_assist_offset_m)
                else np.nan
            ),
            "lift_start_grip_force_N": (
                self.lift_start_grip_force_n
            ),
            "eval_grip_target_N": (
                self.eval_grip_target_n
            ),
            "grip_error_to_eval_target_N": (
                grip_error
            ),
            "below_eval_target": (
                below_target
            ),

            "hand_distance_m": (
                hand_distance_m
            ),
            "lift_start_hand_distance_m": (
                self.lift_start_hand_distance_m
            ),

            "effective_closing_m": (
                effective_closing_m
            ),
            "effective_closing_mm": (
                effective_closing_mm
            ),

            "assist_target_hand_distance_m": (
                assist_target_hand_distance_m
            ),

            "q_goal_fk_hand_distance_m": (
                q_goal_fk_hand_distance_m
            ),
            "lift_start_q_goal_fk_distance_m": (
                self.lift_start_q_goal_fk_distance_m
            ),
            "q_goal_fk_closing_m": (
                q_goal_fk_closing_m
            ),
            "q_goal_fk_closing_mm": (
                q_goal_fk_closing_m * 1000.0
                if math.isfinite(q_goal_fk_closing_m)
                else np.nan
            ),

            "q_fk_hand_distance_m": (
                q_fk_hand_distance_m
            ),
            "lift_start_q_fk_distance_m": (
                self.lift_start_q_fk_distance_m
            ),
            "q_fk_closing_m": (
                q_fk_closing_m
            ),
            "q_fk_closing_mm": (
                q_fk_closing_m * 1000.0
                if math.isfinite(q_fk_closing_m)
                else np.nan
            ),

            "ik_unrealized_closing_mm": (
                ik_unrealized_closing_mm
            ),
            "robot_unrealized_closing_mm": (
                robot_unrealized_closing_mm
            ),

            "left_ee_world_x_m": (
                left_world[0]
                if left_world is not None
                else np.nan
            ),
            "left_ee_world_y_m": (
                left_world[1]
                if left_world is not None
                else np.nan
            ),
            "left_ee_world_z_m": (
                left_world[2]
                if left_world is not None
                else np.nan
            ),

            "right_ee_world_x_m": (
                right_world[0]
                if right_world is not None
                else np.nan
            ),
            "right_ee_world_y_m": (
                right_world[1]
                if right_world is not None
                else np.nan
            ),
            "right_ee_world_z_m": (
                right_world[2]
                if right_world is not None
                else np.nan
            ),

            "grip_axis_x": grip_axis[0],
            "grip_axis_y": grip_axis[1],
            "grip_axis_z": grip_axis[2],

            "left_fx_N": (
                self.left_force_world[0]
            ),
            "left_fy_N": (
                self.left_force_world[1]
            ),
            "left_fz_N": (
                self.left_force_world[2]
            ),

            "right_fx_N": (
                self.right_force_world[0]
            ),
            "right_fy_N": (
                self.right_force_world[1]
            ),
            "right_fz_N": (
                self.right_force_world[2]
            ),

            "left_tx_Nm": (
                self.left_torque_world[0]
            ),
            "left_ty_Nm": (
                self.left_torque_world[1]
            ),
            "left_tz_Nm": (
                self.left_torque_world[2]
            ),

            "right_tx_Nm": (
                self.right_torque_world[0]
            ),
            "right_ty_Nm": (
                self.right_torque_world[1]
            ),
            "right_tz_Nm": (
                self.right_torque_world[2]
            ),

            "left_force_cmd_x_N": (
                self.left_force_cmd[0]
            ),
            "left_force_cmd_y_N": (
                self.left_force_cmd[1]
            ),
            "left_force_cmd_z_N": (
                self.left_force_cmd[2]
            ),

            "right_force_cmd_x_N": (
                self.right_force_cmd[0]
            ),
            "right_force_cmd_y_N": (
                self.right_force_cmd[1]
            ),
            "right_force_cmd_z_N": (
                self.right_force_cmd[2]
            ),

            "right_omega_x": (
                self.right_omega_pos[0]
            ),
            "right_omega_y": (
                self.right_omega_pos[1]
            ),
            "right_omega_z": (
                self.right_omega_pos[2]
            ),

            "right_gripper_angle": (
                self.right_gripper_angle
            ),
            "right_gripper_button": int(
                self.right_gripper_button
            ),
        }

        for i in range(7):
            row[f"left_q{i+1}_rad"] = (
                self.left_q[i]
            )

        for i in range(7):
            row[f"right_q{i+1}_rad"] = (
                self.right_q[i]
            )

        for i in range(7):
            row[f"left_q_goal{i+1}_rad"] = (
                self.left_q_goal[i]
            )

        for i in range(7):
            row[f"right_q_goal{i+1}_rad"] = (
                self.right_q_goal[i]
            )

        self.writer.writerow(row)

        if self.lifting_mode:
            self.current_lifting_rows.append(
                row.copy()
            )

        self.row_count += 1

        # 1秒ごとにflush
        if (
            self.row_count
            % max(
                1,
                int(self.log_hz),
            )
            == 0
        ):
            self.csv_file.flush()

        self.prev_lifting_mode = (
            self.lifting_mode
        )

    # ========================================================
    # 終了
    # ========================================================

    def close(self):
        # Ctrl+Cがlifting中でも最後の区間を保存
        if (
            self.lifting_mode
            and self.current_lifting_rows
        ):
            try:
                self.finish_lifting_section()
            except Exception:
                pass

        try:
            self.csv_file.flush()
        except Exception:
            pass

        try:
            self.csv_file.close()
        except Exception:
            pass

        self.get_logger().info(
            f"CSV saved: {self.csv_path}"
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "SII Experiment 1 "
            "measurement-only logger"
        )
    )

    parser.add_argument(
        "--condition",
        required=True,
        choices=[
            "off",
            "on",
        ],
        help=(
            "実験条件のラベル。"
            "制御を変更する引数ではない。"
        ),
    )

    parser.add_argument(
        "--trial",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--object",
        dest="object_name",
        default="box1",
    )

    parser.add_argument(
        "--subject",
        default="operator",
    )

    parser.add_argument(
        "--note",
        default="",
    )

    parser.add_argument(
        "--log-hz",
        type=float,
        default=100.0,
    )

    # 論文評価用の目標。
    # 制御器には渡さない。
    parser.add_argument(
        "--target-margin",
        type=float,
        default=0.3,
        help=(
            "lifting開始時Fgからの"
            "評価目標増分[N]"
        ),
    )

    # 現在の45deg設置をdefaultにする。
    # 将来架台を変えてもCLIだけで変更できる。
    parser.add_argument(
        "--left-roll-deg",
        type=float,
        default=45.0,
    )

    parser.add_argument(
        "--right-roll-deg",
        type=float,
        default=-45.0,
    )

    parser.add_argument(
        "--right-base-x",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--right-base-y",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--right-base-z",
        type=float,
        default=0.0,
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    right_base_xyz = np.array([
        args.right_base_x,
        args.right_base_y,
        args.right_base_z,
    ], dtype=float)

    rclpy.init()

    node = Experiment1Logger(
        condition=args.condition,
        trial=args.trial,
        object_name=args.object_name,
        subject=args.subject,
        note=args.note,
        log_hz=args.log_hz,
        target_margin_n=(
            args.target_margin
        ),
        left_roll_deg=(
            args.left_roll_deg
        ),
        right_roll_deg=(
            args.right_roll_deg
        ),
        right_base_xyz=(
            right_base_xyz
        ),
    )

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.close()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
