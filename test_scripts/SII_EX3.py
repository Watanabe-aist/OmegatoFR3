#!/usr/bin/env python3
"""
SII_EX3.py

SII 実験3：冗長自由度を用いた中間リンク操作の「計測専用」ROS 2 logger

目的:
- 左Omega.7のグリッパ入力から左右FR3へ同一量の中間リンク変位を与えたとき，
  手先位置・姿勢を大きく変えずに中間リンク姿勢を変更できることを確認する。
- 制御は行わない。既存の teleoperation / TimeSeries IK と別ターミナルで起動する。
- 実FR3関節角 q と q_goal を同じURDFでFKし，
  手先 fr3_hand_tcp と中間リンク fr3_link5 の変位をCSVへ保存する。

想定topic:
  /franka/left/obs_franka_state
  /franka/right/obs_franka_state
  /omega_fr3/left/q_goal
  /omega_fr3/right/q_goal
  /left/state

実行例:
  python3 ~/franka_ros2_ws/test_scripts/SII_EX3.py \
    --trial 1 \
    --object box1 \
    --subject watanabe

重要:
- このloggerはFR3/Omegaへ制御指令をpublishしない。
- baselineは起動後 --baseline-delay 秒経過し，
  左右の実q/q_goalが揃った最初の時点で自動取得する。
- "BASELINE CAPTURED" が表示されてから中間リンク操作を開始する。
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

from sensor_msgs.msg import JointState
from ohrc_msgs.msg import State


# ============================================================
# Topic
# ============================================================

LEFT_STATE_TOPIC = "/franka/left/obs_franka_state"
RIGHT_STATE_TOPIC = "/franka/right/obs_franka_state"

LEFT_Q_GOAL_TOPIC = "/omega_fr3/left/q_goal"
RIGHT_Q_GOAL_TOPIC = "/omega_fr3/right/q_goal"

LEFT_OMEGA_STATE_TOPIC = "/left/state"


# ============================================================
# Current experiment geometry / controller settings
# ============================================================

URDF_PATH = (
    Path.home()
    / "franka_ros2_ws"
    / "src"
    / "fr3_urdf"
    / "fr3.urdf"
)

LOG_ROOT = (
    Path.home()
    / "franka_ros2_ws"
    / "experiment3_logs"
)

SUMMARY_PATH = LOG_ROOT / "experiment3_summary.csv"

DEFAULT_EE_FRAME = "fr3_hand_tcp"
DEFAULT_MID_FRAME = "fr3_link5"

# Current midpoint controller:
# MIDPOINT_AXIS = [0, 1, 0] in common world.
DEFAULT_MID_AXIS_WORLD = np.array(
    [0.0, 1.0, 0.0],
    dtype=float,
)

# Left Omega gripper -> midpoint command.
# Logger-side estimate only. The authoritative outputs are q_goal FK values.
DEFAULT_FINGER_TO_MID_SCALE = 0.5
DEFAULT_MID_GRIPPER_DEADBAND = 0.001
DEFAULT_MIDPOINT_TARGET_LIMIT = 0.5


def rot_x(theta_rad: float) -> np.ndarray:
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)

    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ], dtype=float)


def safe_joint_array(msg: JointState) -> np.ndarray:
    q = np.full(7, np.nan, dtype=float)

    n = min(7, len(msg.position))

    if n > 0:
        q[:n] = np.asarray(
            msg.position[:n],
            dtype=float,
        )

    return q


def finite_q7(q: np.ndarray) -> bool:
    q = np.asarray(q, dtype=float).reshape(-1)

    return (
        q.size >= 7
        and np.all(np.isfinite(q[:7]))
    )


def rotation_change_deg(
    R0: np.ndarray,
    R1: np.ndarray,
) -> float:
    if R0 is None or R1 is None:
        return np.nan

    try:
        rot_vec = pin.log3(
            np.asarray(R0, dtype=float).T
            @ np.asarray(R1, dtype=float)
        )
        return float(
            np.linalg.norm(rot_vec)
            * 180.0
            / math.pi
        )
    except Exception:
        return np.nan


class Experiment3Logger(Node):
    def __init__(
        self,
        *,
        trial: int,
        object_name: str,
        subject: str,
        note: str,
        log_hz: float,
        baseline_delay_s: float,
        ee_frame_name: str,
        mid_frame_name: str,
        left_roll_deg: float,
        right_roll_deg: float,
        right_base_xyz: np.ndarray,
        mid_axis_world: np.ndarray,
        finger_to_mid_scale: float,
        mid_gripper_deadband: float,
        midpoint_target_limit: float,
    ):
        super().__init__(
            "sii_ex3_midpoint_logger"
        )

        # ====================================================
        # Experiment metadata
        # ====================================================

        self.trial = int(trial)
        self.object_name = str(object_name)
        self.subject = str(subject)
        self.note = str(note)

        self.log_hz = max(
            1.0,
            float(log_hz),
        )

        self.baseline_delay_s = max(
            0.0,
            float(baseline_delay_s),
        )

        # ====================================================
        # Geometry
        # ====================================================

        self.left_base_world = np.zeros(
            3,
            dtype=float,
        )

        self.right_base_world = np.asarray(
            right_base_xyz,
            dtype=float,
        ).reshape(3)

        self.left_rot_world = rot_x(
            math.radians(
                float(left_roll_deg)
            )
        )

        self.right_rot_world = rot_x(
            math.radians(
                float(right_roll_deg)
            )
        )

        axis = np.asarray(
            mid_axis_world,
            dtype=float,
        ).reshape(3)

        axis_norm = float(
            np.linalg.norm(axis)
        )

        if axis_norm < 1.0e-12:
            raise ValueError(
                "mid_axis_world norm is zero"
            )

        self.mid_axis_world = (
            axis / axis_norm
        )

        # Controller convention:
        # left  = +MIDPOINT_AXIS
        # right = -MIDPOINT_AXIS
        self.left_mid_signed_axis_world = (
            self.mid_axis_world.copy()
        )

        self.right_mid_signed_axis_world = (
            -self.mid_axis_world.copy()
        )

        self.finger_to_mid_scale = float(
            finger_to_mid_scale
        )

        self.mid_gripper_deadband = float(
            mid_gripper_deadband
        )

        self.midpoint_target_limit = abs(
            float(midpoint_target_limit)
        )

        # ====================================================
        # Pinocchio model
        # ====================================================

        if not URDF_PATH.exists():
            raise FileNotFoundError(
                f"FR3 URDF not found: {URDF_PATH}"
            )

        self.model = pin.buildModelFromUrdf(
            str(URDF_PATH)
        )

        self.data = self.model.createData()

        self.ee_frame_name = str(
            ee_frame_name
        )

        if not self.model.existFrame(
            self.ee_frame_name
        ):
            if self.model.existFrame(
                "fr3_link8"
            ):
                self.ee_frame_name = (
                    "fr3_link8"
                )
            else:
                raise ValueError(
                    "EE frame not found: "
                    f"{ee_frame_name}"
                )

        self.mid_frame_name = str(
            mid_frame_name
        )

        if not self.model.existFrame(
            self.mid_frame_name
        ):
            raise ValueError(
                "Middle frame not found: "
                f"{self.mid_frame_name}"
            )

        self.ee_frame_id = (
            self.model.getFrameId(
                self.ee_frame_name
            )
        )

        self.mid_frame_id = (
            self.model.getFrameId(
                self.mid_frame_name
            )
        )

        # ====================================================
        # Latest ROS values
        # ====================================================

        self.left_q = np.full(
            7,
            np.nan,
            dtype=float,
        )

        self.right_q = np.full(
            7,
            np.nan,
            dtype=float,
        )

        self.left_q_goal = np.full(
            7,
            np.nan,
            dtype=float,
        )

        self.right_q_goal = np.full(
            7,
            np.nan,
            dtype=float,
        )

        self.left_omega_gripper_angle = (
            np.nan
        )

        self.left_omega_gripper_button = (
            False
        )

        # ====================================================
        # Baseline
        # ====================================================

        self.baseline_ready = False
        self.baseline_elapsed_s = np.nan

        self.left_gripper_reference = (
            np.nan
        )

        self.baseline = {}

        # ====================================================
        # Result buffer for summary
        # ====================================================

        self.best_abs_avg_mid_mm = -1.0
        self.best_row = None

        # ====================================================
        # Subscribers
        # ====================================================

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
            LEFT_OMEGA_STATE_TOPIC,
            self.left_omega_callback,
            10,
        )

        # ====================================================
        # CSV
        # ====================================================

        LOG_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        safe_object = (
            self.object_name
            .replace("/", "_")
            .replace(" ", "_")
        )

        safe_subject = (
            self.subject
            .replace("/", "_")
            .replace(" ", "_")
        )

        filename = (
            f"{stamp}_"
            f"ex3_midpoint_"
            f"trial{self.trial:02d}_"
            f"{safe_object}_"
            f"{safe_subject}.csv"
        )

        self.csv_path = (
            LOG_ROOT
            / filename
        )

        self.csv_file = (
            self.csv_path.open(
                "w",
                newline="",
                encoding="utf-8",
            )
        )

        self.fieldnames = (
            self.make_fieldnames()
        )

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
        # Startup log
        # ====================================================

        self.get_logger().info(
            "============================================="
        )
        self.get_logger().info(
            "SII EXPERIMENT 3 LOGGER ONLY"
        )
        self.get_logger().info(
            "NO control command is published."
        )
        self.get_logger().info(
            f"trial = {self.trial}"
        )
        self.get_logger().info(
            f"object = {self.object_name}"
        )
        self.get_logger().info(
            f"subject = {self.subject}"
        )
        self.get_logger().info(
            f"EE frame = {self.ee_frame_name}"
        )
        self.get_logger().info(
            f"mid frame = {self.mid_frame_name}"
        )
        self.get_logger().info(
            "mid axis world = "
            f"{self.mid_axis_world.tolist()}"
        )
        self.get_logger().info(
            "left signed axis  = "
            f"{self.left_mid_signed_axis_world.tolist()}"
        )
        self.get_logger().info(
            "right signed axis = "
            f"{self.right_mid_signed_axis_world.tolist()}"
        )
        self.get_logger().info(
            f"baseline delay = "
            f"{self.baseline_delay_s:.2f} s"
        )
        self.get_logger().info(
            f"log = {self.csv_path}"
        )
        self.get_logger().warn(
            "Wait for 'BASELINE CAPTURED' "
            "before moving the left Omega gripper."
        )
        self.get_logger().info(
            "============================================="
        )

    # ========================================================
    # CSV columns
    # ========================================================

    def make_fieldnames(self):
        fields = [
            "wall_time",
            "elapsed_s",

            "trial",
            "object",
            "subject",
            "note",

            "baseline_ready",
            "baseline_elapsed_s",

            "left_omega_gripper_angle_rad",
            "left_omega_gripper_delta_rad",
            "estimated_mid_command_m",
            "estimated_mid_command_mm",

            # Actual q: hand change from baseline
            "left_actual_ee_pos_change_mm",
            "right_actual_ee_pos_change_mm",
            "left_actual_ee_rot_change_deg",
            "right_actual_ee_rot_change_deg",

            # Actual q: midpoint motion along each inward/signed axis
            "left_actual_mid_axis_disp_mm",
            "right_actual_mid_axis_disp_mm",
            "actual_mid_avg_disp_mm",
            "actual_mid_lr_difference_mm",
            "left_actual_mid_norm_disp_mm",
            "right_actual_mid_norm_disp_mm",

            # q_goal FK
            "left_goal_ee_pos_change_mm",
            "right_goal_ee_pos_change_mm",
            "left_goal_ee_rot_change_deg",
            "right_goal_ee_rot_change_deg",

            "left_goal_mid_axis_disp_mm",
            "right_goal_mid_axis_disp_mm",
            "goal_mid_avg_disp_mm",
            "goal_mid_lr_difference_mm",
            "left_goal_mid_norm_disp_mm",
            "right_goal_mid_norm_disp_mm",

            # Goal - actual tracking / realization diagnostics
            "left_mid_axis_goal_minus_actual_mm",
            "right_mid_axis_goal_minus_actual_mm",
            "left_ee_goal_actual_pos_error_mm",
            "right_ee_goal_actual_pos_error_mm",
            "left_mid_goal_actual_pos_error_mm",
            "right_mid_goal_actual_pos_error_mm",

            # Current world positions for plotting / reconstruction
            "left_actual_ee_world_x_m",
            "left_actual_ee_world_y_m",
            "left_actual_ee_world_z_m",

            "right_actual_ee_world_x_m",
            "right_actual_ee_world_y_m",
            "right_actual_ee_world_z_m",

            "left_actual_mid_world_x_m",
            "left_actual_mid_world_y_m",
            "left_actual_mid_world_z_m",

            "right_actual_mid_world_x_m",
            "right_actual_mid_world_y_m",
            "right_actual_mid_world_z_m",

            "left_goal_ee_world_x_m",
            "left_goal_ee_world_y_m",
            "left_goal_ee_world_z_m",

            "right_goal_ee_world_x_m",
            "right_goal_ee_world_y_m",
            "right_goal_ee_world_z_m",

            "left_goal_mid_world_x_m",
            "left_goal_mid_world_y_m",
            "left_goal_mid_world_z_m",

            "right_goal_mid_world_x_m",
            "right_goal_mid_world_y_m",
            "right_goal_mid_world_z_m",
        ]

        for side in ["left", "right"]:
            for i in range(7):
                fields.append(
                    f"{side}_q{i+1}_rad"
                )

        for side in ["left", "right"]:
            for i in range(7):
                fields.append(
                    f"{side}_q_goal{i+1}_rad"
                )

        return fields

    # ========================================================
    # ROS callbacks
    # ========================================================

    def left_state_callback(
        self,
        msg: JointState,
    ):
        self.left_q = (
            safe_joint_array(msg)
        )

    def right_state_callback(
        self,
        msg: JointState,
    ):
        self.right_q = (
            safe_joint_array(msg)
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

    def left_omega_callback(
        self,
        msg: State,
    ):
        try:
            self.left_omega_gripper_angle = (
                float(msg.gripper.angle)
            )
        except Exception:
            self.left_omega_gripper_angle = (
                np.nan
            )

        try:
            self.left_omega_gripper_button = (
                bool(msg.gripper.button)
            )
        except Exception:
            self.left_omega_gripper_button = (
                False
            )

    # ========================================================
    # FK
    # ========================================================

    def fk_local(
        self,
        q7: np.ndarray,
    ):
        if not finite_q7(q7):
            return None

        q_full = pin.neutral(
            self.model
        )

        q_full[:7] = np.asarray(
            q7[:7],
            dtype=float,
        )

        if self.model.nq >= 9:
            q_full[7] = 0.02
            q_full[8] = 0.02

        pin.forwardKinematics(
            self.model,
            self.data,
            q_full,
        )

        pin.updateFramePlacements(
            self.model,
            self.data,
        )

        ee = self.data.oMf[
            self.ee_frame_id
        ]

        mid = self.data.oMf[
            self.mid_frame_id
        ]

        return {
            "ee_pos": (
                ee.translation.copy()
            ),
            "ee_rot": (
                ee.rotation.copy()
            ),
            "mid_pos": (
                mid.translation.copy()
            ),
            "mid_rot": (
                mid.rotation.copy()
            ),
        }

    def local_pose_to_world(
        self,
        side: str,
        pose: dict | None,
    ):
        if pose is None:
            return None

        if side == "left":
            base_pos = (
                self.left_base_world
            )
            R_world_base = (
                self.left_rot_world
            )
        elif side == "right":
            base_pos = (
                self.right_base_world
            )
            R_world_base = (
                self.right_rot_world
            )
        else:
            raise ValueError(side)

        return {
            "ee_pos": (
                base_pos
                + R_world_base
                @ pose["ee_pos"]
            ),
            "ee_rot": (
                R_world_base
                @ pose["ee_rot"]
            ),
            "mid_pos": (
                base_pos
                + R_world_base
                @ pose["mid_pos"]
            ),
            "mid_rot": (
                R_world_base
                @ pose["mid_rot"]
            ),
        }

    def fk_world(
        self,
        side: str,
        q7: np.ndarray,
    ):
        return self.local_pose_to_world(
            side,
            self.fk_local(q7),
        )

    # ========================================================
    # Baseline
    # ========================================================

    def all_joint_data_valid(self):
        return (
            finite_q7(self.left_q)
            and finite_q7(self.right_q)
            and finite_q7(
                self.left_q_goal
            )
            and finite_q7(
                self.right_q_goal
            )
        )

    def capture_baseline(
        self,
        elapsed_s: float,
        current: dict,
    ):
        self.baseline = {
            key: {
                subkey: np.asarray(
                    value,
                    dtype=float,
                ).copy()
                for subkey, value
                in pose.items()
            }
            for key, pose
            in current.items()
        }

        self.baseline_ready = True

        self.baseline_elapsed_s = float(
            elapsed_s
        )

        if math.isfinite(
            self.left_omega_gripper_angle
        ):
            self.left_gripper_reference = (
                self.left_omega_gripper_angle
            )

        self.get_logger().warn(
            "============================================="
        )
        self.get_logger().warn(
            "BASELINE CAPTURED"
        )
        self.get_logger().warn(
            f"t = {elapsed_s:.3f} s"
        )
        self.get_logger().warn(
            "Now move ONLY the left Omega gripper "
            "to command the midpoint."
        )
        self.get_logger().warn(
            "============================================="
        )

    # ========================================================
    # Metrics
    # ========================================================

    @staticmethod
    def pos_change_mm(
        p0,
        p1,
    ):
        if p0 is None or p1 is None:
            return np.nan

        return float(
            1000.0
            * np.linalg.norm(
                np.asarray(p1)
                - np.asarray(p0)
            )
        )

    @staticmethod
    def axis_displacement_mm(
        p0,
        p1,
        axis,
    ):
        if (
            p0 is None
            or p1 is None
            or axis is None
        ):
            return np.nan

        return float(
            1000.0
            * (
                np.asarray(axis)
                @ (
                    np.asarray(p1)
                    - np.asarray(p0)
                )
            )
        )

    @staticmethod
    def xyz_or_nan(
        pose,
        key,
    ):
        if pose is None:
            return (
                np.nan,
                np.nan,
                np.nan,
            )

        arr = np.asarray(
            pose[key],
            dtype=float,
        ).reshape(3)

        return (
            float(arr[0]),
            float(arr[1]),
            float(arr[2]),
        )

    def estimate_mid_command(self):
        if (
            not math.isfinite(
                self.left_omega_gripper_angle
            )
            or not math.isfinite(
                self.left_gripper_reference
            )
        ):
            return (
                np.nan,
                np.nan,
            )

        delta = float(
            self.left_omega_gripper_angle
            - self.left_gripper_reference
        )

        if (
            abs(delta)
            < self.mid_gripper_deadband
        ):
            command = 0.0
        else:
            command = (
                delta
                * self.finger_to_mid_scale
            )

        command = float(
            np.clip(
                command,
                -self.midpoint_target_limit,
                self.midpoint_target_limit,
            )
        )

        return delta, command

    # ========================================================
    # Main logger
    # ========================================================

    def log_row(self):
        elapsed_s = float(
            time.monotonic()
            - self.start_monotonic
        )

        # Require all four joint vectors so actual/q_goal can be
        # evaluated with exactly the same FK model.
        if not self.all_joint_data_valid():
            if (
                self.row_count % int(
                    max(1.0, self.log_hz)
                )
                == 0
            ):
                self.get_logger().warn(
                    "Waiting for left/right actual q "
                    "and q_goal..."
                )
            self.row_count += 1
            return

        current = {
            "left_actual": self.fk_world(
                "left",
                self.left_q,
            ),
            "right_actual": self.fk_world(
                "right",
                self.right_q,
            ),
            "left_goal": self.fk_world(
                "left",
                self.left_q_goal,
            ),
            "right_goal": self.fk_world(
                "right",
                self.right_q_goal,
            ),
        }

        if (
            not self.baseline_ready
            and elapsed_s
            >= self.baseline_delay_s
        ):
            self.capture_baseline(
                elapsed_s,
                current,
            )

        # ----------------------------------------------------
        # Default values before baseline
        # ----------------------------------------------------

        metric = {
            "left_actual_ee_pos_change_mm": np.nan,
            "right_actual_ee_pos_change_mm": np.nan,
            "left_actual_ee_rot_change_deg": np.nan,
            "right_actual_ee_rot_change_deg": np.nan,

            "left_actual_mid_axis_disp_mm": np.nan,
            "right_actual_mid_axis_disp_mm": np.nan,
            "actual_mid_avg_disp_mm": np.nan,
            "actual_mid_lr_difference_mm": np.nan,
            "left_actual_mid_norm_disp_mm": np.nan,
            "right_actual_mid_norm_disp_mm": np.nan,

            "left_goal_ee_pos_change_mm": np.nan,
            "right_goal_ee_pos_change_mm": np.nan,
            "left_goal_ee_rot_change_deg": np.nan,
            "right_goal_ee_rot_change_deg": np.nan,

            "left_goal_mid_axis_disp_mm": np.nan,
            "right_goal_mid_axis_disp_mm": np.nan,
            "goal_mid_avg_disp_mm": np.nan,
            "goal_mid_lr_difference_mm": np.nan,
            "left_goal_mid_norm_disp_mm": np.nan,
            "right_goal_mid_norm_disp_mm": np.nan,

            "left_mid_axis_goal_minus_actual_mm": np.nan,
            "right_mid_axis_goal_minus_actual_mm": np.nan,
            "left_ee_goal_actual_pos_error_mm": np.nan,
            "right_ee_goal_actual_pos_error_mm": np.nan,
            "left_mid_goal_actual_pos_error_mm": np.nan,
            "right_mid_goal_actual_pos_error_mm": np.nan,
        }

        if self.baseline_ready:
            b = self.baseline

            # Actual EE
            metric[
                "left_actual_ee_pos_change_mm"
            ] = self.pos_change_mm(
                b["left_actual"]["ee_pos"],
                current["left_actual"]["ee_pos"],
            )

            metric[
                "right_actual_ee_pos_change_mm"
            ] = self.pos_change_mm(
                b["right_actual"]["ee_pos"],
                current["right_actual"]["ee_pos"],
            )

            metric[
                "left_actual_ee_rot_change_deg"
            ] = rotation_change_deg(
                b["left_actual"]["ee_rot"],
                current["left_actual"]["ee_rot"],
            )

            metric[
                "right_actual_ee_rot_change_deg"
            ] = rotation_change_deg(
                b["right_actual"]["ee_rot"],
                current["right_actual"]["ee_rot"],
            )

            # Actual midpoint
            left_actual_mid_axis = (
                self.axis_displacement_mm(
                    b["left_actual"]["mid_pos"],
                    current["left_actual"]["mid_pos"],
                    self.left_mid_signed_axis_world,
                )
            )

            right_actual_mid_axis = (
                self.axis_displacement_mm(
                    b["right_actual"]["mid_pos"],
                    current["right_actual"]["mid_pos"],
                    self.right_mid_signed_axis_world,
                )
            )

            metric[
                "left_actual_mid_axis_disp_mm"
            ] = left_actual_mid_axis

            metric[
                "right_actual_mid_axis_disp_mm"
            ] = right_actual_mid_axis

            metric[
                "actual_mid_avg_disp_mm"
            ] = 0.5 * (
                left_actual_mid_axis
                + right_actual_mid_axis
            )

            metric[
                "actual_mid_lr_difference_mm"
            ] = (
                left_actual_mid_axis
                - right_actual_mid_axis
            )

            metric[
                "left_actual_mid_norm_disp_mm"
            ] = self.pos_change_mm(
                b["left_actual"]["mid_pos"],
                current["left_actual"]["mid_pos"],
            )

            metric[
                "right_actual_mid_norm_disp_mm"
            ] = self.pos_change_mm(
                b["right_actual"]["mid_pos"],
                current["right_actual"]["mid_pos"],
            )

            # q_goal EE
            metric[
                "left_goal_ee_pos_change_mm"
            ] = self.pos_change_mm(
                b["left_goal"]["ee_pos"],
                current["left_goal"]["ee_pos"],
            )

            metric[
                "right_goal_ee_pos_change_mm"
            ] = self.pos_change_mm(
                b["right_goal"]["ee_pos"],
                current["right_goal"]["ee_pos"],
            )

            metric[
                "left_goal_ee_rot_change_deg"
            ] = rotation_change_deg(
                b["left_goal"]["ee_rot"],
                current["left_goal"]["ee_rot"],
            )

            metric[
                "right_goal_ee_rot_change_deg"
            ] = rotation_change_deg(
                b["right_goal"]["ee_rot"],
                current["right_goal"]["ee_rot"],
            )

            # q_goal midpoint
            left_goal_mid_axis = (
                self.axis_displacement_mm(
                    b["left_goal"]["mid_pos"],
                    current["left_goal"]["mid_pos"],
                    self.left_mid_signed_axis_world,
                )
            )

            right_goal_mid_axis = (
                self.axis_displacement_mm(
                    b["right_goal"]["mid_pos"],
                    current["right_goal"]["mid_pos"],
                    self.right_mid_signed_axis_world,
                )
            )

            metric[
                "left_goal_mid_axis_disp_mm"
            ] = left_goal_mid_axis

            metric[
                "right_goal_mid_axis_disp_mm"
            ] = right_goal_mid_axis

            metric[
                "goal_mid_avg_disp_mm"
            ] = 0.5 * (
                left_goal_mid_axis
                + right_goal_mid_axis
            )

            metric[
                "goal_mid_lr_difference_mm"
            ] = (
                left_goal_mid_axis
                - right_goal_mid_axis
            )

            metric[
                "left_goal_mid_norm_disp_mm"
            ] = self.pos_change_mm(
                b["left_goal"]["mid_pos"],
                current["left_goal"]["mid_pos"],
            )

            metric[
                "right_goal_mid_norm_disp_mm"
            ] = self.pos_change_mm(
                b["right_goal"]["mid_pos"],
                current["right_goal"]["mid_pos"],
            )

            # Relative goal/actual diagnostics
            metric[
                "left_mid_axis_goal_minus_actual_mm"
            ] = (
                left_goal_mid_axis
                - left_actual_mid_axis
            )

            metric[
                "right_mid_axis_goal_minus_actual_mm"
            ] = (
                right_goal_mid_axis
                - right_actual_mid_axis
            )

            metric[
                "left_ee_goal_actual_pos_error_mm"
            ] = self.pos_change_mm(
                current["left_actual"]["ee_pos"],
                current["left_goal"]["ee_pos"],
            )

            metric[
                "right_ee_goal_actual_pos_error_mm"
            ] = self.pos_change_mm(
                current["right_actual"]["ee_pos"],
                current["right_goal"]["ee_pos"],
            )

            metric[
                "left_mid_goal_actual_pos_error_mm"
            ] = self.pos_change_mm(
                current["left_actual"]["mid_pos"],
                current["left_goal"]["mid_pos"],
            )

            metric[
                "right_mid_goal_actual_pos_error_mm"
            ] = self.pos_change_mm(
                current["right_actual"]["mid_pos"],
                current["right_goal"]["mid_pos"],
            )

        gripper_delta, estimated_command_m = (
            self.estimate_mid_command()
        )

        # Current xyz
        lae = self.xyz_or_nan(
            current["left_actual"],
            "ee_pos",
        )
        rae = self.xyz_or_nan(
            current["right_actual"],
            "ee_pos",
        )
        lam = self.xyz_or_nan(
            current["left_actual"],
            "mid_pos",
        )
        ram = self.xyz_or_nan(
            current["right_actual"],
            "mid_pos",
        )

        lge = self.xyz_or_nan(
            current["left_goal"],
            "ee_pos",
        )
        rge = self.xyz_or_nan(
            current["right_goal"],
            "ee_pos",
        )
        lgm = self.xyz_or_nan(
            current["left_goal"],
            "mid_pos",
        )
        rgm = self.xyz_or_nan(
            current["right_goal"],
            "mid_pos",
        )

        row = {
            "wall_time": (
                datetime.now().isoformat(
                    timespec="milliseconds"
                )
            ),
            "elapsed_s": elapsed_s,

            "trial": self.trial,
            "object": self.object_name,
            "subject": self.subject,
            "note": self.note,

            "baseline_ready": int(
                self.baseline_ready
            ),
            "baseline_elapsed_s": (
                self.baseline_elapsed_s
            ),

            "left_omega_gripper_angle_rad": (
                self.left_omega_gripper_angle
            ),
            "left_omega_gripper_delta_rad": (
                gripper_delta
            ),
            "estimated_mid_command_m": (
                estimated_command_m
            ),
            "estimated_mid_command_mm": (
                estimated_command_m * 1000.0
                if math.isfinite(
                    estimated_command_m
                )
                else np.nan
            ),

            **metric,

            "left_actual_ee_world_x_m": lae[0],
            "left_actual_ee_world_y_m": lae[1],
            "left_actual_ee_world_z_m": lae[2],

            "right_actual_ee_world_x_m": rae[0],
            "right_actual_ee_world_y_m": rae[1],
            "right_actual_ee_world_z_m": rae[2],

            "left_actual_mid_world_x_m": lam[0],
            "left_actual_mid_world_y_m": lam[1],
            "left_actual_mid_world_z_m": lam[2],

            "right_actual_mid_world_x_m": ram[0],
            "right_actual_mid_world_y_m": ram[1],
            "right_actual_mid_world_z_m": ram[2],

            "left_goal_ee_world_x_m": lge[0],
            "left_goal_ee_world_y_m": lge[1],
            "left_goal_ee_world_z_m": lge[2],

            "right_goal_ee_world_x_m": rge[0],
            "right_goal_ee_world_y_m": rge[1],
            "right_goal_ee_world_z_m": rge[2],

            "left_goal_mid_world_x_m": lgm[0],
            "left_goal_mid_world_y_m": lgm[1],
            "left_goal_mid_world_z_m": lgm[2],

            "right_goal_mid_world_x_m": rgm[0],
            "right_goal_mid_world_y_m": rgm[1],
            "right_goal_mid_world_z_m": rgm[2],
        }

        for i in range(7):
            row[
                f"left_q{i+1}_rad"
            ] = float(
                self.left_q[i]
            )

            row[
                f"right_q{i+1}_rad"
            ] = float(
                self.right_q[i]
            )

            row[
                f"left_q_goal{i+1}_rad"
            ] = float(
                self.left_q_goal[i]
            )

            row[
                f"right_q_goal{i+1}_rad"
            ] = float(
                self.right_q_goal[i]
            )

        self.writer.writerow(row)

        self.row_count += 1

        if self.row_count % 100 == 0:
            self.csv_file.flush()

        # Keep the sample with the largest absolute average
        # actual midpoint displacement for paper-ready summary.
        if self.baseline_ready:
            avg_mid = row[
                "actual_mid_avg_disp_mm"
            ]

            if (
                math.isfinite(avg_mid)
                and abs(avg_mid)
                > self.best_abs_avg_mid_mm
            ):
                self.best_abs_avg_mid_mm = (
                    abs(avg_mid)
                )

                self.best_row = (
                    row.copy()
                )

        # 1-Hz diagnostic print
        if (
            self.baseline_ready
            and self.row_count
            % int(
                max(
                    1.0,
                    round(self.log_hz),
                )
            )
            == 0
        ):
            self.get_logger().info(
                "EX3: "
                f"mid_actual "
                f"L={metric['left_actual_mid_axis_disp_mm']:.1f} mm, "
                f"R={metric['right_actual_mid_axis_disp_mm']:.1f} mm, "
                f"EE_change "
                f"L={metric['left_actual_ee_pos_change_mm']:.1f} mm, "
                f"R={metric['right_actual_ee_pos_change_mm']:.1f} mm"
            )

    # ========================================================
    # Summary
    # ========================================================

    def append_summary(self):
        if self.best_row is None:
            self.get_logger().warn(
                "No valid baseline/result sample "
                "for experiment3 summary."
            )
            return

        LOG_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_fields = [
            "timestamp",
            "trial",
            "object",
            "subject",
            "note",

            "selected_elapsed_s",

            "left_mid_axis_disp_mm",
            "right_mid_axis_disp_mm",
            "mid_avg_disp_mm",
            "mid_lr_difference_mm",

            "left_ee_pos_change_mm",
            "right_ee_pos_change_mm",
            "left_ee_rot_change_deg",
            "right_ee_rot_change_deg",

            "left_goal_mid_axis_disp_mm",
            "right_goal_mid_axis_disp_mm",

            "left_mid_axis_goal_minus_actual_mm",
            "right_mid_axis_goal_minus_actual_mm",

            "estimated_mid_command_mm",
            "raw_csv",
        ]

        r = self.best_row

        summary = {
            "timestamp": (
                datetime.now().isoformat(
                    timespec="seconds"
                )
            ),
            "trial": self.trial,
            "object": self.object_name,
            "subject": self.subject,
            "note": self.note,

            "selected_elapsed_s": (
                r["elapsed_s"]
            ),

            "left_mid_axis_disp_mm": (
                r[
                    "left_actual_mid_axis_disp_mm"
                ]
            ),
            "right_mid_axis_disp_mm": (
                r[
                    "right_actual_mid_axis_disp_mm"
                ]
            ),
            "mid_avg_disp_mm": (
                r[
                    "actual_mid_avg_disp_mm"
                ]
            ),
            "mid_lr_difference_mm": (
                r[
                    "actual_mid_lr_difference_mm"
                ]
            ),

            "left_ee_pos_change_mm": (
                r[
                    "left_actual_ee_pos_change_mm"
                ]
            ),
            "right_ee_pos_change_mm": (
                r[
                    "right_actual_ee_pos_change_mm"
                ]
            ),
            "left_ee_rot_change_deg": (
                r[
                    "left_actual_ee_rot_change_deg"
                ]
            ),
            "right_ee_rot_change_deg": (
                r[
                    "right_actual_ee_rot_change_deg"
                ]
            ),

            "left_goal_mid_axis_disp_mm": (
                r[
                    "left_goal_mid_axis_disp_mm"
                ]
            ),
            "right_goal_mid_axis_disp_mm": (
                r[
                    "right_goal_mid_axis_disp_mm"
                ]
            ),

            "left_mid_axis_goal_minus_actual_mm": (
                r[
                    "left_mid_axis_goal_minus_actual_mm"
                ]
            ),
            "right_mid_axis_goal_minus_actual_mm": (
                r[
                    "right_mid_axis_goal_minus_actual_mm"
                ]
            ),

            "estimated_mid_command_mm": (
                r[
                    "estimated_mid_command_mm"
                ]
            ),

            "raw_csv": str(
                self.csv_path
            ),
        }

        exists = SUMMARY_PATH.exists()

        with SUMMARY_PATH.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=summary_fields,
            )

            if not exists:
                writer.writeheader()

            writer.writerow(summary)

        self.get_logger().warn(
            "============================================="
        )
        self.get_logger().warn(
            "EXPERIMENT 3 SUMMARY"
        )
        self.get_logger().warn(
            "Selected maximum |average midpoint displacement| sample:"
        )
        self.get_logger().warn(
            f"mid actual: "
            f"L={summary['left_mid_axis_disp_mm']:.2f} mm, "
            f"R={summary['right_mid_axis_disp_mm']:.2f} mm, "
            f"avg={summary['mid_avg_disp_mm']:.2f} mm"
        )
        self.get_logger().warn(
            f"EE position change: "
            f"L={summary['left_ee_pos_change_mm']:.2f} mm, "
            f"R={summary['right_ee_pos_change_mm']:.2f} mm"
        )
        self.get_logger().warn(
            f"EE rotation change: "
            f"L={summary['left_ee_rot_change_deg']:.2f} deg, "
            f"R={summary['right_ee_rot_change_deg']:.2f} deg"
        )
        self.get_logger().warn(
            f"summary = {SUMMARY_PATH}"
        )
        self.get_logger().warn(
            "============================================="
        )

    def close(self):
        try:
            self.append_summary()
        except Exception as exc:
            self.get_logger().error(
                f"summary failed: {exc}"
            )

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
            "SII Experiment 3 "
            "midpoint measurement-only logger"
        )
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

    parser.add_argument(
        "--baseline-delay",
        type=float,
        default=2.0,
        help=(
            "logger起動後，baselineを自動取得するまでの待ち時間[s]"
        ),
    )

    parser.add_argument(
        "--ee-frame",
        default=DEFAULT_EE_FRAME,
    )

    parser.add_argument(
        "--mid-frame",
        default=DEFAULT_MID_FRAME,
    )

    # Current installed dual-FR3 geometry
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

    parser.add_argument(
        "--mid-axis-x",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--mid-axis-y",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--mid-axis-z",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--finger-to-mid-scale",
        type=float,
        default=DEFAULT_FINGER_TO_MID_SCALE,
    )

    parser.add_argument(
        "--mid-gripper-deadband",
        type=float,
        default=DEFAULT_MID_GRIPPER_DEADBAND,
    )

    parser.add_argument(
        "--midpoint-target-limit",
        type=float,
        default=DEFAULT_MIDPOINT_TARGET_LIMIT,
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

    mid_axis_world = np.array([
        args.mid_axis_x,
        args.mid_axis_y,
        args.mid_axis_z,
    ], dtype=float)

    rclpy.init()

    node = Experiment3Logger(
        trial=args.trial,
        object_name=args.object_name,
        subject=args.subject,
        note=args.note,
        log_hz=args.log_hz,
        baseline_delay_s=(
            args.baseline_delay
        ),
        ee_frame_name=args.ee_frame,
        mid_frame_name=args.mid_frame,
        left_roll_deg=args.left_roll_deg,
        right_roll_deg=args.right_roll_deg,
        right_base_xyz=right_base_xyz,
        mid_axis_world=mid_axis_world,
        finger_to_mid_scale=(
            args.finger_to_mid_scale
        ),
        mid_gripper_deadband=(
            args.mid_gripper_deadband
        ),
        midpoint_target_limit=(
            args.midpoint_target_limit
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
