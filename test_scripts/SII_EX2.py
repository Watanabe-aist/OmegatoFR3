#!/usr/bin/env python3
"""
SII_EX2.py

SII Experiment 2:
  - C1/C2/C3/C4 condition selection
  - Actual-Q normal DLS + lifting-entry fix + PI grasp assist
  - Midpoint control OFF in all conditions
  - Grasp-force maintenance assist ON in all conditions
  - Raw CSV logging for later task-time / trajectory / force analysis

Condition definition
--------------------
C1 : Lifting OFF / Force Feedback OFF / Finger operation OFF
C2 : Lifting ON  / Force Feedback OFF / Finger operation OFF
C3 : Lifting OFF / Force Feedback ON  / Finger operation ON
C4 : Lifting ON  / Force Feedback ON  / Finger operation ON

Important about "Force Feedback"
--------------------------------
This script directly switches the bilateral Cartesian force-command path:
  OFF -> force_mode="zero"
  ON  -> force_mode="wrench"

Therefore Z/translational feedback is switched by C1-C4 here.

In the current system the right Omega.7 gripper also has a separate inward/grasp
force-feedback path.  That path must use the condition gate published here:
  /sii_ex2/force_feedback_enabled  (std_msgs/Bool)
if it is implemented in another node/process.

DO NOT suppress /dual_grip/grip_force itself for OFF conditions:
the grasp-force maintenance assist uses that same measurement and must stay ON.

Usage
-----
cd ~/franka_ros2_ws/test_scripts

python3 SII_EX2.py \
  --condition C1 \
  --subject S01 \
  --trial 1 \
  --object-set 3stack

python3 SII_EX2.py --condition C2 --subject S01 --trial 1
python3 SII_EX2.py --condition C3 --subject S01 --trial 1
python3 SII_EX2.py --condition C4 --subject S01 --trial 1

Default log directory:
  ~/franka_ros2_ws/experiment2_logs/

Each run creates:
  1) raw CSV
  2) one appended row in experiment2_summary.csv on clean shutdown/Ctrl+C
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy

from geometry_msgs.msg import PoseStamped, WrenchStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64
from ohrc_msgs.msg import State


# ============================================================
# Current controller stack
# ============================================================
#
# Importing this module:
#   - uses normal DLS from latest measured actual q
#   - keeps the lifting-entry Cartesian target continuous
#   - installs PI grasp assist
#   - imports omega_to_fr3_q_goal_dual_bilateral as .bilateral
#
import omega_to_fr3_q_goal_dual as base
import omega_to_fr3_q_goal_dual_actualq_dls_liftfix_pi as actualq_pi

bilateral = actualq_pi.bilateral


# ============================================================
# Experiment 2 conditions
# ============================================================

CONDITIONS = {
    "C1": {
        "lifting": False,
        "feedback": False,
        "finger_control": False,
        "label": "Lifting_OFF_Feedback_OFF_FingerControl_OFF",
    },
    "C2": {
        "lifting": True,
        "feedback": False,
        "finger_control": True,
        "label": "Lifting_ON_Feedback_OFF_FingerControl_ON",
    },
    "C3": {
        "lifting": False,
        "feedback": True,
        "finger_control": False,
        "label": "Lifting_OFF_Feedback_ON_FingerControl_OFF",
    },
    "C4": {
        "lifting": True,
        "feedback": True,
        "finger_control": True,
        "label": "Lifting_ON_Feedback_ON_FingerControl_ON",
    },
}


# ============================================================
# Topics
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

LEFT_OMEGA_STATE_TOPIC = "/left/state"
RIGHT_OMEGA_STATE_TOPIC = "/right/state"

# Separate haptic/gripper-feedback node can subscribe to this gate.
FEEDBACK_ENABLE_TOPIC = "/sii_ex2/force_feedback_enabled"


LOG_ROOT_DEFAULT = (
    Path.home()
    / "franka_ros2_ws"
    / "experiment2_logs"
)


# ============================================================
# Helpers
# ============================================================

def safe_name(text: str) -> str:
    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def nan_array(n: int) -> np.ndarray:
    return np.full(n, np.nan, dtype=float)


def pose_position(msg: PoseStamped) -> np.ndarray:
    return np.array(
        [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ],
        dtype=float,
    )


def pose_quaternion(msg: PoseStamped) -> np.ndarray:
    return np.array(
        [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ],
        dtype=float,
    )


def wrench_force(msg: WrenchStamped) -> np.ndarray:
    return np.array(
        [
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
        ],
        dtype=float,
    )


def wrench_torque(msg: WrenchStamped) -> np.ndarray:
    return np.array(
        [
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z,
        ],
        dtype=float,
    )


def joint7(msg: JointState) -> np.ndarray:
    out = nan_array(7)
    n = min(7, len(msg.position))
    if n:
        out[:n] = np.asarray(msg.position[:n], dtype=float)
    return out


def omega_position(msg: State) -> np.ndarray:
    return np.array(
        [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ],
        dtype=float,
    )


def omega_quaternion(msg: State) -> np.ndarray:
    return np.array(
        [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ],
        dtype=float,
    )


def finite_xyz(v: np.ndarray) -> bool:
    return (
        v is not None
        and np.asarray(v).size >= 3
        and np.all(np.isfinite(np.asarray(v)[:3]))
    )


def append_xyz(row: dict, prefix: str, v: np.ndarray):
    row[f"{prefix}_x"] = float(v[0])
    row[f"{prefix}_y"] = float(v[1])
    row[f"{prefix}_z"] = float(v[2])


def append_quat(row: dict, prefix: str, q: np.ndarray):
    row[f"{prefix}_qx"] = float(q[0])
    row[f"{prefix}_qy"] = float(q[1])
    row[f"{prefix}_qz"] = float(q[2])
    row[f"{prefix}_qw"] = float(q[3])


def append_joint7(row: dict, prefix: str, q: np.ndarray):
    for i in range(7):
        row[f"{prefix}{i + 1}"] = float(q[i])


# ============================================================
# Experiment node
# ============================================================

class Experiment2Node(
    bilateral.IntegratedBilateralOmegaToFR3FactrStyle
):
    def __init__(
        self,
        *,
        condition: str,
        subject: str,
        trial: int,
        object_set: str,
        note: str,
        log_hz: float,
        output_root: Path,
        force_gain_z: float,
        force_limit: float,
        force_deadband: float,
        lpf_alpha: float,
        bias_samples: int,
        force_damping_z: float,
        force_sign_z: float,
    ):
        self.exp_condition = condition
        self.exp_spec = CONDITIONS[condition]
        self.exp_subject = str(subject)
        self.exp_trial = int(trial)
        self.exp_object_set = str(object_set)
        self.exp_note = str(note)
        self.exp_log_hz = max(1.0, float(log_hz))
        self.exp_output_root = Path(output_root).expanduser()

        # ----------------------------------------------------
        # Condition control
        # ----------------------------------------------------
        # Common setting: grasp assist ON
        base.ENABLE_GRASP_ASSIST = True

        # C1/C2: finger operation OFF, C3/C4: ON
        # Omega gripper -> dual-arm grasp-position operation itself.
        base.ENABLE_GRIP_POSITION_CONTROL = bool(
            self.exp_spec["finger_control"]
        )

        # Common setting: midpoint OFF
        base.ENABLE_RIGHT_MIDPOINT_CONTROL = False
        base.ENABLE_LEFT_MIDPOINT_CONTROL = False

        # Experimental factor 1: Lifting support
        base.ENABLE_LIFTING_MODE = bool(
            self.exp_spec["lifting"]
        )

        # Experimental factor 2:
        # known bilateral Cartesian/Z force-feedback path
        selected_force_mode = (
            "wrench"
            if self.exp_spec["feedback"]
            else "zero"
        )

        # ----------------------------------------------------
        # Initialize latest values BEFORE super().__init__()
        # ----------------------------------------------------
        self.exp_grip_force_n = np.nan
        self.exp_grasp_assist_offset_m = np.nan

        self.exp_left_ee_pos = nan_array(3)
        self.exp_right_ee_pos = nan_array(3)
        self.exp_left_ee_quat = nan_array(4)
        self.exp_right_ee_quat = nan_array(4)

        self.exp_left_force = nan_array(3)
        self.exp_right_force = nan_array(3)
        self.exp_left_torque = nan_array(3)
        self.exp_right_torque = nan_array(3)

        self.exp_left_force_cmd = nan_array(3)
        self.exp_right_force_cmd = nan_array(3)

        self.exp_left_q = nan_array(7)
        self.exp_right_q = nan_array(7)
        self.exp_left_q_goal = nan_array(7)
        self.exp_right_q_goal = nan_array(7)

        self.exp_left_omega_pos = nan_array(3)
        self.exp_right_omega_pos = nan_array(3)
        self.exp_left_omega_quat = nan_array(4)
        self.exp_right_omega_quat = nan_array(4)

        self.exp_left_gripper_angle = np.nan
        self.exp_right_gripper_angle = np.nan
        self.exp_left_gripper_button = False
        self.exp_right_gripper_button = False

        # Path-length state
        self.exp_prev_left_omega_pos = None
        self.exp_prev_right_omega_pos = None
        self.exp_prev_left_ee_pos = None
        self.exp_prev_right_ee_pos = None

        self.exp_left_omega_path_m = 0.0
        self.exp_right_omega_path_m = 0.0
        self.exp_left_ee_path_m = 0.0
        self.exp_right_ee_path_m = 0.0

        self.exp_start_monotonic = None
        self.exp_last_elapsed_s = 0.0
        self.exp_row_count = 0
        self.exp_closed = False

        # ----------------------------------------------------
        # Existing actual-q + PI + bilateral controller
        # ----------------------------------------------------
        super().__init__(
            mode="dual",
            force_mode=selected_force_mode,
            constant_force=np.zeros(3, dtype=float),
            enable_xy_force=False,
            force_gain_xy=0.08,
            force_gain_z=float(force_gain_z),
            force_limit=float(force_limit),
            force_deadband=float(force_deadband),
            lpf_alpha=float(lpf_alpha),
            bias_sample_count=int(bias_samples),
            force_sign=np.array(
                [1.0, 1.0, float(force_sign_z)],
                dtype=float,
            ),
            publish_every_n_loops=1,
            force_damping_xy=0.15,
            force_damping_z=float(force_damping_z),
            omega_gravity_comp=np.zeros(3, dtype=float),
        )

        # ----------------------------------------------------
        # Publish condition gate for the separate gripper
        # haptic-feedback path.
        # TRANSIENT_LOCAL is intentionally avoided here to
        # keep dependencies/simple QoS behavior. A 10-Hz timer
        # continuously republishes the selected condition.
        # ----------------------------------------------------
        self.exp_feedback_enable_pub = self.create_publisher(
            Bool,
            FEEDBACK_ENABLE_TOPIC,
            10,
        )

        self.exp_feedback_gate_timer = self.create_timer(
            0.1,
            self.publish_feedback_gate,
        )

        # ----------------------------------------------------
        # Measurement-only subscribers
        # These duplicate some controller subscriptions, but
        # write only exp_* variables and never modify control.
        # ----------------------------------------------------
        self.create_subscription(
            Float64,
            GRIP_FORCE_TOPIC,
            self.exp_grip_force_cb,
            10,
        )

        self.create_subscription(
            Float64,
            GRASP_ASSIST_OFFSET_TOPIC,
            self.exp_grasp_assist_offset_cb,
            10,
        )

        self.create_subscription(
            PoseStamped,
            LEFT_EE_POSE_TOPIC,
            self.exp_left_ee_pose_cb,
            10,
        )

        self.create_subscription(
            PoseStamped,
            RIGHT_EE_POSE_TOPIC,
            self.exp_right_ee_pose_cb,
            10,
        )

        self.create_subscription(
            WrenchStamped,
            LEFT_WRENCH_TOPIC,
            self.exp_left_wrench_cb,
            10,
        )

        self.create_subscription(
            WrenchStamped,
            RIGHT_WRENCH_TOPIC,
            self.exp_right_wrench_cb,
            10,
        )

        self.create_subscription(
            WrenchStamped,
            LEFT_FORCE_CMD_TOPIC,
            self.exp_left_force_cmd_cb,
            10,
        )

        self.create_subscription(
            WrenchStamped,
            RIGHT_FORCE_CMD_TOPIC,
            self.exp_right_force_cmd_cb,
            10,
        )

        self.create_subscription(
            JointState,
            LEFT_STATE_TOPIC,
            self.exp_left_state_cb,
            10,
        )

        self.create_subscription(
            JointState,
            RIGHT_STATE_TOPIC,
            self.exp_right_state_cb,
            10,
        )

        self.create_subscription(
            JointState,
            LEFT_Q_GOAL_TOPIC,
            self.exp_left_q_goal_cb,
            10,
        )

        self.create_subscription(
            JointState,
            RIGHT_Q_GOAL_TOPIC,
            self.exp_right_q_goal_cb,
            10,
        )

        self.create_subscription(
            State,
            LEFT_OMEGA_STATE_TOPIC,
            self.exp_left_omega_cb,
            10,
        )

        self.create_subscription(
            State,
            RIGHT_OMEGA_STATE_TOPIC,
            self.exp_right_omega_cb,
            10,
        )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------
        self.exp_output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        # Unique ID for this execution/trial.
        self.exp_run_id = (
            f"{stamp}_"
            f"{self.exp_condition}_"
            f"{safe_name(self.exp_subject)}_"
            f"trial{self.exp_trial:02d}_"
            f"{safe_name(self.exp_object_set)}"
        )

        # Append every Experiment-2 trial to ONE raw CSV.
        # run_id / subject / condition / trial make later filtering easy.
        self.exp_csv_path = (
            self.exp_output_root
            / "experiment2_all_trials.csv"
        )

        self.exp_fieldnames = self.make_fieldnames()

        csv_exists = (
            self.exp_csv_path.exists()
            and self.exp_csv_path.stat().st_size > 0
        )

        # Prevent accidental append with a different column layout.
        if csv_exists:
            with self.exp_csv_path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as check_file:
                existing_header = next(csv.reader(check_file))
            if existing_header != self.exp_fieldnames:
                raise RuntimeError(
                    "experiment2_all_trials.csv has a different column layout. "
                    "Rename or move the old CSV before using this version."
                )

        self.exp_csv_file = self.exp_csv_path.open(
            "a",
            newline="",
            encoding="utf-8",
        )

        self.exp_writer = csv.DictWriter(
            self.exp_csv_file,
            fieldnames=self.exp_fieldnames,
        )

        if not csv_exists:
            self.exp_writer.writeheader()

        self.exp_csv_file.flush()

        self.exp_start_monotonic = time.monotonic()

        self.exp_log_timer = self.create_timer(
            1.0 / self.exp_log_hz,
            self.log_row,
        )

        # Startup print disabled; CSV logging remains active.

    # ========================================================
    # Condition gate
    # ========================================================

    def publish_feedback_gate(self):
        msg = Bool()

        # Finger force feedback is part of Lifting.
        # C1/C3 = OFF, C2/C4 = ON.
        # Z force feedback remains controlled separately by force_mode.
        msg.data = bool(
            self.exp_spec["lifting"]
        )
        self.exp_feedback_enable_pub.publish(msg)

    # ========================================================
    # Measurement callbacks
    # ========================================================

    def exp_grip_force_cb(self, msg: Float64):
        self.exp_grip_force_n = float(msg.data)

    def exp_grasp_assist_offset_cb(
        self,
        msg: Float64,
    ):
        self.exp_grasp_assist_offset_m = float(
            msg.data
        )

    def exp_left_ee_pose_cb(
        self,
        msg: PoseStamped,
    ):
        self.exp_left_ee_pos = pose_position(msg)
        self.exp_left_ee_quat = pose_quaternion(msg)

    def exp_right_ee_pose_cb(
        self,
        msg: PoseStamped,
    ):
        self.exp_right_ee_pos = pose_position(msg)
        self.exp_right_ee_quat = pose_quaternion(msg)

    def exp_left_wrench_cb(
        self,
        msg: WrenchStamped,
    ):
        self.exp_left_force = wrench_force(msg)
        self.exp_left_torque = wrench_torque(msg)

    def exp_right_wrench_cb(
        self,
        msg: WrenchStamped,
    ):
        self.exp_right_force = wrench_force(msg)
        self.exp_right_torque = wrench_torque(msg)

    def exp_left_force_cmd_cb(
        self,
        msg: WrenchStamped,
    ):
        self.exp_left_force_cmd = wrench_force(msg)

    def exp_right_force_cmd_cb(
        self,
        msg: WrenchStamped,
    ):
        self.exp_right_force_cmd = wrench_force(msg)

    def exp_left_state_cb(
        self,
        msg: JointState,
    ):
        self.exp_left_q = joint7(msg)

    def exp_right_state_cb(
        self,
        msg: JointState,
    ):
        self.exp_right_q = joint7(msg)

    def exp_left_q_goal_cb(
        self,
        msg: JointState,
    ):
        self.exp_left_q_goal = joint7(msg)

    def exp_right_q_goal_cb(
        self,
        msg: JointState,
    ):
        self.exp_right_q_goal = joint7(msg)

    def exp_left_omega_cb(
        self,
        msg: State,
    ):
        self.exp_left_omega_pos = omega_position(msg)
        self.exp_left_omega_quat = omega_quaternion(msg)
        self.exp_left_gripper_angle = float(
            msg.gripper.angle
        )
        self.exp_left_gripper_button = bool(
            msg.gripper.button
        )

    def exp_right_omega_cb(
        self,
        msg: State,
    ):
        self.exp_right_omega_pos = omega_position(msg)
        self.exp_right_omega_quat = omega_quaternion(msg)
        self.exp_right_gripper_angle = float(
            msg.gripper.angle
        )
        self.exp_right_gripper_button = bool(
            msg.gripper.button
        )

    # ========================================================
    # Path lengths
    # ========================================================

    @staticmethod
    def path_increment(
        current: np.ndarray,
        previous,
    ):
        if not finite_xyz(current):
            return 0.0, previous

        current = np.asarray(
            current[:3],
            dtype=float,
        ).copy()

        if previous is None:
            return 0.0, current

        previous = np.asarray(
            previous[:3],
            dtype=float,
        )

        if not finite_xyz(previous):
            return 0.0, current

        step = float(
            np.linalg.norm(
                current - previous
            )
        )

        # A single impossible jump is not accumulated.
        # This is only a logging guard, not a control filter.
        if not math.isfinite(step) or step > 0.25:
            return 0.0, current

        return step, current

    def update_paths(self):
        step, self.exp_prev_left_omega_pos = (
            self.path_increment(
                self.exp_left_omega_pos,
                self.exp_prev_left_omega_pos,
            )
        )
        self.exp_left_omega_path_m += step

        step, self.exp_prev_right_omega_pos = (
            self.path_increment(
                self.exp_right_omega_pos,
                self.exp_prev_right_omega_pos,
            )
        )
        self.exp_right_omega_path_m += step

        step, self.exp_prev_left_ee_pos = (
            self.path_increment(
                self.exp_left_ee_pos,
                self.exp_prev_left_ee_pos,
            )
        )
        self.exp_left_ee_path_m += step

        step, self.exp_prev_right_ee_pos = (
            self.path_increment(
                self.exp_right_ee_pos,
                self.exp_prev_right_ee_pos,
            )
        )
        self.exp_right_ee_path_m += step

    # ========================================================
    # CSV
    # ========================================================

    @staticmethod
    def make_fieldnames():
        fields = [
            "wall_time",
            "elapsed_s",
            "run_id",

            "condition",
            "condition_label",
            "subject",
            "trial",
            "object_set",
            "note",

            "lifting_support_enabled",
            "force_feedback_requested",
            "cartesian_z_feedback_enabled",
            "grasp_assist_enabled",
            "grip_position_control_enabled",
            "midpoint_control_enabled",
            "actual_lifting_mode",

            "grip_force_topic_N",
            "grasp_assist_offset_m",
            "grasp_assist_offset_mm",

            "left_omega_path_m",
            "right_omega_path_m",
            "left_ee_path_m",
            "right_ee_path_m",
        ]

        for prefix in [
            "left_ee",
            "right_ee",
            "left_force",
            "right_force",
            "left_torque",
            "right_torque",
            "left_force_cmd",
            "right_force_cmd",
            "left_omega",
            "right_omega",
        ]:
            fields += [
                f"{prefix}_x",
                f"{prefix}_y",
                f"{prefix}_z",
            ]

        for prefix in [
            "left_ee",
            "right_ee",
            "left_omega",
            "right_omega",
        ]:
            fields += [
                f"{prefix}_qx",
                f"{prefix}_qy",
                f"{prefix}_qz",
                f"{prefix}_qw",
            ]

        fields += [
            "left_gripper_angle",
            "right_gripper_angle",
            "left_gripper_button",
            "right_gripper_button",
        ]

        for prefix in [
            "left_q",
            "right_q",
            "left_q_goal",
            "right_q_goal",
        ]:
            fields += [
                f"{prefix}{i}"
                for i in range(1, 8)
            ]

        return fields

    def log_row(self):
        if self.exp_closed:
            return

        now_mono = time.monotonic()

        if self.exp_start_monotonic is None:
            return

        elapsed_s = float(
            now_mono - self.exp_start_monotonic
        )

        self.exp_last_elapsed_s = elapsed_s

        self.update_paths()

        actual_lifting = bool(
            getattr(
                self,
                "lifting_mode",
                False,
            )
        )

        row = {
            "wall_time": datetime.now().isoformat(
                timespec="milliseconds"
            ),
            "elapsed_s": elapsed_s,
            "run_id": self.exp_run_id,

            "condition": self.exp_condition,
            "condition_label": self.exp_spec["label"],
            "subject": self.exp_subject,
            "trial": self.exp_trial,
            "object_set": self.exp_object_set,
            "note": self.exp_note,

            "lifting_support_enabled": int(
                self.exp_spec["lifting"]
            ),
            "force_feedback_requested": int(
                self.exp_spec["feedback"]
            ),
            "cartesian_z_feedback_enabled": int(
                self.exp_spec["feedback"]
            ),
            "grasp_assist_enabled": 1,
            "grip_position_control_enabled": int(
                self.exp_spec["finger_control"]
            ),
            "midpoint_control_enabled": 0,
            "actual_lifting_mode": int(
                actual_lifting
            ),

            "grip_force_topic_N": (
                self.exp_grip_force_n
            ),
            "grasp_assist_offset_m": (
                self.exp_grasp_assist_offset_m
            ),
            "grasp_assist_offset_mm": (
                1000.0
                * self.exp_grasp_assist_offset_m
                if math.isfinite(
                    self.exp_grasp_assist_offset_m
                )
                else np.nan
            ),

            "left_omega_path_m": (
                self.exp_left_omega_path_m
            ),
            "right_omega_path_m": (
                self.exp_right_omega_path_m
            ),
            "left_ee_path_m": (
                self.exp_left_ee_path_m
            ),
            "right_ee_path_m": (
                self.exp_right_ee_path_m
            ),

            "left_gripper_angle": (
                self.exp_left_gripper_angle
            ),
            "right_gripper_angle": (
                self.exp_right_gripper_angle
            ),
            "left_gripper_button": int(
                self.exp_left_gripper_button
            ),
            "right_gripper_button": int(
                self.exp_right_gripper_button
            ),
        }

        append_xyz(
            row,
            "left_ee",
            self.exp_left_ee_pos,
        )
        append_xyz(
            row,
            "right_ee",
            self.exp_right_ee_pos,
        )

        append_xyz(
            row,
            "left_force",
            self.exp_left_force,
        )
        append_xyz(
            row,
            "right_force",
            self.exp_right_force,
        )

        append_xyz(
            row,
            "left_torque",
            self.exp_left_torque,
        )
        append_xyz(
            row,
            "right_torque",
            self.exp_right_torque,
        )

        append_xyz(
            row,
            "left_force_cmd",
            self.exp_left_force_cmd,
        )
        append_xyz(
            row,
            "right_force_cmd",
            self.exp_right_force_cmd,
        )

        append_xyz(
            row,
            "left_omega",
            self.exp_left_omega_pos,
        )
        append_xyz(
            row,
            "right_omega",
            self.exp_right_omega_pos,
        )

        append_quat(
            row,
            "left_ee",
            self.exp_left_ee_quat,
        )
        append_quat(
            row,
            "right_ee",
            self.exp_right_ee_quat,
        )

        append_quat(
            row,
            "left_omega",
            self.exp_left_omega_quat,
        )
        append_quat(
            row,
            "right_omega",
            self.exp_right_omega_quat,
        )

        append_joint7(
            row,
            "left_q",
            self.exp_left_q,
        )
        append_joint7(
            row,
            "right_q",
            self.exp_right_q,
        )
        append_joint7(
            row,
            "left_q_goal",
            self.exp_left_q_goal,
        )
        append_joint7(
            row,
            "right_q_goal",
            self.exp_right_q_goal,
        )

        self.exp_writer.writerow(row)
        self.exp_row_count += 1

        if self.exp_row_count % 100 == 0:
            self.exp_csv_file.flush()

    # ========================================================
    # Summary / close
    # ========================================================

    def summary_path(self) -> Path:
        return (
            self.exp_output_root
            / "experiment2_summary.csv"
        )

    @staticmethod
    def summary_fields():
        return [
            "timestamp",
            "run_id",
            "condition",
            "condition_label",
            "subject",
            "trial",
            "object_set",
            "lifting_support_enabled",
            "force_feedback_requested",
            "cartesian_z_feedback_enabled",
            "grasp_assist_enabled",
            "grip_position_control_enabled",
            "midpoint_control_enabled",
            "duration_s",
            "left_omega_path_m",
            "right_omega_path_m",
            "omega_total_path_m",
            "left_ee_path_m",
            "right_ee_path_m",
            "ee_total_path_m",
            "raw_csv",
            "note",
        ]

    def append_summary(self):
        path = self.summary_path()
        exists = (
            path.exists()
            and path.stat().st_size > 0
        )

        row = {
            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "run_id": self.exp_run_id,
            "condition": self.exp_condition,
            "condition_label": self.exp_spec["label"],
            "subject": self.exp_subject,
            "trial": self.exp_trial,
            "object_set": self.exp_object_set,
            "lifting_support_enabled": int(
                self.exp_spec["lifting"]
            ),
            "force_feedback_requested": int(
                self.exp_spec["feedback"]
            ),
            "cartesian_z_feedback_enabled": int(
                self.exp_spec["feedback"]
            ),
            "grasp_assist_enabled": 1,
            "grip_position_control_enabled": int(
                self.exp_spec["finger_control"]
            ),
            "midpoint_control_enabled": 0,
            "duration_s": self.exp_last_elapsed_s,
            "left_omega_path_m": (
                self.exp_left_omega_path_m
            ),
            "right_omega_path_m": (
                self.exp_right_omega_path_m
            ),
            "omega_total_path_m": (
                self.exp_left_omega_path_m
                + self.exp_right_omega_path_m
            ),
            "left_ee_path_m": (
                self.exp_left_ee_path_m
            ),
            "right_ee_path_m": (
                self.exp_right_ee_path_m
            ),
            "ee_total_path_m": (
                self.exp_left_ee_path_m
                + self.exp_right_ee_path_m
            ),
            "raw_csv": str(
                self.exp_csv_path
            ),
            "note": self.exp_note,
        }

        with path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.summary_fields(),
            )
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def close_experiment_log(self):
        if self.exp_closed:
            return

        self.exp_closed = True

        try:
            self.exp_csv_file.flush()
            self.exp_csv_file.close()
        finally:
            self.append_summary()

        self.get_logger().info(
            "=========================================="
        )
        self.get_logger().info(
            "SII Experiment 2 log saved"
        )
        self.get_logger().info(
            f"raw CSV = {self.exp_csv_path}"
        )
        self.get_logger().info(
            f"duration = "
            f"{self.exp_last_elapsed_s:.3f} s"
        )
        self.get_logger().info(
            "Omega path total = "
            f"{self.exp_left_omega_path_m + self.exp_right_omega_path_m:.4f} m"
        )
        self.get_logger().info(
            "EE path total = "
            f"{self.exp_left_ee_path_m + self.exp_right_ee_path_m:.4f} m"
        )
        self.get_logger().info(
            f"summary = {self.summary_path()}"
        )
        self.get_logger().info(
            "=========================================="
        )

    # ========================================================
    # Startup display
    # ========================================================

    def print_startup(self):
        print()
        print("=" * 72)
        print(" SII EXPERIMENT 2")
        print("=" * 72)
        print(
            f"condition : {self.exp_condition} "
            f"({self.exp_spec['label']})"
        )
        print(
            "Lifting support     :",
            "ON"
            if self.exp_spec["lifting"]
            else "OFF",
        )
        print(
            "Force feedback      :",
            "ON"
            if self.exp_spec["feedback"]
            else "OFF",
        )
        print(
            "  Cartesian/Z path  :",
            "wrench"
            if self.exp_spec["feedback"]
            else "zero",
        )
        print("Grasp-force assist  : ON")
        print("Midpoint control    : OFF")
        print("Actual-Q normal DLS : ON")
        print("PI grasp assist     : ON")
        print(
            f"feedback gate topic : "
            f"{FEEDBACK_ENABLE_TOPIC}"
        )
        print(
            f"subject/trial       : "
            f"{self.exp_subject} / {self.exp_trial}"
        )
        print(
            f"raw log             : "
            f"{self.exp_csv_path}"
        )

        if not self.exp_spec["feedback"]:
            print()
            print(
                "[IMPORTANT] C1/C2:"
            )
            print(
                "  Z/translational force feedback is OFF here."
            )
            print(
                "  The separate right-Omega gripper inward-force feedback"
            )
            print(
                "  must also obey /sii_ex2/force_feedback_enabled."
            )
            print(
                "  /dual_grip/grip_force itself remains active because"
            )
            print(
                "  the grasp-force maintenance assist needs it."
            )

        print("=" * 72)
        print()


# ============================================================
# CLI
# ============================================================

def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "SII Experiment 2 C1-C4 runner + logger"
        )
    )

    p.add_argument(
        "--condition",
        required=True,
        type=str.upper,
        choices=["C1", "C2", "C3", "C4"],
    )

    p.add_argument(
        "--subject",
        required=True,
        help="e.g. S01",
    )

    p.add_argument(
        "--trial",
        required=True,
        type=int,
    )

    p.add_argument(
        "--object-set",
        default="3stack",
    )

    p.add_argument(
        "--note",
        default="",
    )

    p.add_argument(
        "--log-hz",
        type=float,
        default=100.0,
    )

    p.add_argument(
        "--output-root",
        default=str(LOG_ROOT_DEFAULT),
    )

    # Keep the Experiment-1 force-feedback defaults.
    p.add_argument(
        "--force-gain-z",
        type=float,
        default=0.12,
    )

    p.add_argument(
        "--force-limit",
        type=float,
        default=1.5,
    )

    p.add_argument(
        "--force-deadband",
        type=float,
        default=0.2,
    )

    p.add_argument(
        "--lpf-alpha",
        type=float,
        default=0.40,
    )

    p.add_argument(
        "--bias-samples",
        type=int,
        default=200,
    )

    p.add_argument(
        "--force-damping-z",
        type=float,
        default=0.15,
    )

    p.add_argument(
        "--force-sign-z",
        type=float,
        default=-1.0,
    )

    return p


def main():
    args = build_parser().parse_args()

    # Re-assert common settings immediately before constructing node.
    # actualq_pi already disables midpoint, but Experiment 2 should be
    # explicit and self-contained about its conditions.
    spec = CONDITIONS[args.condition]

    base.ENABLE_GRASP_ASSIST = True
    base.ENABLE_GRIP_POSITION_CONTROL = bool(
        spec["finger_control"]
    )
    base.ENABLE_RIGHT_MIDPOINT_CONTROL = False
    base.ENABLE_LEFT_MIDPOINT_CONTROL = False
    base.ENABLE_LIFTING_MODE = bool(
        spec["lifting"]
    )

    rclpy.init(args=["--ros-args", "--log-level", "WARN"])

    node = Experiment2Node(
        condition=args.condition,
        subject=args.subject,
        trial=args.trial,
        object_set=args.object_set,
        note=args.note,
        log_hz=args.log_hz,
        output_root=Path(args.output_root),
        force_gain_z=args.force_gain_z,
        force_limit=args.force_limit,
        force_deadband=args.force_deadband,
        lpf_alpha=args.lpf_alpha,
        bias_samples=args.bias_samples,
        force_damping_z=args.force_damping_z,
        force_sign_z=args.force_sign_z,
    )

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        # Stop Cartesian force output before shutdown.
        try:
            node.publish_zero_force_once()
            rclpy.spin_once(
                node,
                timeout_sec=0.05,
            )
        except Exception:
            pass

        try:
            node.close_experiment_log()
        except Exception as exc:
            print(
                "[WARN] failed to close Experiment-2 log:",
                exc,
            )

        try:
            node.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
