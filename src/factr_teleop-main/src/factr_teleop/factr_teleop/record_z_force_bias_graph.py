#!/usr/bin/env python3
"""
record_z_force_bias_graph.py

目的:
  2つのZ方向外力推定方法を同時に比較する。

  方法1:
    /franka/right/ee_wrench_world
    /franka/left/ee_wrench_world
    の WrenchStamped.force.z を使う方法

  方法2:
    /franka/right/obs_franka_state
    /franka/left/obs_franka_state
    /franka/right/obs_franka_torque
    /franka/left/obs_franka_torque
    を使い，関節外乱トルクから手先外力を推定する方法

      tau_ext = J(q)^T F_ext

    を解いて F_ext のZ成分を使う。

出力:
  ~/franka_ros2_ws/bilateral_logs_z_compare/YYYYmmdd_HHMMSS/
    z_force_method_comparison.csv
    z_force_method_comparison.png

実行:
  cd ~/franka_ros2_ws
  source /opt/ros/humble/setup.bash
  source install/setup.bash

  python3 ~/franka_ros2_ws/test_scripts/record_z_force_bias_graph.py

または，今の場所で実行するなら:
  python record_z_force_bias_graph.py

終了:
  Ctrl+C
"""

import os
import csv
import glob
import argparse
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped

try:
    import pinocchio as pin
except ImportError as e:
    raise ImportError(
        "pinocchio が import できません。pinocchio が使える環境で実行してください。"
    ) from e


# ==========================================================
# Topic settings
# ==========================================================
RIGHT_WRENCH_TOPIC = "/franka/right/ee_wrench_world"
LEFT_WRENCH_TOPIC = "/franka/left/ee_wrench_world"

RIGHT_STATE_TOPIC = "/franka/right/obs_franka_state"
LEFT_STATE_TOPIC = "/franka/left/obs_franka_state"

RIGHT_TORQUE_TOPIC = "/franka/right/obs_franka_torque"
LEFT_TORQUE_TOPIC = "/franka/left/obs_franka_torque"


# ==========================================================
# Default settings
# ==========================================================
DEFAULT_OUTPUT_ROOT = os.path.expanduser("~/franka_ros2_ws/bilateral_logs_z_compare")

DEFAULT_SAMPLE_PERIOD = 0.05
DEFAULT_BIAS_SEC = 2.0

DEFAULT_WRENCH_Z_SIGN = 1.0
DEFAULT_TORQUE_Z_SIGN = 1.0

DEFAULT_Z_DEADBAND = 0.2
DEFAULT_RATIO_MIN_TOTAL_LOAD = 0.5

DEFAULT_FORCE_YMIN = -5.0
DEFAULT_FORCE_YMAX = 10.0

DEFAULT_DAMPING = 0.02


# ==========================================================
# Utility
# ==========================================================
def find_default_urdf() -> Optional[str]:
    search_roots = [
        os.path.expanduser("~/franka_ros2_ws"),
        os.path.expanduser("~/ws_backup"),
        os.getcwd(),
    ]

    candidates = []

    for root in search_roots:
        if not os.path.isdir(root):
            continue

        pattern = os.path.join(root, "**", "*.urdf")
        candidates.extend(glob.glob(pattern, recursive=True))

    if len(candidates) == 0:
        return None

    def score(path: str) -> int:
        p = path.lower()
        base = os.path.basename(p)

        s = 0

        if "fr3" in p:
            s += 100
        if "panda" in p:
            s += 50

        if base == "fr3.urdf":
            s += 100
        if base == "panda.urdf":
            s += 50

        if "ros2_control" in p:
            s -= 20
        if "gazebo" in p:
            s -= 20
        if "hand" in p and "fr3" not in p:
            s -= 10

        return s

    candidates = sorted(candidates, key=score, reverse=True)

    return candidates[0]


def choose_frame_id(model: pin.Model, requested_frame: str) -> Tuple[int, str]:
    frame_names = [frame.name for frame in model.frames]

    if requested_frame != "auto":
        if model.existFrame(requested_frame):
            return model.getFrameId(requested_frame), requested_frame

        raise ValueError(
            f"指定されたee-frame '{requested_frame}' がURDF内にありません。\n"
            f"利用可能なframe例: {frame_names[-30:]}"
        )

    preferred_names = [
        "fr3_hand_tcp",
        "fr3_hand",
        "fr3_link8",
        "fr3_link7",
        "link_7",
        "panda_hand_tcp",
        "panda_hand",
        "panda_link8",
        "panda_link7",
        "ee",
        "tool0",
    ]

    for name in preferred_names:
        if model.existFrame(name):
            return model.getFrameId(name), name

    keywords = ["link8", "link_8", "link7", "link_7", "hand", "tcp", "ee"]

    for name in frame_names:
        lower = name.lower()
        if any(k in lower for k in keywords):
            return model.getFrameId(name), name

    raise ValueError(
        "手先frameを自動選択できませんでした。"
        " --ee-frame <frame_name> を指定してください。\n"
        f"利用可能なframe例: {frame_names[-30:]}"
    )


def unique_time_vector(
    times: np.ndarray,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(times)

    times = times[order]
    values = values[order]

    unique_times, unique_idx = np.unique(times, return_index=True)
    unique_values = values[unique_idx]

    return unique_times, unique_values


def interp_matrix(
    t_src: np.ndarray,
    x_src: np.ndarray,
    t_dst: np.ndarray,
) -> np.ndarray:
    dim = x_src.shape[1]
    out = np.zeros((len(t_dst), dim), dtype=float)

    for j in range(dim):
        out[:, j] = np.interp(t_dst, t_src, x_src[:, j])

    return out


def interp_vector(
    t_src: np.ndarray,
    x_src: np.ndarray,
    t_dst: np.ndarray,
) -> np.ndarray:
    return np.interp(t_dst, t_src, x_src)


def deadband(value: float, threshold: float) -> float:
    if abs(value) < threshold:
        return 0.0
    return value


def calc_ratio(left_load: float, right_load: float, min_total: float):
    total = left_load + right_load

    if total >= min_total:
        left_ratio = left_load / total
        right_ratio = right_load / total
        valid = 1
    else:
        left_ratio = np.nan
        right_ratio = np.nan
        valid = 0

    return total, left_ratio, right_ratio, valid


# ==========================================================
# Joint torque estimator
# ==========================================================
class JointTorqueWrenchEstimator:
    def __init__(
        self,
        urdf_path: str,
        ee_frame: str,
        damping: float,
    ):
        self.urdf_path = urdf_path
        self.damping = float(damping)

        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()

        self.frame_id, self.frame_name = choose_frame_id(self.model, ee_frame)

        if self.model.nq < 7:
            raise RuntimeError(
                f"URDF model.nq={self.model.nq} です。FR3の7関節モデルとして使えません。"
            )

    def estimate_wrench_from_tau(
        self,
        q_arm: np.ndarray,
        tau_ext_arm: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        q_arm = np.asarray(q_arm, dtype=float).reshape(-1)
        tau_ext_arm = np.asarray(tau_ext_arm, dtype=float).reshape(-1)

        if q_arm.shape[0] < 7:
            raise ValueError(f"q_arm length must be >= 7, got {q_arm.shape[0]}")

        if tau_ext_arm.shape[0] < 7:
            raise ValueError(
                f"tau_ext_arm length must be >= 7, got {tau_ext_arm.shape[0]}"
            )

        q_model = pin.neutral(self.model)
        q_model[:7] = q_arm[:7]

        tau7 = tau_ext_arm[:7]

        pin.forwardKinematics(self.model, self.data, q_model)
        pin.updateFramePlacements(self.model, self.data)

        # LOCAL_WORLD_ALIGNED:
        # 原点は手先，軸はworldに揃った表現
        # F_ext[2] をworld Z方向として読みたいのでこれを使う
        J6 = pin.computeFrameJacobian(
            self.model,
            self.data,
            q_model,
            self.frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        # FR3の7関節だけ使う
        J6_arm = J6[:, :7]

        # tau = J^T wrench
        A = J6_arm.T

        if self.damping > 0.0:
            lhs = A.T @ A + (self.damping ** 2) * np.eye(6)
            rhs = A.T @ tau7
            wrench6 = np.linalg.solve(lhs, rhs)
        else:
            wrench6, _, _, _ = np.linalg.lstsq(A, tau7, rcond=None)

        tau_pred = A @ wrench6
        residual_norm = float(np.linalg.norm(tau7 - tau_pred))

        force_world = wrench6[:3].copy()
        torque_world = wrench6[3:6].copy()

        return force_world, torque_world, residual_norm


# ==========================================================
# ROS Node
# ==========================================================
class ZForceMethodComparator(Node):
    def __init__(self, args):
        super().__init__("z_force_method_comparator")

        self.output_root = os.path.expanduser(args.output_root)
        self.sample_period = float(args.sample_period)
        self.bias_sec = float(args.bias_sec)

        self.wrench_z_sign = float(args.wrench_z_sign)
        self.torque_z_sign = float(args.torque_z_sign)

        self.z_deadband = float(args.z_deadband)
        self.ratio_min_total_load = float(args.ratio_min_total_load)

        self.force_ymin = float(args.force_ymin)
        self.force_ymax = float(args.force_ymax)

        self.time_source = str(args.time_source)

        urdf_path = args.urdf

        if urdf_path == "auto":
            urdf_path = find_default_urdf()

        if urdf_path is None:
            raise RuntimeError(
                "URDFを自動検出できませんでした。"
                " --urdf /path/to/fr3.urdf を指定してください。"
            )

        urdf_path = os.path.expanduser(urdf_path)

        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        self.estimator = JointTorqueWrenchEstimator(
            urdf_path=urdf_path,
            ee_frame=args.ee_frame,
            damping=args.damping,
        )

        self.start_ros_time = self.get_clock().now()

        # method 1: ee_wrench_world
        self.right_wrench_samples: List[Tuple[float, float]] = []
        self.left_wrench_samples: List[Tuple[float, float]] = []

        # method 2: joint torque estimation
        self.right_state_samples: List[Tuple[float, np.ndarray]] = []
        self.left_state_samples: List[Tuple[float, np.ndarray]] = []

        self.right_torque_samples: List[Tuple[float, np.ndarray]] = []
        self.left_torque_samples: List[Tuple[float, np.ndarray]] = []

        self.last_right_wrench_t = None
        self.last_left_wrench_t = None

        self.last_right_state_t = None
        self.last_left_state_t = None

        self.last_right_torque_t = None
        self.last_left_torque_t = None

        self.rows = []

        self.warned_short_right_state = False
        self.warned_short_left_state = False
        self.warned_short_right_torque = False
        self.warned_short_left_torque = False

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # method 1
        self.create_subscription(
            WrenchStamped,
            RIGHT_WRENCH_TOPIC,
            self.right_wrench_callback,
            sensor_qos,
        )

        self.create_subscription(
            WrenchStamped,
            LEFT_WRENCH_TOPIC,
            self.left_wrench_callback,
            sensor_qos,
        )

        # method 2
        self.create_subscription(
            JointState,
            RIGHT_STATE_TOPIC,
            self.right_state_callback,
            sensor_qos,
        )

        self.create_subscription(
            JointState,
            LEFT_STATE_TOPIC,
            self.left_state_callback,
            sensor_qos,
        )

        self.create_subscription(
            JointState,
            RIGHT_TORQUE_TOPIC,
            self.right_torque_callback,
            sensor_qos,
        )

        self.create_subscription(
            JointState,
            LEFT_TORQUE_TOPIC,
            self.left_torque_callback,
            sensor_qos,
        )

        self.get_logger().info("Z force method comparator started.")
        self.get_logger().info("=== direct wrench method ===")
        self.get_logger().info(f"right wrench topic: {RIGHT_WRENCH_TOPIC}")
        self.get_logger().info(f"left  wrench topic: {LEFT_WRENCH_TOPIC}")

        self.get_logger().info("=== joint torque estimation method ===")
        self.get_logger().info(f"right state topic : {RIGHT_STATE_TOPIC}")
        self.get_logger().info(f"left  state topic : {LEFT_STATE_TOPIC}")
        self.get_logger().info(f"right torque topic: {RIGHT_TORQUE_TOPIC}")
        self.get_logger().info(f"left  torque topic: {LEFT_TORQUE_TOPIC}")

        self.get_logger().info(f"output root: {self.output_root}")
        self.get_logger().info(f"URDF: {self.estimator.urdf_path}")
        self.get_logger().info(f"EE frame: {self.estimator.frame_name}")
        self.get_logger().info(
            f"Pinocchio nq={self.estimator.model.nq}, nv={self.estimator.model.nv}"
        )
        self.get_logger().info(f"damping: {self.estimator.damping:.4f}")
        self.get_logger().info(f"sample period: {self.sample_period:.3f} s")
        self.get_logger().info(f"bias sec: {self.bias_sec:.3f} s")
        self.get_logger().info(f"wrench_z_sign: {self.wrench_z_sign}")
        self.get_logger().info(f"torque_z_sign: {self.torque_z_sign}")
        self.get_logger().info(
            f"force graph y range: {self.force_ymin:.1f} to {self.force_ymax:.1f} N"
        )

    # ------------------------------------------------------
    # Time
    # ------------------------------------------------------
    def now_sec(self) -> float:
        now = self.get_clock().now()
        return (now - self.start_ros_time).nanoseconds * 1.0e-9

    def msg_time_sec(self, msg) -> float:
        receive_t = self.now_sec()

        stamp = msg.header.stamp
        header_t_abs = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9

        if self.time_source == "receive":
            return receive_t

        if self.time_source == "header":
            return header_t_abs

        # auto
        if header_t_abs <= 1.0e-9:
            return receive_t

        return header_t_abs

    @staticmethod
    def parse_first_7_position(msg: JointState) -> Optional[np.ndarray]:
        if len(msg.position) < 7:
            return None

        return np.asarray(msg.position[:7], dtype=float)

    # ------------------------------------------------------
    # Wrench callbacks
    # ------------------------------------------------------
    def right_wrench_callback(self, msg: WrenchStamped):
        t = self.msg_time_sec(msg)
        z = float(msg.wrench.force.z)

        if self.last_right_wrench_t is not None:
            if (t - self.last_right_wrench_t) < self.sample_period:
                return

        self.right_wrench_samples.append((t, z))
        self.last_right_wrench_t = t

    def left_wrench_callback(self, msg: WrenchStamped):
        t = self.msg_time_sec(msg)
        z = float(msg.wrench.force.z)

        if self.last_left_wrench_t is not None:
            if (t - self.last_left_wrench_t) < self.sample_period:
                return

        self.left_wrench_samples.append((t, z))
        self.last_left_wrench_t = t

    # ------------------------------------------------------
    # JointState callbacks
    # ------------------------------------------------------
    def right_state_callback(self, msg: JointState):
        t = self.msg_time_sec(msg)
        q = self.parse_first_7_position(msg)

        if q is None:
            if not self.warned_short_right_state:
                self.get_logger().warn(
                    f"Right state position length is {len(msg.position)} < 7"
                )
                self.warned_short_right_state = True
            return

        if self.last_right_state_t is not None:
            if (t - self.last_right_state_t) < self.sample_period:
                return

        self.right_state_samples.append((t, q))
        self.last_right_state_t = t

    def left_state_callback(self, msg: JointState):
        t = self.msg_time_sec(msg)
        q = self.parse_first_7_position(msg)

        if q is None:
            if not self.warned_short_left_state:
                self.get_logger().warn(
                    f"Left state position length is {len(msg.position)} < 7"
                )
                self.warned_short_left_state = True
            return

        if self.last_left_state_t is not None:
            if (t - self.last_left_state_t) < self.sample_period:
                return

        self.left_state_samples.append((t, q))
        self.last_left_state_t = t

    def right_torque_callback(self, msg: JointState):
        t = self.msg_time_sec(msg)
        tau = self.parse_first_7_position(msg)

        if tau is None:
            if not self.warned_short_right_torque:
                self.get_logger().warn(
                    f"Right torque position length is {len(msg.position)} < 7"
                )
                self.warned_short_right_torque = True
            return

        if self.last_right_torque_t is not None:
            if (t - self.last_right_torque_t) < self.sample_period:
                return

        self.right_torque_samples.append((t, tau))
        self.last_right_torque_t = t

    def left_torque_callback(self, msg: JointState):
        t = self.msg_time_sec(msg)
        tau = self.parse_first_7_position(msg)

        if tau is None:
            if not self.warned_short_left_torque:
                self.get_logger().warn(
                    f"Left torque position length is {len(msg.position)} < 7"
                )
                self.warned_short_left_torque = True
            return

        if self.last_left_torque_t is not None:
            if (t - self.last_left_torque_t) < self.sample_period:
                return

        self.left_torque_samples.append((t, tau))
        self.last_left_torque_t = t

    # ------------------------------------------------------
    # Prepare streams
    # ------------------------------------------------------
    def prepare_scalar_stream(
        self,
        samples: List[Tuple[float, float]],
        label: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if len(samples) < 2:
            print(f"[WARN] Not enough samples for {label}: {len(samples)}")
            return None, None

        t = np.array([s[0] for s in samples], dtype=float)
        x = np.array([s[1] for s in samples], dtype=float).reshape(-1, 1)

        t, x = unique_time_vector(t, x)

        if len(t) < 2:
            print(f"[WARN] Not enough unique timestamp samples for {label}.")
            return None, None

        return t, x[:, 0]

    def prepare_vector_stream(
        self,
        samples: List[Tuple[float, np.ndarray]],
        label: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if len(samples) < 2:
            print(f"[WARN] Not enough samples for {label}: {len(samples)}")
            return None, None

        t = np.array([s[0] for s in samples], dtype=float)
        x = np.vstack([s[1] for s in samples]).astype(float)

        t, x = unique_time_vector(t, x)

        if len(t) < 2:
            print(f"[WARN] Not enough unique timestamp samples for {label}.")
            return None, None

        return t, x

    # ------------------------------------------------------
    # Build synchronized rows
    # ------------------------------------------------------
    def build_synchronized_rows(self):
        rw_t_abs, rw_z = self.prepare_scalar_stream(
            self.right_wrench_samples,
            "right_wrench",
        )
        lw_t_abs, lw_z = self.prepare_scalar_stream(
            self.left_wrench_samples,
            "left_wrench",
        )

        rs_t_abs, rs_q = self.prepare_vector_stream(
            self.right_state_samples,
            "right_state",
        )
        ls_t_abs, ls_q = self.prepare_vector_stream(
            self.left_state_samples,
            "left_state",
        )

        rt_t_abs, rt_tau = self.prepare_vector_stream(
            self.right_torque_samples,
            "right_torque",
        )
        lt_t_abs, lt_tau = self.prepare_vector_stream(
            self.left_torque_samples,
            "left_torque",
        )

        streams = [
            rw_t_abs,
            lw_t_abs,
            rs_t_abs,
            ls_t_abs,
            rt_t_abs,
            lt_t_abs,
        ]

        if any(s is None for s in streams):
            return []

        t0 = min(s[0] for s in streams)

        rw_t = rw_t_abs - t0
        lw_t = lw_t_abs - t0
        rs_t = rs_t_abs - t0
        ls_t = ls_t_abs - t0
        rt_t = rt_t_abs - t0
        lt_t = lt_t_abs - t0

        start_t = max(
            rw_t[0],
            lw_t[0],
            rs_t[0],
            ls_t[0],
            rt_t[0],
            lt_t[0],
        )

        end_t = min(
            rw_t[-1],
            lw_t[-1],
            rs_t[-1],
            ls_t[-1],
            rt_t[-1],
            lt_t[-1],
        )

        if end_t <= start_t:
            print(f"[WARN] No overlapping time range. start={start_t:.3f}, end={end_t:.3f}")
            return []

        t_common = np.arange(start_t, end_t, self.sample_period)

        if len(t_common) < 2:
            print("[WARN] Common time axis too short.")
            return []

        # method 1: direct wrench
        right_z_wrench_raw = interp_vector(rw_t, rw_z, t_common)
        left_z_wrench_raw = interp_vector(lw_t, lw_z, t_common)

        # method 2: joint torque estimation
        right_q = interp_matrix(rs_t, rs_q, t_common)
        left_q = interp_matrix(ls_t, ls_q, t_common)

        right_tau = interp_matrix(rt_t, rt_tau, t_common)
        left_tau = interp_matrix(lt_t, lt_tau, t_common)

        right_force_torque_raw = []
        left_force_torque_raw = []

        right_residual = []
        left_residual = []

        for i in range(len(t_common)):
            rf, _, rr = self.estimator.estimate_wrench_from_tau(
                right_q[i],
                right_tau[i],
            )
            lf, _, lr = self.estimator.estimate_wrench_from_tau(
                left_q[i],
                left_tau[i],
            )

            right_force_torque_raw.append(rf)
            left_force_torque_raw.append(lf)

            right_residual.append(rr)
            left_residual.append(lr)

        right_force_torque_raw = np.vstack(right_force_torque_raw)
        left_force_torque_raw = np.vstack(left_force_torque_raw)

        right_residual = np.asarray(right_residual, dtype=float)
        left_residual = np.asarray(left_residual, dtype=float)

        t_rel = t_common - t_common[0]

        bias_mask = t_rel <= self.bias_sec

        if np.any(bias_mask):
            right_wrench_bias = float(np.mean(right_z_wrench_raw[bias_mask]))
            left_wrench_bias = float(np.mean(left_z_wrench_raw[bias_mask]))

            right_torque_bias = float(np.mean(right_force_torque_raw[bias_mask, 2]))
            left_torque_bias = float(np.mean(left_force_torque_raw[bias_mask, 2]))
        else:
            n = min(10, len(t_common))

            right_wrench_bias = float(np.mean(right_z_wrench_raw[:n]))
            left_wrench_bias = float(np.mean(left_z_wrench_raw[:n]))

            right_torque_bias = float(np.mean(right_force_torque_raw[:n, 2]))
            left_torque_bias = float(np.mean(left_force_torque_raw[:n, 2]))

        print(
            "[INFO] wrench bias: "
            f"right={right_wrench_bias:.3f}, "
            f"left={left_wrench_bias:.3f}"
        )

        print(
            "[INFO] torque-est bias: "
            f"right={right_torque_bias:.3f}, "
            f"left={left_torque_bias:.3f}"
        )

        rows = []

        for i, t in enumerate(t_rel):
            # bias区間はグラフとCSVから除外
            if t < self.bias_sec:
                continue

            # ------------------------------
            # method 1: ee_wrench_world
            # ------------------------------
            rz_wrench_raw = float(right_z_wrench_raw[i])
            lz_wrench_raw = float(left_z_wrench_raw[i])

            rz_wrench_corrected = self.wrench_z_sign * (
                rz_wrench_raw - right_wrench_bias
            )
            lz_wrench_corrected = self.wrench_z_sign * (
                lz_wrench_raw - left_wrench_bias
            )

            rz_wrench_corrected = deadband(
                rz_wrench_corrected,
                self.z_deadband,
            )
            lz_wrench_corrected = deadband(
                lz_wrench_corrected,
                self.z_deadband,
            )

            right_wrench_load = max(0.0, rz_wrench_corrected)
            left_wrench_load = max(0.0, lz_wrench_corrected)

            (
                total_wrench_load,
                left_wrench_ratio,
                right_wrench_ratio,
                wrench_ratio_valid,
            ) = calc_ratio(
                left_wrench_load,
                right_wrench_load,
                self.ratio_min_total_load,
            )

            # ------------------------------
            # method 2: joint torque estimation
            # ------------------------------
            rz_torque_raw = float(right_force_torque_raw[i, 2])
            lz_torque_raw = float(left_force_torque_raw[i, 2])

            rz_torque_corrected = self.torque_z_sign * (
                rz_torque_raw - right_torque_bias
            )
            lz_torque_corrected = self.torque_z_sign * (
                lz_torque_raw - left_torque_bias
            )

            rz_torque_corrected = deadband(
                rz_torque_corrected,
                self.z_deadband,
            )
            lz_torque_corrected = deadband(
                lz_torque_corrected,
                self.z_deadband,
            )

            right_torque_load = max(0.0, rz_torque_corrected)
            left_torque_load = max(0.0, lz_torque_corrected)

            (
                total_torque_load,
                left_torque_ratio,
                right_torque_ratio,
                torque_ratio_valid,
            ) = calc_ratio(
                left_torque_load,
                right_torque_load,
                self.ratio_min_total_load,
            )

            # ------------------------------
            # Difference
            # ------------------------------
            right_diff = rz_torque_corrected - rz_wrench_corrected
            left_diff = lz_torque_corrected - lz_wrench_corrected

            rows.append(
                {
                    "t": float(t - self.bias_sec),

                    # direct wrench method
                    "right_z_wrench_raw": rz_wrench_raw,
                    "left_z_wrench_raw": lz_wrench_raw,
                    "right_z_wrench_bias": right_wrench_bias,
                    "left_z_wrench_bias": left_wrench_bias,
                    "right_z_wrench_corrected": rz_wrench_corrected,
                    "left_z_wrench_corrected": lz_wrench_corrected,
                    "right_wrench_load": right_wrench_load,
                    "left_wrench_load": left_wrench_load,
                    "total_wrench_load": total_wrench_load,
                    "left_wrench_ratio": left_wrench_ratio,
                    "right_wrench_ratio": right_wrench_ratio,
                    "wrench_ratio_valid": wrench_ratio_valid,

                    # joint torque estimation method
                    "right_z_torque_raw_est": rz_torque_raw,
                    "left_z_torque_raw_est": lz_torque_raw,
                    "right_z_torque_bias": right_torque_bias,
                    "left_z_torque_bias": left_torque_bias,
                    "right_z_torque_corrected": rz_torque_corrected,
                    "left_z_torque_corrected": lz_torque_corrected,
                    "right_torque_load": right_torque_load,
                    "left_torque_load": left_torque_load,
                    "total_torque_load": total_torque_load,
                    "left_torque_ratio": left_torque_ratio,
                    "right_torque_ratio": right_torque_ratio,
                    "torque_ratio_valid": torque_ratio_valid,

                    # difference
                    "right_torque_minus_wrench": right_diff,
                    "left_torque_minus_wrench": left_diff,

                    # least-square residual
                    "right_lstsq_residual_norm": float(right_residual[i]),
                    "left_lstsq_residual_norm": float(left_residual[i]),
                }
            )

        return rows

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------
    def save_outputs(self):
        self.rows = self.build_synchronized_rows()

        if len(self.rows) == 0:
            print("[WARN] No synchronized data recorded.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_root, timestamp)
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, "z_force_method_comparison.csv")
        png_path = os.path.join(output_dir, "z_force_method_comparison.png")

        self.save_csv(csv_path)
        print(f"[INFO] Saved CSV : {csv_path}")

        self.save_plot(png_path)
        print(f"[INFO] Saved plot: {png_path}")

        print(
            "[INFO] raw samples: "
            f"right_wrench={len(self.right_wrench_samples)}, "
            f"left_wrench={len(self.left_wrench_samples)}, "
            f"right_state={len(self.right_state_samples)}, "
            f"left_state={len(self.left_state_samples)}, "
            f"right_torque={len(self.right_torque_samples)}, "
            f"left_torque={len(self.left_torque_samples)}"
        )

        print(f"[INFO] synchronized rows: {len(self.rows)}")

    def save_csv(self, csv_path: str):
        fieldnames = list(self.rows[0].keys())

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in self.rows:
                writer.writerow(row)

    def save_plot(self, png_path: str):
        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        t = np.array([row["t"] for row in self.rows], dtype=float)

        # method 1: ee_wrench_world
        left_wrench = np.array(
            [row["left_z_wrench_corrected"] for row in self.rows],
            dtype=float,
        )
        right_wrench = np.array(
            [row["right_z_wrench_corrected"] for row in self.rows],
            dtype=float,
        )

        # method 2: joint torque estimation
        left_torque = np.array(
            [row["left_z_torque_corrected"] for row in self.rows],
            dtype=float,
        )
        right_torque = np.array(
            [row["right_z_torque_corrected"] for row in self.rows],
            dtype=float,
        )

        left_diff = np.array(
            [row["left_torque_minus_wrench"] for row in self.rows],
            dtype=float,
        )
        right_diff = np.array(
            [row["right_torque_minus_wrench"] for row in self.rows],
            dtype=float,
        )

        left_wrench_ratio = 100.0 * np.array(
            [row["left_wrench_ratio"] for row in self.rows],
            dtype=float,
        )
        left_torque_ratio = 100.0 * np.array(
            [row["left_torque_ratio"] for row in self.rows],
            dtype=float,
        )

        wrench_ratio_valid = np.array(
            [row["wrench_ratio_valid"] for row in self.rows],
            dtype=bool,
        )
        torque_ratio_valid = np.array(
            [row["torque_ratio_valid"] for row in self.rows],
            dtype=bool,
        )

        # plot() には where が使えないので，
        # 無効な割合区間は NaN にして描画しないようにする
        left_wrench_ratio_plot = left_wrench_ratio.copy()
        left_torque_ratio_plot = left_torque_ratio.copy()

        left_wrench_ratio_plot[~wrench_ratio_valid] = np.nan
        left_torque_ratio_plot[~torque_ratio_valid] = np.nan

        fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

        # ------------------------------
        # 1. Left comparison
        # ------------------------------
        axes[0].plot(t, left_wrench, label="Left ee_wrench_world Z")
        axes[0].plot(
            t,
            left_torque,
            label="Left joint-torque-estimated Z",
            linestyle="--",
        )
        axes[0].axhline(0.0, linestyle=":", linewidth=1.0)
        axes[0].set_ylabel("Force [N]")
        axes[0].set_title("Left FR3 Z-force comparison")
        axes[0].set_ylim(self.force_ymin, self.force_ymax)
        axes[0].grid(True)
        axes[0].legend(loc="upper right")

        # ------------------------------
        # 2. Right comparison
        # ------------------------------
        axes[1].plot(t, right_wrench, label="Right ee_wrench_world Z")
        axes[1].plot(
            t,
            right_torque,
            label="Right joint-torque-estimated Z",
            linestyle="--",
        )
        axes[1].axhline(0.0, linestyle=":", linewidth=1.0)
        axes[1].set_ylabel("Force [N]")
        axes[1].set_title("Right FR3 Z-force comparison")
        axes[1].set_ylim(self.force_ymin, self.force_ymax)
        axes[1].grid(True)
        axes[1].legend(loc="upper right")

        # ------------------------------
        # 3. Difference
        # ------------------------------
        axes[2].plot(t, left_diff, label="Left torque-est - wrench")
        axes[2].plot(t, right_diff, label="Right torque-est - wrench")
        axes[2].axhline(0.0, linestyle=":", linewidth=1.0)
        axes[2].set_ylabel("Difference [N]")
        axes[2].set_title("Difference between methods")
        axes[2].grid(True)
        axes[2].legend(loc="upper right")

        # ------------------------------
        # 4. Load sharing ratio comparison
        # ------------------------------
        axes[3].plot(
            t,
            left_wrench_ratio_plot,
            label="Left ratio from ee_wrench_world",
        )
        axes[3].plot(
            t,
            left_torque_ratio_plot,
            label="Left ratio from joint torque",
            linestyle="--",
        )
        axes[3].axhline(50.0, linestyle=":", linewidth=1.5, label="Balanced 50%")
        axes[3].set_xlabel("Time [s]")
        axes[3].set_ylabel("Left load ratio [%]")
        axes[3].set_title("Left load sharing ratio comparison")
        axes[3].set_ylim(0.0, 100.0)
        axes[3].grid(True)
        axes[3].legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(png_path, dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where CSV and PNG are saved.",
    )

    parser.add_argument(
        "--sample-period",
        type=float,
        default=DEFAULT_SAMPLE_PERIOD,
        help="Sampling / interpolation period [s]. Default 0.05 = 20 Hz.",
    )

    parser.add_argument(
        "--bias-sec",
        type=float,
        default=DEFAULT_BIAS_SEC,
        help="Initial seconds for Z-force bias estimation.",
    )

    parser.add_argument(
        "--wrench-z-sign",
        type=float,
        default=DEFAULT_WRENCH_Z_SIGN,
        help="Use 1.0 or -1.0 for ee_wrench_world Z sign.",
    )

    parser.add_argument(
        "--torque-z-sign",
        type=float,
        default=DEFAULT_TORQUE_Z_SIGN,
        help="Use 1.0 or -1.0 for joint-torque-estimated Z sign.",
    )

    parser.add_argument(
        "--z-deadband",
        type=float,
        default=DEFAULT_Z_DEADBAND,
        help="Deadband for Z force [N].",
    )

    parser.add_argument(
        "--ratio-min-total-load",
        type=float,
        default=DEFAULT_RATIO_MIN_TOTAL_LOAD,
        help="Minimum total load [N] to calculate load ratio.",
    )

    parser.add_argument(
        "--force-ymin",
        type=float,
        default=DEFAULT_FORCE_YMIN,
        help="Y-axis minimum force [N] for force comparison graphs.",
    )

    parser.add_argument(
        "--force-ymax",
        type=float,
        default=DEFAULT_FORCE_YMAX,
        help="Y-axis maximum force [N] for force comparison graphs.",
    )

    parser.add_argument(
        "--time-source",
        type=str,
        default="auto",
        choices=["auto", "header", "receive"],
        help="Time source for samples.",
    )

    parser.add_argument(
        "--urdf",
        type=str,
        default="auto",
        help="Path to FR3 URDF. Use 'auto' to search under ~/franka_ros2_ws.",
    )

    parser.add_argument(
        "--ee-frame",
        type=str,
        default="auto",
        help="End-effector frame name in URDF. Use 'auto' for automatic selection.",
    )

    parser.add_argument(
        "--damping",
        type=float,
        default=DEFAULT_DAMPING,
        help="Damping for solving J^T F = tau. Set 0.0 for ordinary least squares.",
    )

    args = parser.parse_args()

    rclpy.init()

    node = ZForceMethodComparator(args)

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("[INFO] Ctrl+C received. Saving outputs...")

    finally:
        try:
            node.save_outputs()
        except Exception as e:
            print(f"[ERROR] Failed to save outputs: {e}")

        try:
            node.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# record_z_force_bias_graph.py

# 目的:
#   /franka/right/obs_franka_state
#   /franka/right/obs_franka_torque
#   /franka/left/obs_franka_state
#   /franka/left/obs_franka_torque

#   を使って，各関節外乱トルクから手先外力を推定し，
#   左右FR3のZ方向荷重と荷重分担割合をグラフ化する。

# 重要:
#   以前の /franka/*/ee_wrench_world は使わない。
#   関節外乱トルク tau_ext と手先ヤコビアン J(q) から

#       tau_ext = J(q)^T F_ext

#   を解き，F_ext のZ成分を使う。

# 出力:
#   ~/franka_ros2_ws/bilateral_logs_z/YYYYmmdd_HHMMSS/
#     z_force_log.csv
#     z_force_and_ratio.png

# 実行:
#   cd ~/franka_ros2_ws
#   source /opt/ros/humble/setup.bash
#   source install/setup.bash

#   nice -n 10 python3 ~/franka_ros2_ws/test_scripts/record_z_force_bias_graph.py

# 終了:
#   Ctrl+C
# """

# import os
# import csv
# import glob
# import argparse
# from datetime import datetime
# from typing import List, Tuple, Optional

# import numpy as np

# import rclpy
# from rclpy.node import Node
# from rclpy.qos import (
#     QoSProfile,
#     ReliabilityPolicy,
#     HistoryPolicy,
#     DurabilityPolicy,
# )
# from sensor_msgs.msg import JointState

# try:
#     import pinocchio as pin
# except ImportError as e:
#     raise ImportError(
#         "pinocchio が import できません。先に pinocchio が使える環境で実行してください。"
#     ) from e


# # ==========================================================
# # Topic settings
# # ==========================================================
# RIGHT_STATE_TOPIC = "/franka/right/obs_franka_state"
# LEFT_STATE_TOPIC = "/franka/left/obs_franka_state"

# RIGHT_TORQUE_TOPIC = "/franka/right/obs_franka_torque"
# LEFT_TORQUE_TOPIC = "/franka/left/obs_franka_torque"


# # ==========================================================
# # Default settings
# # ==========================================================
# DEFAULT_OUTPUT_ROOT = os.path.expanduser("~/franka_ros2_ws/bilateral_logs_z")

# # 実機でZ方向が逆なら -1.0 にする
# DEFAULT_Z_SIGN = 1.0

# # callbackで受け取った全データを保存せず，この間隔以上空いたときだけ保存する
# DEFAULT_SAMPLE_PERIOD = 0.05

# # 割合計算するときの最小合計荷重 [N]
# DEFAULT_RATIO_MIN_TOTAL_LOAD = 0.5

# # 1段目のZ荷重グラフ範囲 [N]
# DEFAULT_FORCE_YMIN = -5.0
# DEFAULT_FORCE_YMAX = 10.0

# # J^T F = tau を解くときの減衰係数
# # 0.0 にすると通常の最小二乗になる。
# # 特異姿勢付近では少し入れた方が安定する。
# DEFAULT_DAMPING = 0.02


# # ==========================================================
# # Utility
# # ==========================================================
# def find_default_urdf() -> Optional[str]:
#     """
#     franka_ros2_ws 配下から fr3/panda のURDFを探す。
#     見つからない場合は None を返す。
#     """

#     search_roots = [
#         os.path.expanduser("~/franka_ros2_ws"),
#         os.path.expanduser("~/ws_backup"),
#         os.getcwd(),
#     ]

#     candidates = []

#     for root in search_roots:
#         if not os.path.isdir(root):
#             continue

#         pattern = os.path.join(root, "**", "*.urdf")
#         candidates.extend(glob.glob(pattern, recursive=True))

#     if len(candidates) == 0:
#         return None

#     def score(path: str) -> int:
#         p = path.lower()
#         s = 0

#         # FR3を優先
#         if "fr3" in p:
#             s += 100

#         # pandaも一応候補
#         if "panda" in p:
#             s += 50

#         # ファイル名が短くて本体っぽいものを優先
#         base = os.path.basename(p)
#         if base == "fr3.urdf":
#             s += 100
#         if base == "panda.urdf":
#             s += 50

#         # ros2_controlやgazebo専用っぽいものは少し下げる
#         if "ros2_control" in p:
#             s -= 20
#         if "gazebo" in p:
#             s -= 20
#         if "hand" in p and "fr3" not in p:
#             s -= 10

#         return s

#     candidates = sorted(candidates, key=score, reverse=True)

#     return candidates[0]


# def choose_frame_id(model: pin.Model, requested_frame: str) -> Tuple[int, str]:
#     """
#     requested_frame が auto の場合，よくある手先フレーム名から自動選択する。
#     """

#     frame_names = [frame.name for frame in model.frames]

#     if requested_frame != "auto":
#         if model.existFrame(requested_frame):
#             return model.getFrameId(requested_frame), requested_frame

#         raise ValueError(
#             f"指定されたee-frame '{requested_frame}' がURDF内にありません。\n"
#             f"利用可能なframe例: {frame_names[-30:]}"
#         )

#     preferred_names = [
#         "fr3_hand_tcp",
#         "fr3_hand",
#         "fr3_link8",
#         "fr3_link7",
#         "link_7",
#         "panda_hand_tcp",
#         "panda_hand",
#         "panda_link8",
#         "panda_link7",
#         "ee",
#         "tool0",
#     ]

#     for name in preferred_names:
#         if model.existFrame(name):
#             return model.getFrameId(name), name

#     # 最後の手段として，名前に link7/link8/hand/tcp が入るframeを探す
#     keywords = ["link8", "link_8", "link7", "link_7", "hand", "tcp", "ee"]
#     for name in frame_names:
#         lower = name.lower()
#         if any(k in lower for k in keywords):
#             return model.getFrameId(name), name

#     raise ValueError(
#         "手先frameを自動選択できませんでした。"
#         " --ee-frame <frame_name> を指定してください。\n"
#         f"利用可能なframe例: {frame_names[-30:]}"
#     )


# def unique_time_vector(
#     times: np.ndarray,
#     values: np.ndarray,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     時刻順に並べ，重複時刻を除去する。
#     values は shape=(N,D)
#     """

#     order = np.argsort(times)
#     times = times[order]
#     values = values[order]

#     unique_times, unique_idx = np.unique(times, return_index=True)
#     unique_values = values[unique_idx]

#     return unique_times, unique_values


# def interp_matrix(
#     t_src: np.ndarray,
#     x_src: np.ndarray,
#     t_dst: np.ndarray,
# ) -> np.ndarray:
#     """
#     多次元配列を各列ごとに線形補間する。
#     x_src shape = (N, D)
#     return shape = (len(t_dst), D)
#     """

#     dim = x_src.shape[1]
#     out = np.zeros((len(t_dst), dim), dtype=float)

#     for j in range(dim):
#         out[:, j] = np.interp(t_dst, t_src, x_src[:, j])

#     return out


# # ==========================================================
# # Estimator
# # ==========================================================
# class JointTorqueWrenchEstimator:
#     def __init__(
#         self,
#         urdf_path: str,
#         ee_frame: str,
#         damping: float,
#     ):
#         self.urdf_path = urdf_path
#         self.damping = float(damping)

#         self.model = pin.buildModelFromUrdf(self.urdf_path)
#         self.data = self.model.createData()

#         self.frame_id, self.frame_name = choose_frame_id(self.model, ee_frame)

#         if self.model.nq < 7:
#             raise RuntimeError(
#                 f"URDF model.nq={self.model.nq} です。FR3の7関節モデルとして使えません。"
#             )

#     def estimate_wrench_from_tau(
#         self,
#         q_arm: np.ndarray,
#         tau_ext_arm: np.ndarray,
#     ) -> Tuple[np.ndarray, np.ndarray, float]:
#         """
#         q_arm:
#           shape=(7,)

#         tau_ext_arm:
#           shape=(7,)
#           FACTRの /franka/*/obs_franka_torque の position[:7] を想定

#         return:
#           force_world: shape=(3,)
#           torque_world: shape=(3,)
#           residual_norm: float
#         """

#         q_arm = np.asarray(q_arm, dtype=float).reshape(-1)
#         tau_ext_arm = np.asarray(tau_ext_arm, dtype=float).reshape(-1)

#         if q_arm.shape[0] < 7:
#             raise ValueError(f"q_arm length must be >= 7, got {q_arm.shape[0]}")
#         if tau_ext_arm.shape[0] < 7:
#             raise ValueError(
#                 f"tau_ext_arm length must be >= 7, got {tau_ext_arm.shape[0]}"
#             )

#         q_model = pin.neutral(self.model)
#         q_model[:7] = q_arm[:7]

#         tau7 = tau_ext_arm[:7]

#         pin.forwardKinematics(self.model, self.data, q_model)
#         pin.updateFramePlacements(self.model, self.data)

#         # LOCAL_WORLD_ALIGNED:
#         #   原点は手先，軸はworldに揃った表現。
#         #   Z方向をworld Zとして読みたいのでこれを使う。
#         J6 = pin.computeFrameJacobian(
#             self.model,
#             self.data,
#             q_model,
#             self.frame_id,
#             pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
#         )

#         # FR3の7関節だけ使う
#         J6_arm = J6[:, :7]

#         # tau = J^T wrench
#         # A wrench = tau として解く
#         A = J6_arm.T  # shape=(7,6)

#         if self.damping > 0.0:
#             lhs = A.T @ A + (self.damping ** 2) * np.eye(6)
#             rhs = A.T @ tau7
#             wrench6 = np.linalg.solve(lhs, rhs)
#         else:
#             wrench6, _, _, _ = np.linalg.lstsq(A, tau7, rcond=None)

#         tau_pred = A @ wrench6
#         residual_norm = float(np.linalg.norm(tau7 - tau_pred))

#         force_world = wrench6[:3].copy()
#         torque_world = wrench6[3:6].copy()

#         return force_world, torque_world, residual_norm


# # ==========================================================
# # ROS Node
# # ==========================================================
# class ZForceLogger(Node):
#     def __init__(self, args):
#         super().__init__("z_force_logger_joint_torque_estimation")

#         self.output_root = os.path.expanduser(args.output_root)
#         self.sample_period = float(args.sample_period)
#         self.bias_sec = float(args.bias_sec)
#         self.z_sign = float(args.z_sign)
#         self.z_deadband = float(args.z_deadband)
#         self.ratio_min_total_load = float(args.ratio_min_total_load)
#         self.force_ymin = float(args.force_ymin)
#         self.force_ymax = float(args.force_ymax)
#         self.time_source = str(args.time_source)

#         urdf_path = args.urdf
#         if urdf_path == "auto":
#             urdf_path = find_default_urdf()

#         if urdf_path is None:
#             raise RuntimeError(
#                 "URDFを自動検出できませんでした。"
#                 " --urdf /path/to/fr3.urdf を指定してください。"
#             )

#         urdf_path = os.path.expanduser(urdf_path)

#         if not os.path.exists(urdf_path):
#             raise FileNotFoundError(f"URDF not found: {urdf_path}")

#         self.estimator = JointTorqueWrenchEstimator(
#             urdf_path=urdf_path,
#             ee_frame=args.ee_frame,
#             damping=args.damping,
#         )

#         self.start_ros_time = self.get_clock().now()

#         # state: q
#         self.right_state_samples: List[Tuple[float, np.ndarray]] = []
#         self.left_state_samples: List[Tuple[float, np.ndarray]] = []

#         # torque: tau_ext
#         self.right_torque_samples: List[Tuple[float, np.ndarray]] = []
#         self.left_torque_samples: List[Tuple[float, np.ndarray]] = []

#         self.last_right_state_t = None
#         self.last_left_state_t = None
#         self.last_right_torque_t = None
#         self.last_left_torque_t = None

#         self.warned_short_right_state = False
#         self.warned_short_left_state = False
#         self.warned_short_right_torque = False
#         self.warned_short_left_torque = False

#         self.rows = []

#         # --------------------------------------------------
#         # 軽量Subscriber用QoS
#         # --------------------------------------------------
#         sensor_qos = QoSProfile(
#             history=HistoryPolicy.KEEP_LAST,
#             depth=1,
#             reliability=ReliabilityPolicy.BEST_EFFORT,
#             durability=DurabilityPolicy.VOLATILE,
#         )

#         self.create_subscription(
#             JointState,
#             RIGHT_STATE_TOPIC,
#             self.right_state_callback,
#             sensor_qos,
#         )
#         self.create_subscription(
#             JointState,
#             LEFT_STATE_TOPIC,
#             self.left_state_callback,
#             sensor_qos,
#         )
#         self.create_subscription(
#             JointState,
#             RIGHT_TORQUE_TOPIC,
#             self.right_torque_callback,
#             sensor_qos,
#         )
#         self.create_subscription(
#             JointState,
#             LEFT_TORQUE_TOPIC,
#             self.left_torque_callback,
#             sensor_qos,
#         )

#         self.get_logger().info("Joint-torque-based Z force logger started.")
#         self.get_logger().info(f"right state topic : {RIGHT_STATE_TOPIC}")
#         self.get_logger().info(f"left  state topic : {LEFT_STATE_TOPIC}")
#         self.get_logger().info(f"right torque topic: {RIGHT_TORQUE_TOPIC}")
#         self.get_logger().info(f"left  torque topic: {LEFT_TORQUE_TOPIC}")
#         self.get_logger().info(f"output root: {self.output_root}")
#         self.get_logger().info(f"URDF: {self.estimator.urdf_path}")
#         self.get_logger().info(f"EE frame: {self.estimator.frame_name}")
#         self.get_logger().info(f"Pinocchio nq={self.estimator.model.nq}, nv={self.estimator.model.nv}")
#         self.get_logger().info(f"damping: {self.estimator.damping:.4f}")
#         self.get_logger().info(
#             f"sample/interpolation period: {self.sample_period:.3f} s"
#         )
#         self.get_logger().info(f"z_sign: {self.z_sign}")
#         self.get_logger().info(f"time source: {self.time_source}")
#         self.get_logger().info(
#             f"ratio_min_total_load: {self.ratio_min_total_load:.3f} N"
#         )
#         self.get_logger().info(
#             f"force graph y range: {self.force_ymin:.1f} to {self.force_ymax:.1f} N"
#         )
#         self.get_logger().warn(
#             f"First {self.bias_sec:.1f} sec are used for estimated Z-force bias."
#         )

#     # ------------------------------------------------------
#     # Time utility
#     # ------------------------------------------------------
#     def now_sec(self) -> float:
#         now = self.get_clock().now()
#         return (now - self.start_ros_time).nanoseconds * 1.0e-9

#     def msg_time_sec(self, msg: JointState) -> float:
#         receive_t = self.now_sec()

#         stamp = msg.header.stamp
#         header_t_abs = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9

#         if self.time_source == "receive":
#             return receive_t

#         if self.time_source == "header":
#             return header_t_abs

#         # auto
#         if header_t_abs <= 1.0e-9:
#             return receive_t

#         return header_t_abs

#     @staticmethod
#     def deadband(value: float, threshold: float) -> float:
#         if abs(value) < threshold:
#             return 0.0
#         return value

#     @staticmethod
#     def parse_first_7_position(msg: JointState) -> Optional[np.ndarray]:
#         if len(msg.position) < 7:
#             return None
#         return np.asarray(msg.position[:7], dtype=float)

#     # ------------------------------------------------------
#     # Callbacks
#     # ------------------------------------------------------
#     def right_state_callback(self, msg: JointState):
#         t = self.msg_time_sec(msg)
#         q = self.parse_first_7_position(msg)

#         if q is None:
#             if not self.warned_short_right_state:
#                 self.get_logger().warn(
#                     f"Right state position length is {len(msg.position)} < 7"
#                 )
#                 self.warned_short_right_state = True
#             return

#         if self.last_right_state_t is not None:
#             if (t - self.last_right_state_t) < self.sample_period:
#                 return

#         self.right_state_samples.append((t, q))
#         self.last_right_state_t = t

#     def left_state_callback(self, msg: JointState):
#         t = self.msg_time_sec(msg)
#         q = self.parse_first_7_position(msg)

#         if q is None:
#             if not self.warned_short_left_state:
#                 self.get_logger().warn(
#                     f"Left state position length is {len(msg.position)} < 7"
#                 )
#                 self.warned_short_left_state = True
#             return

#         if self.last_left_state_t is not None:
#             if (t - self.last_left_state_t) < self.sample_period:
#                 return

#         self.left_state_samples.append((t, q))
#         self.last_left_state_t = t

#     def right_torque_callback(self, msg: JointState):
#         t = self.msg_time_sec(msg)
#         tau = self.parse_first_7_position(msg)

#         if tau is None:
#             if not self.warned_short_right_torque:
#                 self.get_logger().warn(
#                     f"Right torque position length is {len(msg.position)} < 7"
#                 )
#                 self.warned_short_right_torque = True
#             return

#         if self.last_right_torque_t is not None:
#             if (t - self.last_right_torque_t) < self.sample_period:
#                 return

#         self.right_torque_samples.append((t, tau))
#         self.last_right_torque_t = t

#     def left_torque_callback(self, msg: JointState):
#         t = self.msg_time_sec(msg)
#         tau = self.parse_first_7_position(msg)

#         if tau is None:
#             if not self.warned_short_left_torque:
#                 self.get_logger().warn(
#                     f"Left torque position length is {len(msg.position)} < 7"
#                 )
#                 self.warned_short_left_torque = True
#             return

#         if self.last_left_torque_t is not None:
#             if (t - self.last_left_torque_t) < self.sample_period:
#                 return

#         self.left_torque_samples.append((t, tau))
#         self.last_left_torque_t = t

#     # ------------------------------------------------------
#     # Build synchronized rows
#     # ------------------------------------------------------
#     def prepare_stream(
#         self,
#         samples: List[Tuple[float, np.ndarray]],
#         label: str,
#     ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
#         if len(samples) < 2:
#             self.get_logger().warn(f"Not enough samples for {label}: {len(samples)}")
#             return None, None

#         t = np.array([s[0] for s in samples], dtype=float)
#         x = np.vstack([s[1] for s in samples]).astype(float)

#         t, x = unique_time_vector(t, x)

#         if len(t) < 2:
#             self.get_logger().warn(f"Not enough unique timestamp samples for {label}.")
#             return None, None

#         return t, x

#     def build_synchronized_rows(self):
#         rs_t_abs, rs_q = self.prepare_stream(self.right_state_samples, "right_state")
#         ls_t_abs, ls_q = self.prepare_stream(self.left_state_samples, "left_state")
#         rt_t_abs, rt_tau = self.prepare_stream(self.right_torque_samples, "right_torque")
#         lt_t_abs, lt_tau = self.prepare_stream(self.left_torque_samples, "left_torque")

#         if (
#             rs_t_abs is None
#             or ls_t_abs is None
#             or rt_t_abs is None
#             or lt_t_abs is None
#         ):
#             return []

#         # 共通の0点
#         t0 = min(rs_t_abs[0], ls_t_abs[0], rt_t_abs[0], lt_t_abs[0])

#         rs_t = rs_t_abs - t0
#         ls_t = ls_t_abs - t0
#         rt_t = rt_t_abs - t0
#         lt_t = lt_t_abs - t0

#         # 4つのstreamすべてが存在する範囲だけ使う
#         start_t = max(rs_t[0], ls_t[0], rt_t[0], lt_t[0])
#         end_t = min(rs_t[-1], ls_t[-1], rt_t[-1], lt_t[-1])

#         if end_t <= start_t:
#             self.get_logger().warn(
#                 f"No overlapping time range. start={start_t:.3f}, end={end_t:.3f}"
#             )
#             return []

#         t_common = np.arange(start_t, end_t, self.sample_period)

#         if len(t_common) < 2:
#             self.get_logger().warn("Common time axis too short.")
#             return []

#         # 各streamを共通時間軸に補間
#         right_q = interp_matrix(rs_t, rs_q, t_common)
#         left_q = interp_matrix(ls_t, ls_q, t_common)
#         right_tau = interp_matrix(rt_t, rt_tau, t_common)
#         left_tau = interp_matrix(lt_t, lt_tau, t_common)

#         right_force_raw = []
#         left_force_raw = []
#         right_residual = []
#         left_residual = []

#         for i in range(len(t_common)):
#             rf, _, rr = self.estimator.estimate_wrench_from_tau(
#                 right_q[i],
#                 right_tau[i],
#             )
#             lf, _, lr = self.estimator.estimate_wrench_from_tau(
#                 left_q[i],
#                 left_tau[i],
#             )

#             right_force_raw.append(rf)
#             left_force_raw.append(lf)
#             right_residual.append(rr)
#             left_residual.append(lr)

#         right_force_raw = np.vstack(right_force_raw)
#         left_force_raw = np.vstack(left_force_raw)
#         right_residual = np.asarray(right_residual, dtype=float)
#         left_residual = np.asarray(left_residual, dtype=float)

#         # 表示上の時刻
#         t_rel = t_common - t_common[0]

#         # バイアス推定
#         bias_mask = t_rel <= self.bias_sec

#         if np.any(bias_mask):
#             right_z_bias = float(np.mean(right_force_raw[bias_mask, 2]))
#             left_z_bias = float(np.mean(left_force_raw[bias_mask, 2]))
#         else:
#             right_z_bias = float(np.mean(right_force_raw[: min(10, len(right_force_raw)), 2]))
#             left_z_bias = float(np.mean(left_force_raw[: min(10, len(left_force_raw)), 2]))

#         self.get_logger().info(
#             "Estimated force bias: "
#             f"right_z_bias={right_z_bias:.3f}, "
#             f"left_z_bias={left_z_bias:.3f}"
#         )

#         rows = []

#         for i, t in enumerate(t_rel):
#             # bias区間はグラフ出力から除外
#             if t < self.bias_sec:
#                 continue

#             right_fx_raw = float(right_force_raw[i, 0])
#             right_fy_raw = float(right_force_raw[i, 1])
#             right_fz_raw = float(right_force_raw[i, 2])

#             left_fx_raw = float(left_force_raw[i, 0])
#             left_fy_raw = float(left_force_raw[i, 1])
#             left_fz_raw = float(left_force_raw[i, 2])

#             right_z_corrected = self.z_sign * (right_fz_raw - right_z_bias)
#             left_z_corrected = self.z_sign * (left_fz_raw - left_z_bias)

#             right_z_corrected = self.deadband(right_z_corrected, self.z_deadband)
#             left_z_corrected = self.deadband(left_z_corrected, self.z_deadband)

#             # 荷重・割合用
#             right_load = max(0.0, right_z_corrected)
#             left_load = max(0.0, left_z_corrected)

#             total_load = right_load + left_load

#             if total_load >= self.ratio_min_total_load:
#                 left_ratio = left_load / total_load
#                 right_ratio = right_load / total_load
#                 ratio_valid = 1
#             else:
#                 left_ratio = np.nan
#                 right_ratio = np.nan
#                 ratio_valid = 0

#             if ratio_valid:
#                 com_position_from_left_ratio = right_ratio
#             else:
#                 com_position_from_left_ratio = np.nan

#             rows.append(
#                 {
#                     "t": float(t - self.bias_sec),

#                     "right_fx_raw_est": right_fx_raw,
#                     "right_fy_raw_est": right_fy_raw,
#                     "right_z_raw": right_fz_raw,

#                     "left_fx_raw_est": left_fx_raw,
#                     "left_fy_raw_est": left_fy_raw,
#                     "left_z_raw": left_fz_raw,

#                     "right_z_bias": right_z_bias,
#                     "left_z_bias": left_z_bias,

#                     "right_z_corrected": right_z_corrected,
#                     "left_z_corrected": left_z_corrected,

#                     "right_load": right_load,
#                     "left_load": left_load,
#                     "total_load": total_load,

#                     "left_ratio": left_ratio,
#                     "right_ratio": right_ratio,
#                     "ratio_valid": ratio_valid,
#                     "com_position_from_left_ratio": com_position_from_left_ratio,

#                     "right_lstsq_residual_norm": float(right_residual[i]),
#                     "left_lstsq_residual_norm": float(left_residual[i]),
#                 }
#             )

#         return rows

#     # ------------------------------------------------------
#     # Save
#     # ------------------------------------------------------
#     def save_outputs(self):
#         self.rows = self.build_synchronized_rows()

#         if len(self.rows) == 0:
#             self.get_logger().warn("No synchronized data recorded.")
#             return

#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_dir = os.path.join(self.output_root, timestamp)
#         os.makedirs(output_dir, exist_ok=True)

#         csv_path = os.path.join(output_dir, "z_force_log.csv")
#         png_path = os.path.join(output_dir, "z_force_and_ratio.png")

#         self.save_csv(csv_path)
#         self.save_plot(png_path)

#         self.get_logger().info(f"Saved CSV: {csv_path}")
#         self.get_logger().info(f"Saved plot: {png_path}")
#         self.get_logger().info(
#             "raw samples: "
#             f"right_state={len(self.right_state_samples)}, "
#             f"left_state={len(self.left_state_samples)}, "
#             f"right_torque={len(self.right_torque_samples)}, "
#             f"left_torque={len(self.left_torque_samples)}"
#         )
#         self.get_logger().info(f"synchronized rows: {len(self.rows)}")

#     def save_csv(self, csv_path: str):
#         fieldnames = list(self.rows[0].keys())

#         with open(csv_path, "w", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()

#             for row in self.rows:
#                 writer.writerow(row)

#     def save_plot(self, png_path: str):
#         import matplotlib

#         matplotlib.use("Agg")

#         import matplotlib.pyplot as plt

#         t = np.array([row["t"] for row in self.rows], dtype=float)

#         left_load = np.array(
#             [row["left_load"] for row in self.rows],
#             dtype=float,
#         )
#         right_load = np.array(
#             [row["right_load"] for row in self.rows],
#             dtype=float,
#         )

#         left_z_corrected = np.array(
#             [row["left_z_corrected"] for row in self.rows],
#             dtype=float,
#         )
#         right_z_corrected = np.array(
#             [row["right_z_corrected"] for row in self.rows],
#             dtype=float,
#         )

#         left_ratio = np.array(
#             [row["left_ratio"] for row in self.rows],
#             dtype=float,
#         )
#         right_ratio = np.array(
#             [row["right_ratio"] for row in self.rows],
#             dtype=float,
#         )
#         ratio_valid = np.array(
#             [row["ratio_valid"] for row in self.rows],
#             dtype=bool,
#         )

#         left_ratio_percent = 100.0 * left_ratio
#         right_ratio_percent = 100.0 * right_ratio

#         fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

#         # ------------------------------
#         # 1. Left / Right estimated Z load
#         # ------------------------------
#         axes[0].plot(t, left_load, label="Left estimated Z load")
#         axes[0].plot(t, right_load, label="Right estimated Z load")

#         # 補正後の符号付き値も薄い破線で残す
#         axes[0].plot(
#             t,
#             left_z_corrected,
#             label="Left estimated Z corrected",
#             linestyle="--",
#             alpha=0.45,
#         )
#         axes[0].plot(
#             t,
#             right_z_corrected,
#             label="Right estimated Z corrected",
#             linestyle="--",
#             alpha=0.45,
#         )

#         axes[0].axhline(0.0, linestyle="--", linewidth=1.0)
#         axes[0].set_ylabel("Force [N]")
#         axes[0].set_title("Estimated left and right FR3 Z-direction load")
#         axes[0].set_ylim(self.force_ymin, self.force_ymax)
#         axes[0].grid(True)
#         axes[0].legend(loc="upper right")

#         # ------------------------------
#         # 2. Left / Right load sharing ratio
#         # ------------------------------
#         axes[1].fill_between(
#             t,
#             0.0,
#             left_ratio_percent,
#             where=ratio_valid,
#             alpha=0.65,
#             label="Left load ratio",
#         )
#         axes[1].fill_between(
#             t,
#             left_ratio_percent,
#             100.0,
#             where=ratio_valid,
#             alpha=0.65,
#             label="Right load ratio",
#         )

#         # 左右割合の境界線
#         axes[1].plot(
#             t,
#             left_ratio_percent,
#             linestyle="--",
#             linewidth=2.0,
#             label="Left / Right boundary",
#         )

#         # 50%基準線
#         axes[1].axhline(
#             50.0,
#             linestyle=":",
#             linewidth=1.5,
#             label="Balanced 50%",
#         )

#         axes[1].set_xlabel("Time [s]")
#         axes[1].set_ylabel("Load ratio [%]")
#         axes[1].set_title("Estimated left and right load sharing ratio")
#         axes[1].set_ylim(0.0, 100.0)
#         axes[1].grid(True)
#         axes[1].legend(loc="upper right")

#         invalid_count = int(np.sum(~ratio_valid))
#         if invalid_count > 0:
#             axes[1].text(
#                 0.01,
#                 0.02,
#                 f"ratio hidden when total load < {self.ratio_min_total_load:.2f} N",
#                 transform=axes[1].transAxes,
#                 fontsize=9,
#             )

#         fig.tight_layout()
#         fig.savefig(png_path, dpi=200)
#         plt.close(fig)


# def main():
#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--output-root",
#         type=str,
#         default=DEFAULT_OUTPUT_ROOT,
#         help="Directory where CSV and PNG are saved.",
#     )

#     parser.add_argument(
#         "--sample-period",
#         type=float,
#         default=DEFAULT_SAMPLE_PERIOD,
#         help="Sampling / interpolation period [s]. Default 0.05 = 20 Hz.",
#     )

#     parser.add_argument(
#         "--bias-sec",
#         type=float,
#         default=2.0,
#         help="Initial seconds for estimated Z-force bias.",
#     )

#     parser.add_argument(
#         "--z-sign",
#         type=float,
#         default=DEFAULT_Z_SIGN,
#         help="Use 1.0 or -1.0. If estimated Z force direction is wrong, flip this.",
#     )

#     parser.add_argument(
#         "--z-deadband",
#         type=float,
#         default=0.2,
#         help="Deadband for estimated Z force [N].",
#     )

#     parser.add_argument(
#         "--ratio-min-total-load",
#         type=float,
#         default=DEFAULT_RATIO_MIN_TOTAL_LOAD,
#         help="Minimum total load [N] to calculate load ratio.",
#     )

#     parser.add_argument(
#         "--force-ymin",
#         type=float,
#         default=DEFAULT_FORCE_YMIN,
#         help="Y-axis minimum force [N] for the first graph.",
#     )

#     parser.add_argument(
#         "--force-ymax",
#         type=float,
#         default=DEFAULT_FORCE_YMAX,
#         help="Y-axis maximum force [N] for the first graph.",
#     )

#     parser.add_argument(
#         "--time-source",
#         type=str,
#         default="auto",
#         choices=["auto", "header", "receive"],
#         help="Time source for samples. auto uses header stamp if valid, otherwise receive time.",
#     )

#     parser.add_argument(
#         "--urdf",
#         type=str,
#         default="auto",
#         help="Path to FR3 URDF. Use 'auto' to search under ~/franka_ros2_ws.",
#     )

#     parser.add_argument(
#         "--ee-frame",
#         type=str,
#         default="auto",
#         help="End-effector frame name in URDF. Use 'auto' for automatic selection.",
#     )

#     parser.add_argument(
#         "--damping",
#         type=float,
#         default=DEFAULT_DAMPING,
#         help="Damping for solving J^T F = tau. Set 0.0 for ordinary least squares.",
#     )

#     args = parser.parse_args()

#     rclpy.init()

#     node = ZForceLogger(args)

#     try:
#         rclpy.spin(node)

#     except KeyboardInterrupt:
#         node.get_logger().info("Ctrl+C received. Saving outputs...")

#     finally:
#         try:
#             node.save_outputs()
#         except Exception as e:
#             node.get_logger().error(f"Failed to save outputs: {e}")

#         try:
#             node.destroy_node()
#         except Exception:
#             pass

#         if rclpy.ok():
#             rclpy.shutdown()


# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# """
# record_z_force_bias_graph.py

# 目的:
#   左右FR3のZ方向外力を記録し，
#   左右それぞれが全荷重の何割を支えているかをグラフ化する。

# 重要:
#   左右のlatest値をtimerで同時刻扱いしない。
#   左右それぞれのWrenchStampedをtimestamp付きで保存し，
#   終了時に共通時間軸へ補間してから割合を計算する。

# 今回の表示:
#   1段目:
#     Left Z load と Right Z load を同じグラフに重ねて描画
#     縦軸は -5〜10 N に固定

#   2段目:
#     Left / Right load sharing ratio
#     100%積み上げ表示
#     境界線を点線で表示

# 出力:
#   ~/franka_ros2_ws/bilateral_logs_z/YYYYmmdd_HHMMSS/
#     z_force_log.csv
#     z_force_and_ratio.png

# 実行:
#   cd ~/franka_ros2_ws
#   source /opt/ros/humble/setup.bash
#   source install/setup.bash

#   nice -n 10 python3 ~/franka_ros2_ws/test_scripts/record_z_force_bias_graph.py

# 終了:
#   Ctrl+C
# """

# import os
# import csv
# import argparse
# from datetime import datetime
# from typing import List, Tuple

# import numpy as np

# import rclpy
# from rclpy.node import Node
# from rclpy.qos import (
#     QoSProfile,
#     ReliabilityPolicy,
#     HistoryPolicy,
#     DurabilityPolicy,
# )
# from geometry_msgs.msg import WrenchStamped


# # ==========================================================
# # Topic settings
# # ==========================================================
# RIGHT_WRENCH_TOPIC = "/franka/right/ee_wrench_world"
# LEFT_WRENCH_TOPIC = "/franka/left/ee_wrench_world"


# # ==========================================================
# # Default settings
# # ==========================================================
# DEFAULT_OUTPUT_ROOT = os.path.expanduser("~/franka_ros2_ws/bilateral_logs_z")

# # 実機でZ方向が逆なら -1.0 にする
# DEFAULT_Z_SIGN = 1.0

# # callbackで受け取った全データを保存せず，この間隔以上空いたときだけ保存する
# DEFAULT_SAMPLE_PERIOD = 0.05

# # 割合計算するときの最小合計荷重 [N]
# # total_load が小さいと割合が暴れるので，それ未満は割合グラフを空白にする
# DEFAULT_RATIO_MIN_TOTAL_LOAD = 0.5

# # 1段目のZ荷重グラフの最小値 [N]
# DEFAULT_FORCE_YMIN = -5.0

# # 1段目のZ荷重グラフの最大値 [N]
# DEFAULT_FORCE_YMAX = 10.0


# class ZForceLogger(Node):
#     def __init__(self, args):
#         super().__init__("z_force_logger")

#         self.output_root = os.path.expanduser(args.output_root)
#         self.sample_period = float(args.sample_period)
#         self.bias_sec = float(args.bias_sec)
#         self.z_sign = float(args.z_sign)
#         self.z_deadband = float(args.z_deadband)
#         self.ratio_min_total_load = float(args.ratio_min_total_load)
#         self.force_ymin = float(args.force_ymin)
#         self.force_ymax = float(args.force_ymax)
#         self.time_source = str(args.time_source)

#         self.start_ros_time = self.get_clock().now()

#         # 左右は別々にtimestamp付きで保存する
#         self.right_samples: List[Tuple[float, float]] = []
#         self.left_samples: List[Tuple[float, float]] = []

#         self.last_right_store_t = None
#         self.last_left_store_t = None

#         self.rows = []

#         # --------------------------------------------------
#         # 軽量Subscriber用QoS
#         # --------------------------------------------------
#         sensor_qos = QoSProfile(
#             history=HistoryPolicy.KEEP_LAST,
#             depth=1,
#             reliability=ReliabilityPolicy.BEST_EFFORT,
#             durability=DurabilityPolicy.VOLATILE,
#         )

#         self.create_subscription(
#             WrenchStamped,
#             RIGHT_WRENCH_TOPIC,
#             self.right_wrench_callback,
#             sensor_qos,
#         )

#         self.create_subscription(
#             WrenchStamped,
#             LEFT_WRENCH_TOPIC,
#             self.left_wrench_callback,
#             sensor_qos,
#         )

#         self.get_logger().info("Z force logger started.")
#         self.get_logger().info(f"right wrench topic: {RIGHT_WRENCH_TOPIC}")
#         self.get_logger().info(f"left  wrench topic: {LEFT_WRENCH_TOPIC}")
#         self.get_logger().info(f"output root: {self.output_root}")
#         self.get_logger().info(
#             f"sample/interpolation period: {self.sample_period:.3f} s"
#         )
#         self.get_logger().info(f"z_sign: {self.z_sign}")
#         self.get_logger().info(f"time source: {self.time_source}")
#         self.get_logger().info(
#             f"ratio_min_total_load: {self.ratio_min_total_load:.3f} N"
#         )
#         self.get_logger().info(
#             f"force graph y range: {self.force_ymin:.1f} to {self.force_ymax:.1f} N"
#         )
#         self.get_logger().warn(
#             f"First {self.bias_sec:.1f} sec are used for Z-force bias estimation."
#         )
#         self.get_logger().warn(
#             "This logger stores left/right wrench samples separately and synchronizes them offline."
#         )

#     # ------------------------------------------------------
#     # Time utility
#     # ------------------------------------------------------
#     def now_sec(self) -> float:
#         now = self.get_clock().now()
#         return (now - self.start_ros_time).nanoseconds * 1.0e-9

#     def msg_time_sec(self, msg: WrenchStamped) -> float:
#         """
#         使用する時刻を返す。

#         time_source:
#           receive:
#             callback到着時刻を使う
#           header:
#             msg.header.stampを使う
#           auto:
#             header.stampが有効そうならheader，ゼロならreceive
#         """

#         receive_t = self.now_sec()

#         stamp = msg.header.stamp
#         header_t_abs = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9

#         if self.time_source == "receive":
#             return receive_t

#         if self.time_source == "header":
#             return header_t_abs

#         # auto
#         # header stampが0ならreceive timeを使う
#         if header_t_abs <= 1.0e-9:
#             return receive_t

#         return header_t_abs

#     @staticmethod
#     def deadband(value: float, threshold: float) -> float:
#         if abs(value) < threshold:
#             return 0.0
#         return value

#     # ------------------------------------------------------
#     # Callbacks
#     # ------------------------------------------------------
#     def right_wrench_callback(self, msg: WrenchStamped):
#         t = self.msg_time_sec(msg)
#         z = float(msg.wrench.force.z)

#         if self.last_right_store_t is not None:
#             if (t - self.last_right_store_t) < self.sample_period:
#                 return

#         self.right_samples.append((t, z))
#         self.last_right_store_t = t

#     def left_wrench_callback(self, msg: WrenchStamped):
#         t = self.msg_time_sec(msg)
#         z = float(msg.wrench.force.z)

#         if self.last_left_store_t is not None:
#             if (t - self.last_left_store_t) < self.sample_period:
#                 return

#         self.left_samples.append((t, z))
#         self.last_left_store_t = t

#     # ------------------------------------------------------
#     # Data processing
#     # ------------------------------------------------------
#     def build_synchronized_rows(self):
#         if len(self.right_samples) < 2 or len(self.left_samples) < 2:
#             self.get_logger().warn(
#                 "Not enough samples. "
#                 f"right={len(self.right_samples)}, left={len(self.left_samples)}"
#             )
#             return []

#         right_t_abs = np.array([x[0] for x in self.right_samples], dtype=float)
#         right_raw_abs = np.array([x[1] for x in self.right_samples], dtype=float)

#         left_t_abs = np.array([x[0] for x in self.left_samples], dtype=float)
#         left_raw_abs = np.array([x[1] for x in self.left_samples], dtype=float)

#         # header時刻を使う場合，絶対時刻が大きいので共通の0点を作る
#         t0 = min(right_t_abs[0], left_t_abs[0])

#         right_t = right_t_abs - t0
#         left_t = left_t_abs - t0

#         # 念のため時刻順に並べる
#         right_order = np.argsort(right_t)
#         left_order = np.argsort(left_t)

#         right_t = right_t[right_order]
#         right_raw_abs = right_raw_abs[right_order]

#         left_t = left_t[left_order]
#         left_raw_abs = left_raw_abs[left_order]

#         # 重複時刻があるとnp.interpが不安定なので取り除く
#         right_t, right_unique_idx = np.unique(right_t, return_index=True)
#         right_raw_abs = right_raw_abs[right_unique_idx]

#         left_t, left_unique_idx = np.unique(left_t, return_index=True)
#         left_raw_abs = left_raw_abs[left_unique_idx]

#         if len(right_t) < 2 or len(left_t) < 2:
#             self.get_logger().warn("Not enough unique timestamp samples.")
#             return []

#         # バイアス推定
#         right_bias_mask = right_t <= self.bias_sec
#         left_bias_mask = left_t <= self.bias_sec

#         if np.any(right_bias_mask):
#             right_z_bias = float(np.mean(right_raw_abs[right_bias_mask]))
#         else:
#             right_z_bias = float(np.mean(right_raw_abs[: min(10, len(right_raw_abs))]))

#         if np.any(left_bias_mask):
#             left_z_bias = float(np.mean(left_raw_abs[left_bias_mask]))
#         else:
#             left_z_bias = float(np.mean(left_raw_abs[: min(10, len(left_raw_abs))]))

#         self.get_logger().info(
#             "Bias estimated: "
#             f"right_z_bias={right_z_bias:.3f}, "
#             f"left_z_bias={left_z_bias:.3f}"
#         )

#         # 左右が両方存在する時間範囲だけ使う
#         start_t = max(right_t[0], left_t[0], self.bias_sec)
#         end_t = min(right_t[-1], left_t[-1])

#         if end_t <= start_t:
#             self.get_logger().warn(
#                 f"No overlapping time range. start={start_t:.3f}, end={end_t:.3f}"
#             )
#             return []

#         # 共通時間軸
#         t_common = np.arange(start_t, end_t, self.sample_period)

#         if len(t_common) < 2:
#             self.get_logger().warn("Common time axis too short.")
#             return []

#         # 左右を共通時間軸へ補間
#         right_z_raw = np.interp(t_common, right_t, right_raw_abs)
#         left_z_raw = np.interp(t_common, left_t, left_raw_abs)

#         rows = []

#         for i, t in enumerate(t_common):
#             rz_raw = float(right_z_raw[i])
#             lz_raw = float(left_z_raw[i])

#             right_z_corrected = self.z_sign * (rz_raw - right_z_bias)
#             left_z_corrected = self.z_sign * (lz_raw - left_z_bias)

#             right_z_corrected = self.deadband(right_z_corrected, self.z_deadband)
#             left_z_corrected = self.deadband(left_z_corrected, self.z_deadband)

#             # 荷重・割合用
#             right_load = max(0.0, right_z_corrected)
#             left_load = max(0.0, left_z_corrected)

#             total_load = right_load + left_load

#             if total_load >= self.ratio_min_total_load:
#                 left_ratio = left_load / total_load
#                 right_ratio = right_load / total_load
#                 ratio_valid = 1
#             else:
#                 left_ratio = np.nan
#                 right_ratio = np.nan
#                 ratio_valid = 0

#             if ratio_valid:
#                 com_position_from_left_ratio = right_ratio
#             else:
#                 com_position_from_left_ratio = np.nan

#             rows.append(
#                 {
#                     "t": float(t - start_t),
#                     "right_z_raw": rz_raw,
#                     "left_z_raw": lz_raw,
#                     "right_z_bias": right_z_bias,
#                     "left_z_bias": left_z_bias,
#                     "right_z_corrected": right_z_corrected,
#                     "left_z_corrected": left_z_corrected,
#                     "right_load": right_load,
#                     "left_load": left_load,
#                     "total_load": total_load,
#                     "left_ratio": left_ratio,
#                     "right_ratio": right_ratio,
#                     "ratio_valid": ratio_valid,
#                     "com_position_from_left_ratio": com_position_from_left_ratio,
#                 }
#             )

#         return rows

#     # ------------------------------------------------------
#     # Save
#     # ------------------------------------------------------
#     def save_outputs(self):
#         self.rows = self.build_synchronized_rows()

#         if len(self.rows) == 0:
#             self.get_logger().warn("No synchronized data recorded.")
#             return

#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_dir = os.path.join(self.output_root, timestamp)
#         os.makedirs(output_dir, exist_ok=True)

#         csv_path = os.path.join(output_dir, "z_force_log.csv")
#         png_path = os.path.join(output_dir, "z_force_and_ratio.png")

#         self.save_csv(csv_path)
#         self.save_plot(png_path)

#         self.get_logger().info(f"Saved CSV: {csv_path}")
#         self.get_logger().info(f"Saved plot: {png_path}")
#         self.get_logger().info(
#             f"raw samples: right={len(self.right_samples)}, left={len(self.left_samples)}"
#         )
#         self.get_logger().info(f"synchronized rows: {len(self.rows)}")

#     def save_csv(self, csv_path: str):
#         fieldnames = list(self.rows[0].keys())

#         with open(csv_path, "w", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()

#             for row in self.rows:
#                 writer.writerow(row)

#     def save_plot(self, png_path: str):
#         import matplotlib

#         matplotlib.use("Agg")

#         import matplotlib.pyplot as plt

#         t = np.array([row["t"] for row in self.rows], dtype=float)

#         left_load = np.array(
#             [row["left_load"] for row in self.rows],
#             dtype=float,
#         )
#         right_load = np.array(
#             [row["right_load"] for row in self.rows],
#             dtype=float,
#         )

#         left_z_corrected = np.array(
#             [row["left_z_corrected"] for row in self.rows],
#             dtype=float,
#         )
#         right_z_corrected = np.array(
#             [row["right_z_corrected"] for row in self.rows],
#             dtype=float,
#         )

#         left_ratio = np.array(
#             [row["left_ratio"] for row in self.rows],
#             dtype=float,
#         )
#         right_ratio = np.array(
#             [row["right_ratio"] for row in self.rows],
#             dtype=float,
#         )
#         ratio_valid = np.array(
#             [row["ratio_valid"] for row in self.rows],
#             dtype=bool,
#         )

#         left_ratio_percent = 100.0 * left_ratio
#         right_ratio_percent = 100.0 * right_ratio

#         fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

#         # ------------------------------
#         # 1. Left / Right Z load
#         # ------------------------------
#         axes[0].plot(t, left_load, label="Left Z load")
#         axes[0].plot(t, right_load, label="Right Z load")

#         # 補正後の符号付き値も薄い破線で残す
#         axes[0].plot(
#             t,
#             left_z_corrected,
#             label="Left Z corrected",
#             linestyle="--",
#             alpha=0.45,
#         )
#         axes[0].plot(
#             t,
#             right_z_corrected,
#             label="Right Z corrected",
#             linestyle="--",
#             alpha=0.45,
#         )

#         axes[0].axhline(0.0, linestyle="--", linewidth=1.0)
#         axes[0].set_ylabel("Force [N]")
#         axes[0].set_title("Left and right FR3 Z-direction load")
#         axes[0].set_ylim(self.force_ymin, self.force_ymax)
#         axes[0].grid(True)
#         axes[0].legend(loc="upper right")

#         # ------------------------------
#         # 2. Left / Right load sharing ratio
#         # ------------------------------
#         # total_loadが小さいところは割合として意味が薄いので空白にする
#         axes[1].fill_between(
#             t,
#             0.0,
#             left_ratio_percent,
#             where=ratio_valid,
#             alpha=0.65,
#             label="Left load ratio",
#         )
#         axes[1].fill_between(
#             t,
#             left_ratio_percent,
#             100.0,
#             where=ratio_valid,
#             alpha=0.65,
#             label="Right load ratio",
#         )

#         # 左右割合の境界線
#         # この点線より下がLeft，点線より上がRight
#         axes[1].plot(
#             t,
#             left_ratio_percent,
#             linestyle="--",
#             linewidth=2.0,
#             label="Left / Right boundary",
#         )

#         # 50%基準線
#         axes[1].axhline(
#             50.0,
#             linestyle=":",
#             linewidth=1.5,
#             label="Balanced 50%",
#         )

#         axes[1].set_xlabel("Time [s]")
#         axes[1].set_ylabel("Load ratio [%]")
#         axes[1].set_title("Left and right load sharing ratio")
#         axes[1].set_ylim(0.0, 100.0)
#         axes[1].grid(True)
#         axes[1].legend(loc="upper right")

#         # ratioが無効なところの説明を小さく入れる
#         invalid_count = int(np.sum(~ratio_valid))
#         if invalid_count > 0:
#             axes[1].text(
#                 0.01,
#                 0.02,
#                 f"ratio hidden when total load < {self.ratio_min_total_load:.2f} N",
#                 transform=axes[1].transAxes,
#                 fontsize=9,
#             )

#         fig.tight_layout()
#         fig.savefig(png_path, dpi=200)
#         plt.close(fig)


# def main():
#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--output-root",
#         type=str,
#         default=DEFAULT_OUTPUT_ROOT,
#         help="Directory where CSV and PNG are saved.",
#     )

#     parser.add_argument(
#         "--sample-period",
#         type=float,
#         default=DEFAULT_SAMPLE_PERIOD,
#         help="Sampling / interpolation period [s]. Default 0.05 = 20 Hz.",
#     )

#     parser.add_argument(
#         "--bias-sec",
#         type=float,
#         default=2.0,
#         help="Initial seconds for bias estimation.",
#     )

#     parser.add_argument(
#         "--z-sign",
#         type=float,
#         default=DEFAULT_Z_SIGN,
#         help="Use 1.0 or -1.0. If Z force direction is wrong, flip this.",
#     )

#     parser.add_argument(
#         "--z-deadband",
#         type=float,
#         default=0.2,
#         help="Deadband for Z force [N].",
#     )

#     parser.add_argument(
#         "--ratio-min-total-load",
#         type=float,
#         default=DEFAULT_RATIO_MIN_TOTAL_LOAD,
#         help="Minimum total load [N] to calculate load ratio.",
#     )

#     parser.add_argument(
#         "--force-ymin",
#         type=float,
#         default=DEFAULT_FORCE_YMIN,
#         help="Y-axis minimum force [N] for the first graph.",
#     )

#     parser.add_argument(
#         "--force-ymax",
#         type=float,
#         default=DEFAULT_FORCE_YMAX,
#         help="Y-axis maximum force [N] for the first graph.",
#     )

#     parser.add_argument(
#         "--time-source",
#         type=str,
#         default="auto",
#         choices=["auto", "header", "receive"],
#         help="Time source for samples. auto uses header stamp if valid, otherwise receive time.",
#     )

#     args = parser.parse_args()

#     rclpy.init()

#     node = ZForceLogger(args)

#     try:
#         rclpy.spin(node)

#     except KeyboardInterrupt:
#         node.get_logger().info("Ctrl+C received. Saving outputs...")

#     finally:
#         try:
#             node.save_outputs()
#         except Exception as e:
#             node.get_logger().error(f"Failed to save outputs: {e}")

#         try:
#             node.destroy_node()
#         except Exception:
#             pass

#         if rclpy.ok():
#             rclpy.shutdown()


# if __name__ == "__main__":
#     main()