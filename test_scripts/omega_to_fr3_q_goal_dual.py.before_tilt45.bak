#!/usr/bin/env python3

import argparse
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped

import pinocchio as pin

from ohrc_msgs.msg import State


# ===========================
# Topic設定
# ===========================
RIGHT_OMEGA_TOPIC = "/right/state"
LEFT_OMEGA_TOPIC = "/left/state"

RIGHT_Q_GOAL_TOPIC = "/omega_fr3/right/q_goal"
LEFT_Q_GOAL_TOPIC = "/omega_fr3/left/q_goal"


# ===========================
# Timer / publish周期
# ===========================
CONTROL_DT = 0.001  # 1000 Hz


# ===========================
# 関節速度制限
# ===========================
ENABLE_JOINT_VELOCITY_LIMIT = False
JOINT_VEL_LIMIT_RAD_S = 1.0


# ===========================
# Omega本体操作スケール
# ===========================
OMEGA_POS_SCALE = 4.0
OMEGA_ROT_SCALE = 1.0


# ===========================
# 双腕ベース配置
# ===========================

LEFT_FR3_BASE_POS_WORLD = np.array([
    0.0,
    0.0,
    0.0,
])

RIGHT_FR3_BASE_POS_WORLD = np.array([
    0.0,
    1.0,
    0.0,
])


# ===========================
# 右Omegaグリッパによる双腕把持設定
# ===========================
ENABLE_GRIP_POSITION_CONTROL = True

GRIP_POS_SCALE = 2.0

# フォールバック用固定軸。
# 通常は GRASP_ASSIST_AXIS_MODE = "between_ee_world" により，
# 左右手先を結ぶ3次元方向を使う。
GRIP_AXIS = np.array([
    0.0,
    1.0,
    0.0,
])


# ===========================
# 右Omega gripper.buttonによる把持後モード
# ===========================
ENABLE_LIFTING_MODE = True

# False:
#   把持開始時の左右手先姿勢と位置関係を固定し，並進のみ行う。
LIFTING_MODE_USE_RIGHT_OMEGA_ROT = False


# ===========================
# 把持アシスト設定
# ===========================
ENABLE_GRASP_ASSIST = True

RIGHT_EE_WRENCH_TOPIC = "/franka/right/ee_wrench_world"
LEFT_EE_WRENCH_TOPIC = "/franka/left/ee_wrench_world"

FRANKA_FORCE_SIGN = -1.0

GRASP_FORCE_MARGIN_N = 0.3
GRASP_FORCE_LPF_ALPHA = 0.05
GRASP_ASSIST_KI = 0.0006
GRASP_ASSIST_MAX_STEP = 0.000005
GRASP_ASSIST_MAX_OFFSET = 0.005
GRASP_FORCE_DEADBAND_N = 0.2

# "between_ee_world":
#   左右FR3の手先を共通world座標に変換して，
#   left EE -> right EE の3次元単位ベクトルを把持方向にする。
#
# "grip_axis":
#   従来通り GRIP_AXIS を使う。
GRASP_ASSIST_AXIS_MODE = "between_ee_world"


# ===========================
# 左Omegaグリッパによる左右腕中間点制御
# ===========================
ENABLE_RIGHT_MIDPOINT_CONTROL = True
ENABLE_LEFT_MIDPOINT_CONTROL = True

MID_FRAME_NAME = "fr3_link5"

MIDPOINT_AXIS = np.array([
    0.0,
    1.0,
    0.0,
])

FINGER_TO_MID_SCALE = 0.5
MIDPOINT_TARGET_LIMIT = 0.5

K_MID = 0.8
MID_ERR_MAX = 0.05

# 左グリッパの小さな入力ノイズは0扱いにする
MID_GRIPPER_DEADBAND = 0.001


# ===========================
# IK設定
# ===========================
URDF_PATH = "/home/watanaberyuto/franka_ros2_ws/src/fr3_urdf/fr3.urdf"


# ===========================
# DLS設定
# ===========================
ENABLE_ADAPTIVE_DLS = True

PINV_DAMPING = 0.02
DLS_DAMPING_MAX = 0.20
DLS_SIGMA_THRESHOLD = 0.08
DLS_SVD_EPS = 1e-10

MAX_DQ_NORM = 0.25

EE_TASK_MODE = "pose"

LOG_INTERVAL = 10


# ==========================================================
# 右Omega用：Omega位置座標 → FR3位置座標
# ==========================================================
RIGHT_OMEGA_TO_FR3_POS = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])


# ==========================================================
# 左Omega用：Omega位置座標 → FR3位置座標
# ==========================================================
LEFT_OMEGA_TO_FR3_POS = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])


# ==========================================================
# 右Omega用：Omega回転軸 → FR3回転軸
# ==========================================================
RIGHT_OMEGA_TO_FR3_ROT = np.array([
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])


# ==========================================================
# 左Omega用：Omega回転軸 → FR3回転軸
# ==========================================================
LEFT_OMEGA_TO_FR3_ROT = np.array([
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])


# ==========================================================
# 右Omega用：回転方向の符号
# ==========================================================
RIGHT_OMEGA_ROT_SIGN = np.array([
   -1.0,
   -1.0,
   -1.0,
])


# ==========================================================
# 左Omega用：回転方向の符号
# ==========================================================
LEFT_OMEGA_ROT_SIGN = np.array([
   -1.0,
   -1.0,
   -1.0,
])


class OmegaState:
    def __init__(self):
        self.initialized = False

        self.initial_pos = np.zeros(3)
        self.pos = np.zeros(3)

        self.initial_rot = np.eye(3)
        self.rot = np.eye(3)

        self.gripper_initialized = False
        self.gripper_initial = 0.0
        self.gripper_value = 0.0
        self.gripper_delta = 0.0

        self.gripper_button = False


class ArmIK:
    def __init__(
        self,
        node,
        arm_name,
        q_goal_pub,
        enable_midpoint_control=False,
    ):
        self.node = node
        self.arm_name = arm_name
        self.q_goal_pub = q_goal_pub
        self.enable_midpoint_control = enable_midpoint_control

        self.joint_names = [
            "fr3_joint1",
            "fr3_joint2",
            "fr3_joint3",
            "fr3_joint4",
            "fr3_joint5",
            "fr3_joint6",
            "fr3_joint7",
        ]

        self.model = pin.buildModelFromUrdf(URDF_PATH)
        self.data = self.model.createData()

        self.frame_name = "fr3_hand_tcp"

        if not self.model.existFrame(self.frame_name):
            self.frame_name = "fr3_link8"

        if not self.model.existFrame(self.frame_name):
            raise ValueError(
                f"{self.arm_name}: End-effector frame not found"
            )

        self.frame_id = self.model.getFrameId(self.frame_name)

        self.mid_frame_name = MID_FRAME_NAME
        self.mid_frame_id = None

        if self.enable_midpoint_control:
            if not self.model.existFrame(self.mid_frame_name):
                self.node.get_logger().error(
                    f"{self.arm_name}: Middle frame not found: {self.mid_frame_name}"
                )

                for i, frame in enumerate(self.model.frames):
                    self.node.get_logger().error(f"{i}: {frame.name}")

                raise ValueError(
                    f"{self.arm_name}: Middle frame not found"
                )

            self.mid_frame_id = self.model.getFrameId(self.mid_frame_name)

        self.q = np.array([
            0.0,
            -0.7,
            0.0,
            -2.2,
            0.0,
            1.6,
            0.4,
            0.02,
            0.02,
        ])

        self.active_idx = list(range(7))

        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

        initial_se3 = self.data.oMf[self.frame_id]

        self.fr3_initial_pos = initial_se3.translation.copy()
        self.fr3_initial_rot = initial_se3.rotation.copy()

        self.target_pos = self.fr3_initial_pos.copy()
        self.target_rot = self.fr3_initial_rot.copy()

        self.mid_initial_pos = None
        self.mid_target_pos = None

        # 中間リンク追加制御用
        self.mid_task_enabled_now = False
        self.mid_task_enabled_prev = False

        self.mid_activation_pos = None
        self.mid_activation_gripper_delta = 0.0
        self.mid_activation_delta_fr3 = np.zeros(3)

        # ===========================
        # 中間リンク常時ON制御用の基準
        # ===========================
        self.mid_reference_pos = None
        self.mid_reference_delta_fr3 = np.zeros(3)
        self.mid_reference_gripper_delta = 0.0

        self.mid_base_pos = None
        self.mid_signed_axis = np.array([0.0, 1.0, 0.0])
        self.mid_move_amount = 0.0

        if self.enable_midpoint_control:
            mid_initial_se3 = self.data.oMf[self.mid_frame_id]
            self.mid_initial_pos = mid_initial_se3.translation.copy()
            self.mid_target_pos = self.mid_initial_pos.copy()
            self.mid_base_pos = self.mid_initial_pos.copy()
            self.mid_activation_pos = self.mid_initial_pos.copy()

            self.mid_reference_pos = self.mid_initial_pos.copy()
            self.mid_reference_delta_fr3 = np.zeros(3)
            self.mid_reference_gripper_delta = 0.0

            # 中間リンク制御がTrueなら，起動時からタスクは常時ON
            self.mid_task_enabled_now = True
            self.mid_task_enabled_prev = True

        self.prev_published_q = self.q[:7].copy()

        # DLSログ用
        self.last_ee_sigma_min = None
        self.last_ee_damping = None
        self.last_mid_sigma_min = None
        self.last_mid_damping = None

        self.node.get_logger().info(
            f"{self.arm_name}: IK initialized, EE frame={self.frame_name}"
        )

        if self.enable_midpoint_control:
            self.node.get_logger().info(
                f"{self.arm_name}: midpoint control ENABLED, "
                f"mid_frame={self.mid_frame_name}, "
                f"mid_initial_pos={self.mid_initial_pos}"
            )
        else:
            self.node.get_logger().info(
                f"{self.arm_name}: midpoint control DISABLED"
            )

    def get_axis_mapping(self):
        if self.arm_name == "right":
            return (
                RIGHT_OMEGA_TO_FR3_POS,
                RIGHT_OMEGA_TO_FR3_ROT,
                RIGHT_OMEGA_ROT_SIGN,
            )

        if self.arm_name == "left":
            return (
                LEFT_OMEGA_TO_FR3_POS,
                LEFT_OMEGA_TO_FR3_ROT,
                LEFT_OMEGA_ROT_SIGN,
            )

        return (
            np.eye(3),
            np.eye(3),
            np.ones(3),
        )

    def compute_current_delta_fr3(self, omega_state):
        if not omega_state.initialized:
            return np.zeros(3)

        omega_to_fr3_pos, _, _ = self.get_axis_mapping()

        delta_omega = omega_state.pos - omega_state.initial_pos
        delta_fr3 = omega_to_fr3_pos @ delta_omega

        return delta_fr3

    def compute_adaptive_damping(self, sigma_min):
        if not ENABLE_ADAPTIVE_DLS:
            return PINV_DAMPING

        sigma_min = float(max(sigma_min, 0.0))

        if sigma_min >= DLS_SIGMA_THRESHOLD:
            return PINV_DAMPING

        ratio = sigma_min / DLS_SIGMA_THRESHOLD
        ratio = float(np.clip(ratio, 0.0, 1.0))

        damping_sq = (
            PINV_DAMPING * PINV_DAMPING
            + (
                DLS_DAMPING_MAX * DLS_DAMPING_MAX
                - PINV_DAMPING * PINV_DAMPING
            )
            * (1.0 - ratio * ratio)
        )

        return float(np.sqrt(damping_sq))

    def damped_pinv(self, J, label=None):
        if J.size == 0:
            return np.zeros((J.shape[1], J.shape[0]))

        U, S, Vt = np.linalg.svd(J, full_matrices=False)

        if S.size == 0:
            sigma_min = 0.0
        else:
            sigma_min = float(np.min(S))

        damping = self.compute_adaptive_damping(sigma_min)

        gains = np.zeros_like(S)

        for i, sigma in enumerate(S):
            if sigma < DLS_SVD_EPS and damping < DLS_SVD_EPS:
                gains[i] = 0.0
            else:
                gains[i] = sigma / (sigma * sigma + damping * damping)

        J_pinv = Vt.T @ np.diag(gains) @ U.T

        if label == "ee":
            self.last_ee_sigma_min = sigma_min
            self.last_ee_damping = damping
        elif label == "mid":
            self.last_mid_sigma_min = sigma_min
            self.last_mid_damping = damping

        return J_pinv

    def clip_norm(self, v, max_norm):
        norm = np.linalg.norm(v)

        if norm > max_norm:
            return v / norm * max_norm

        return v

    def apply_joint_velocity_limit(self, q_desired):
        q_desired_7 = q_desired[:7].copy()

        if not ENABLE_JOINT_VELOCITY_LIMIT:
            dq_pub = q_desired_7 - self.prev_published_q
            dq_pub_norm = float(np.linalg.norm(dq_pub))
            dq_pub_max = float(np.max(np.abs(dq_pub)))
            dq_vel_max = dq_pub_max / CONTROL_DT
            return q_desired_7, dq_pub, dq_pub_norm, dq_pub_max, dq_vel_max

        max_step = JOINT_VEL_LIMIT_RAD_S * CONTROL_DT

        dq_raw = q_desired_7 - self.prev_published_q

        dq_limited = np.clip(
            dq_raw,
            -max_step,
            max_step,
        )

        q_limited = self.prev_published_q + dq_limited

        dq_pub_norm = float(np.linalg.norm(dq_limited))
        dq_pub_max = float(np.max(np.abs(dq_limited)))
        dq_vel_max = dq_pub_max / CONTROL_DT

        return q_limited, dq_limited, dq_pub_norm, dq_pub_max, dq_vel_max

    def update_target_from_omega(
        self,
        omega_state,
        grip_offset=0.0,
        grip_axis_world=None,
    ):
        if not omega_state.initialized:
            return False

        (
            omega_to_fr3_pos,
            omega_to_fr3_rot,
            omega_rot_sign,
        ) = self.get_axis_mapping()

        delta_omega = omega_state.pos - omega_state.initial_pos
        delta_fr3 = omega_to_fr3_pos @ delta_omega

        self.target_pos = (
            self.fr3_initial_pos
            + OMEGA_POS_SCALE * delta_fr3
        )

        if ENABLE_GRIP_POSITION_CONTROL:
            # ==================================================
            # 右Omegaグリッパによる把持位置補正方向
            # ==================================================

            if grip_axis_world is None:
                axis = GRIP_AXIS.astype(float)
            else:
                axis = np.asarray(grip_axis_world, dtype=float)

            axis_norm = np.linalg.norm(axis)

            if axis_norm < 1e-12:
                axis = GRIP_AXIS.astype(float)
                fallback_norm = np.linalg.norm(axis)

                if fallback_norm < 1e-12:
                    axis = np.array([0.0, 1.0, 0.0])
                else:
                    axis = axis / fallback_norm
            else:
                axis = axis / axis_norm

            if self.arm_name == "left":
                self.target_pos = self.target_pos + grip_offset * axis

            elif self.arm_name == "right":
                self.target_pos = self.target_pos - grip_offset * axis

        R_delta_omega = omega_state.initial_rot.T @ omega_state.rot

        rot_vec_omega = pin.log3(R_delta_omega)

        rot_vec_fr3 = omega_to_fr3_rot @ rot_vec_omega

        rot_vec_fr3 = OMEGA_ROT_SCALE * (
            omega_rot_sign * rot_vec_fr3
        )

        R_delta_mapped = pin.exp3(rot_vec_fr3)

        self.target_rot = self.fr3_initial_rot @ R_delta_mapped

        return True

    def update_mid_target(
        self,
        arm_omega_state,
        left_omega_state,
    ):

        if not self.enable_midpoint_control:
            return

        if self.mid_initial_pos is None:
            return

        if self.mid_frame_id is None:
            return


        self.mid_task_enabled_now = True
        self.mid_task_enabled_prev = True

        # ===========================
        # 現在の中間リンク実位置を取得
        # ===========================
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

        current_mid_se3 = self.data.oMf[self.mid_frame_id]
        current_mid_pos = current_mid_se3.translation.copy()

        # ===========================
        # 中間リンクを動かしたい軸方向
        # ===========================
        axis = MIDPOINT_AXIS.astype(float)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-12:
            axis = np.array([0.0, 1.0, 0.0])
        else:
            axis = axis / axis_norm

        if self.arm_name == "right":
            signed_axis = -axis
        elif self.arm_name == "left":
            signed_axis = axis
        else:
            signed_axis = axis

        self.mid_signed_axis = signed_axis

        # ===========================
        # 左Omegaグリッパの入力量
        # ===========================
        if left_omega_state.gripper_initialized:
            current_gripper_delta = left_omega_state.gripper_delta
        else:
            current_gripper_delta = 0.0

        if self.mid_reference_pos is None:
            self.mid_reference_pos = current_mid_pos.copy()
            self.mid_reference_delta_fr3 = np.zeros(3)
            self.mid_reference_gripper_delta = current_gripper_delta

        gripper_delta_since_reference = (
            current_gripper_delta
            - self.mid_reference_gripper_delta
        )

        if abs(gripper_delta_since_reference) < MID_GRIPPER_DEADBAND:
            move_amount = 0.0
        else:
            move_amount = (
                gripper_delta_since_reference
                * FINGER_TO_MID_SCALE
            )

        move_amount = float(np.clip(
            move_amount,
            -MIDPOINT_TARGET_LIMIT,
            MIDPOINT_TARGET_LIMIT,
        ))

        self.mid_move_amount = move_amount

        self.mid_base_pos = current_mid_pos.copy()

        self.mid_target_pos = (
            self.mid_base_pos
            + self.mid_signed_axis * self.mid_move_amount
        )

    def get_current_ee_pose(self):
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

        current_se3 = self.data.oMf[self.frame_id]

        return current_se3.translation.copy(), current_se3.rotation.copy()

    def set_lifting_target_from_pose(self, target_pos, target_rot):
        self.target_pos = target_pos.copy()
        self.target_rot = target_rot.copy()

    def rebase_omega_mapping_at_current_pose(self, omega_state):
        """
        LIFTING MODE解除後に通常モードへ戻るとき，
        Omegaの現在位置とFR3の現在手先位置を新しい基準にする。
        """
        current_pos, current_rot = self.get_current_ee_pose()

        self.fr3_initial_pos = current_pos.copy()
        self.fr3_initial_rot = current_rot.copy()
        self.target_pos = current_pos.copy()
        self.target_rot = current_rot.copy()

        if omega_state.initialized:
            omega_state.initial_pos = omega_state.pos.copy()
            omega_state.initial_rot = omega_state.rot.copy()

    def rebase_midpoint_reference(
        self,
        arm_omega_state,
        left_omega_state,
    ):

        if not self.enable_midpoint_control:
            return

        if self.mid_frame_id is None:
            return

        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

        current_mid_se3 = self.data.oMf[self.mid_frame_id]
        current_mid_pos = current_mid_se3.translation.copy()

        if arm_omega_state.initialized:
            current_delta_fr3 = self.compute_current_delta_fr3(arm_omega_state)
        else:
            current_delta_fr3 = np.zeros(3)

        if left_omega_state.gripper_initialized:
            current_gripper_delta = left_omega_state.gripper_delta
        else:
            current_gripper_delta = 0.0

        self.mid_reference_pos = current_mid_pos.copy()
        self.mid_reference_delta_fr3 = current_delta_fr3.copy()
        self.mid_reference_gripper_delta = current_gripper_delta

        self.mid_base_pos = current_mid_pos.copy()
        self.mid_target_pos = current_mid_pos.copy()
        self.mid_move_amount = 0.0

        self.mid_task_enabled_now = True
        self.mid_task_enabled_prev = True

        self.node.get_logger().info(
            f"{self.arm_name}: midpoint reference rebased, "
            f"mid_reference_pos={self.mid_reference_pos}, "
            f"gripper_ref={self.mid_reference_gripper_delta:.5f}"
        )

    def compute_task_error_and_jacobian(self, current_se3):
        pos_err = self.target_pos - current_se3.translation
        rot_err = pin.log3(self.target_rot @ current_se3.rotation.T)

        J6 = pin.computeFrameJacobian(
            self.model,
            self.data,
            self.q,
            self.frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        if EE_TASK_MODE == "pose":
            err = np.concatenate([pos_err, rot_err])
            J_task = J6[:, self.active_idx]
        elif EE_TASK_MODE == "position":
            err = pos_err
            J_task = J6[:3, self.active_idx]
        else:
            raise ValueError("EE_TASK_MODE must be pose or position")

        return err, J_task, pos_err, rot_err

    def compute_midpoint_nullspace_dq(self, N_task):

        if not self.enable_midpoint_control:
            return np.zeros(len(self.active_idx)), None, None, None

        if not self.mid_task_enabled_now:
            return np.zeros(len(self.active_idx)), None, None, None

        if self.mid_target_pos is None or self.mid_base_pos is None:
            return np.zeros(len(self.active_idx)), None, None, None

        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

        current_mid_se3 = self.data.oMf[self.mid_frame_id]
        current_mid_pos = current_mid_se3.translation.copy()

        signed_axis = self.mid_signed_axis.astype(float)
        axis_norm = np.linalg.norm(signed_axis)

        if axis_norm < 1e-12:
            signed_axis = np.array([0.0, 1.0, 0.0])
        else:
            signed_axis = signed_axis / axis_norm

        current_extra_scalar = float(
            signed_axis @ (current_mid_pos - self.mid_base_pos)
        )

        target_extra_scalar = float(self.mid_move_amount)

        extra_err_scalar = target_extra_scalar - current_extra_scalar

        extra_err_scalar = float(np.clip(
            extra_err_scalar,
            -MID_ERR_MAX,
            MID_ERR_MAX,
        ))

        xdot_mid_scalar = K_MID * extra_err_scalar

        J_mid6 = pin.computeFrameJacobian(
            self.model,
            self.data,
            self.q,
            self.mid_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        J_mid = J_mid6[:3, self.active_idx]

        J_mid_axis = signed_axis.reshape(1, 3) @ J_mid

        J_mid_axis_null = J_mid_axis @ N_task

        dq_mid = (
            N_task
            @ self.damped_pinv(J_mid_axis_null, label="mid")
            @ np.array([xdot_mid_scalar])
        )

        mid_err_vec = signed_axis * extra_err_scalar

        return dq_mid, current_mid_pos, self.mid_target_pos, mid_err_vec

    def solve_ik(self):
        max_iter = 10
        alpha = 0.8
        eps = 1e-4

        last_pos_err = None
        last_rot_err = None
        last_mid_pos = None
        last_mid_target = None
        last_mid_err = None

        for _ in range(max_iter):
            self.q[7] = 0.02
            self.q[8] = 0.02

            pin.forwardKinematics(self.model, self.data, self.q)
            pin.updateFramePlacements(self.model, self.data)

            current_se3 = self.data.oMf[self.frame_id]

            err, J_task, pos_err, rot_err = self.compute_task_error_and_jacobian(
                current_se3
            )

            last_pos_err = pos_err
            last_rot_err = rot_err

            J_task_pinv = self.damped_pinv(J_task, label="ee")

            dq_task = J_task_pinv @ err

            active_dim = len(self.active_idx)
            N_task = np.eye(active_dim) - J_task_pinv @ J_task

            dq_mid, mid_pos, mid_target, mid_err = self.compute_midpoint_nullspace_dq(
                N_task=N_task,
            )

            last_mid_pos = mid_pos
            last_mid_target = mid_target
            last_mid_err = mid_err

            dq_active = dq_task + dq_mid

            dq = np.zeros(self.model.nv)

            for j, idx in enumerate(self.active_idx):
                dq[idx] = dq_active[j]

            self.q = self.q + alpha * dq

            self.q[7] = 0.02
            self.q[8] = 0.02

            task_norm = np.linalg.norm(err)

            if task_norm < eps:
                break

        return (
            self.q,
            last_pos_err,
            last_rot_err,
            last_mid_pos,
            last_mid_target,
            last_mid_err,
        )

    def publish_q_goal(self, q):
        (
            q_limited_7,
            dq_pub,
            dq_pub_norm,
            dq_pub_max,
            dq_vel_max,
        ) = self.apply_joint_velocity_limit(q)

        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = q_limited_7.tolist()

        self.q_goal_pub.publish(msg)

        self.q[:7] = q_limited_7
        self.q[7] = 0.02
        self.q[8] = 0.02

        self.prev_published_q = q_limited_7.copy()

        return dq_pub_norm, dq_pub_max, dq_vel_max


class OmegaToFR3QGoal(Node):
    def __init__(self, mode):
        super().__init__(f"omega_to_fr3_q_goal_{mode}")

        if mode not in ["right", "left", "dual"]:
            raise ValueError("--name must be right, left, or dual")

        if EE_TASK_MODE not in ["pose", "position"]:
            raise ValueError("EE_TASK_MODE must be pose or position")

        self.mode = mode

        self.right_omega = OmegaState()
        self.left_omega = OmegaState()

        self.right_q_goal_pub = None
        self.left_q_goal_pub = None

        self.right_arm = None
        self.left_arm = None

        if self.mode in ["right", "dual"]:
            self.right_q_goal_pub = self.create_publisher(
                JointState,
                RIGHT_Q_GOAL_TOPIC,
                10,
            )

            self.right_arm = ArmIK(
                node=self,
                arm_name="right",
                q_goal_pub=self.right_q_goal_pub,
                enable_midpoint_control=ENABLE_RIGHT_MIDPOINT_CONTROL,
            )

        if self.mode in ["left", "dual"]:
            self.left_q_goal_pub = self.create_publisher(
                JointState,
                LEFT_Q_GOAL_TOPIC,
                10,
            )

            self.left_arm = ArmIK(
                node=self,
                arm_name="left",
                q_goal_pub=self.left_q_goal_pub,
                enable_midpoint_control=ENABLE_LEFT_MIDPOINT_CONTROL,
            )

        self.create_subscription(
            State,
            RIGHT_OMEGA_TOPIC,
            self.right_omega_callback,
            10,
        )

        self.create_subscription(
            State,
            LEFT_OMEGA_TOPIC,
            self.left_omega_callback,
            10,
        )

        # ===========================
        # 把持アシスト用 外力購読
        # ===========================
        self.right_force_world = np.zeros(3)
        self.left_force_world = np.zeros(3)

        self.right_force_received = False
        self.left_force_received = False

        self.grasp_force_filtered = 0.0
        self.grasp_force_target = None
        self.grasp_assist_offset = 0.0

        self.grasp_assist_waiting_logged = False

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

        self.timer = self.create_timer(CONTROL_DT, self.loop)

        self.loop_count = 0

        # ===========================
        # Lifting mode状態
        # ===========================
        self.lifting_mode = False
        self.lifting_mode_prev_button = False

        self.lift_omega_initial_pos = np.zeros(3)
        self.lift_omega_initial_rot = np.eye(3)

        # local pose
        self.lift_right_pos0 = None
        self.lift_left_pos0 = None
        self.lift_right_rot0 = None
        self.lift_left_rot0 = None

        # common world pose
        self.lift_center_pos0 = None
        self.lift_right_rel_pos = None
        self.lift_left_rel_pos = None

        self.get_logger().info("Omega to FR3 q_goal node started")
        self.get_logger().info(f"mode = {self.mode}")
        self.get_logger().info(
            f"grasp assist axis mode = {GRASP_ASSIST_AXIS_MODE}"
        )
        self.get_logger().info(
            "dual base offset: "
            f"left_base={LEFT_FR3_BASE_POS_WORLD.tolist()}, "
            f"right_base={RIGHT_FR3_BASE_POS_WORLD.tolist()}"
        )

        if ENABLE_GRASP_ASSIST:
            self.get_logger().info(
                "GRASP ASSIST ENABLED: "
                f"right_wrench={RIGHT_EE_WRENCH_TOPIC}, "
                f"left_wrench={LEFT_EE_WRENCH_TOPIC}, "
                f"max_offset={GRASP_ASSIST_MAX_OFFSET:.4f} m"
            )
        else:
            self.get_logger().info("GRASP ASSIST DISABLED")

    def quat_to_rot(self, qx, qy, qz, qw):
        norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)

        if norm < 1e-12:
            return np.eye(3)

        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm

        return np.array([
            [
                1.0 - 2.0 * (qy*qy + qz*qz),
                2.0 * (qx*qy - qz*qw),
                2.0 * (qx*qz + qy*qw),
            ],
            [
                2.0 * (qx*qy + qz*qw),
                1.0 - 2.0 * (qx*qx + qz*qz),
                2.0 * (qy*qz - qx*qw),
            ],
            [
                2.0 * (qx*qz - qy*qw),
                2.0 * (qy*qz + qx*qw),
                1.0 - 2.0 * (qx*qx + qy*qy),
            ],
        ])

    def extract_gripper_angle(self, msg):
        try:
            return float(msg.gripper.angle)
        except Exception:
            return None

    def extract_gripper_button(self, msg):
        try:
            return bool(msg.gripper.button)
        except Exception:
            return False

    def update_omega_state(self, omega_state, msg, name):
        p = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])

        R = self.quat_to_rot(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )

        if not omega_state.initialized:
            omega_state.initial_pos = p.copy()
            omega_state.initial_rot = R.copy()
            omega_state.initialized = True

            self.get_logger().info(
                f"{name} Omega initialized: pos={omega_state.initial_pos}"
            )

        omega_state.pos = p
        omega_state.rot = R

        gripper_angle = self.extract_gripper_angle(msg)

        if gripper_angle is not None:
            if not omega_state.gripper_initialized:
                omega_state.gripper_initial = gripper_angle
                omega_state.gripper_initialized = True

                self.get_logger().info(
                    f"{name} gripper initialized: "
                    f"value={gripper_angle:.6f}, field=gripper.angle"
                )

            omega_state.gripper_value = gripper_angle
            omega_state.gripper_delta = (
                gripper_angle
                - omega_state.gripper_initial
            )

        omega_state.gripper_button = self.extract_gripper_button(msg)

    def compute_right_grip_close_and_offset(self):
        if not ENABLE_GRIP_POSITION_CONTROL:
            return 0.0, 0.0

        if not self.right_omega.gripper_initialized:
            return 0.0, 0.0

        grip_delta_signed = (
            self.right_omega.gripper_initial
            - self.right_omega.gripper_value
        )

        grip_offset = GRIP_POS_SCALE * grip_delta_signed

        return grip_delta_signed, grip_offset
    def right_omega_callback(self, msg):
        self.update_omega_state(
            self.right_omega,
            msg,
            "right",
        )

    def left_omega_callback(self, msg):
        self.update_omega_state(
            self.left_omega,
            msg,
            "left",
        )

    def right_wrench_callback(self, msg):
        self.right_force_world = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
        ])

        self.right_force_received = True

    def left_wrench_callback(self, msg):
        self.left_force_world = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
        ])

        self.left_force_received = True

    def can_use_lifting_mode(self):
        return (
            ENABLE_LIFTING_MODE
            and self.mode == "dual"
            and self.right_arm is not None
            and self.left_arm is not None
            and self.right_omega.initialized
        )

    # ------------------------------------------------------
    # Dual-arm world/local transform utilities
    # ------------------------------------------------------
    def local_to_common_world(self, side, pos_local):
        pos_local = np.asarray(pos_local, dtype=float)

        if side == "right":
            return RIGHT_FR3_BASE_POS_WORLD + pos_local

        if side == "left":
            return LEFT_FR3_BASE_POS_WORLD + pos_local

        return pos_local.copy()

    def common_world_to_local(self, side, pos_world):
        pos_world = np.asarray(pos_world, dtype=float)

        if side == "right":
            return pos_world - RIGHT_FR3_BASE_POS_WORLD

        if side == "left":
            return pos_world - LEFT_FR3_BASE_POS_WORLD

        return pos_world.copy()

    def get_default_grip_axis(self):
        axis = GRIP_AXIS.astype(float)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-12:
            return np.array([0.0, 1.0, 0.0])

        return axis / axis_norm

    def get_grasp_axis_from_local_positions(self, right_pos_local, left_pos_local):
        """
        左右FR3のローカル手先位置を共通world座標へ変換し，
        left EE -> right EE の3次元単位ベクトルを返す。
        """
        if GRASP_ASSIST_AXIS_MODE != "between_ee_world":
            return self.get_default_grip_axis()

        right_pos_world = self.local_to_common_world(
            "right",
            right_pos_local,
        )
        left_pos_world = self.local_to_common_world(
            "left",
            left_pos_local,
        )

        grip_vec = right_pos_world - left_pos_world
        dist = np.linalg.norm(grip_vec)

        if dist < 1e-6:
            return self.get_default_grip_axis()

        return grip_vec / dist

    def compute_current_between_ee_axis(self):
        if self.right_arm is None or self.left_arm is None:
            return self.get_default_grip_axis()

        try:
            right_pos_local, _ = self.right_arm.get_current_ee_pose()
            left_pos_local, _ = self.left_arm.get_current_ee_pose()

            right_pos_world = self.local_to_common_world(
                "right",
                right_pos_local,
            )
            left_pos_world = self.local_to_common_world(
                "left",
                left_pos_local,
            )

            grip_vec = right_pos_world - left_pos_world
            dist = np.linalg.norm(grip_vec)

            if dist < 1e-6:
                return self.get_default_grip_axis()

            grip_axis = grip_vec / dist

            if self.loop_count % 1000 == 0:
                center = 0.5 * (right_pos_world + left_pos_world)
                self.get_logger().info(
                    "BETWEEN_EE_WORLD_AXIS: "
                    f"left_world={left_pos_world.round(3).tolist()}, "
                    f"right_world={right_pos_world.round(3).tolist()}, "
                    f"center={center.round(3).tolist()}, "
                    f"axis_L_to_R={grip_axis.round(3).tolist()}, "
                    f"dist={dist:.3f} m"
                )

            return grip_axis

        except Exception as e:
            self.get_logger().warn(
                f"failed to compute current between-EE world axis: {e}"
            )
            return self.get_default_grip_axis()

    def compute_grip_force_from_wrench(self, right_pos=None, left_pos=None):
        if not self.right_force_received or not self.left_force_received:
            return None

        if right_pos is not None and left_pos is not None:
            grip_axis = self.get_grasp_axis_from_local_positions(
                right_pos_local=right_pos,
                left_pos_local=left_pos,
            )
        else:
            grip_axis = self.compute_current_between_ee_axis()

        left_inward_force = FRANKA_FORCE_SIGN * float(
            self.left_force_world @ grip_axis
        )

        right_inward_force = FRANKA_FORCE_SIGN * float(
            self.right_force_world @ (-grip_axis)
        )

        left_inward_force = max(0.0, left_inward_force)
        right_inward_force = max(0.0, right_inward_force)

        grip_force = 0.5 * (
            left_inward_force
            + right_inward_force
        )

        return grip_force, grip_axis, left_inward_force, right_inward_force

    def reset_grasp_assist(self):
        self.grasp_force_filtered = 0.0
        self.grasp_force_target = None
        self.grasp_assist_offset = 0.0
        self.grasp_assist_waiting_logged = False

    def initialize_grasp_assist_target(self, right_pos=None, left_pos=None):
        if not ENABLE_GRASP_ASSIST:
            return False

        grip_info = self.compute_grip_force_from_wrench(
            right_pos=right_pos,
            left_pos=left_pos,
        )

        if grip_info is None:
            if not self.grasp_assist_waiting_logged:
                self.get_logger().warn(
                    "GRASP ASSIST INIT: wrench not received yet"
                )
                self.grasp_assist_waiting_logged = True

            return False

        grip_force, grip_axis, left_inward, right_inward = grip_info

        self.grasp_force_filtered = grip_force
        self.grasp_force_target = grip_force + GRASP_FORCE_MARGIN_N
        self.grasp_assist_offset = 0.0
        self.grasp_assist_waiting_logged = False

        self.get_logger().info(
            "GRASP ASSIST INIT: "
            f"F_grip={grip_force:.3f} N, "
            f"F_target={self.grasp_force_target:.3f} N, "
            f"axis={grip_axis.round(3).tolist()}, "
            f"F_L_in={left_inward:.3f} N, "
            f"F_R_in={right_inward:.3f} N"
        )

        return True

    def apply_grasp_assist(self, right_target_pos, left_target_pos):
        if not ENABLE_GRASP_ASSIST:
            return right_target_pos, left_target_pos

        if self.grasp_force_target is None:
            initialized = self.initialize_grasp_assist_target(
                right_pos=right_target_pos,
                left_pos=left_target_pos,
            )

            if not initialized:
                return right_target_pos, left_target_pos

        grip_info = self.compute_grip_force_from_wrench(
            right_pos=right_target_pos,
            left_pos=left_target_pos,
        )

        if grip_info is None:
            return right_target_pos, left_target_pos

        grip_force, grip_axis, left_inward, right_inward = grip_info

        self.grasp_force_filtered = (
            (1.0 - GRASP_FORCE_LPF_ALPHA) * self.grasp_force_filtered
            + GRASP_FORCE_LPF_ALPHA * grip_force
        )

        force_error = self.grasp_force_target - self.grasp_force_filtered

        if abs(force_error) < GRASP_FORCE_DEADBAND_N:
            force_error = 0.0

        offset_step = GRASP_ASSIST_KI * force_error * CONTROL_DT

        offset_step = float(np.clip(
            offset_step,
            -GRASP_ASSIST_MAX_STEP,
            GRASP_ASSIST_MAX_STEP,
        ))

        self.grasp_assist_offset += offset_step

        self.grasp_assist_offset = float(np.clip(
            self.grasp_assist_offset,
            0.0,
            GRASP_ASSIST_MAX_OFFSET,
        ))


        left_target_pos_assisted = (
            left_target_pos
            + 0.5 * self.grasp_assist_offset * grip_axis
        )

        right_target_pos_assisted = (
            right_target_pos
            - 0.5 * self.grasp_assist_offset * grip_axis
        )

        if self.loop_count % 500 == 0:
            self.get_logger().info(
                "GRASP ASSIST: "
                f"F={self.grasp_force_filtered:.3f} N, "
                f"F_target={self.grasp_force_target:.3f} N, "
                f"err={force_error:.3f} N, "
                f"offset={self.grasp_assist_offset*1000.0:.2f} mm, "
                f"axis={grip_axis.round(3).tolist()}, "
                f"F_L_in={left_inward:.3f} N, "
                f"F_R_in={right_inward:.3f} N"
            )

        return right_target_pos_assisted, left_target_pos_assisted

    def activate_lifting_mode(self):
        if not self.can_use_lifting_mode():
            self.get_logger().warn(
                "LIFTING trigger detected, but cannot enter mode: "
                f"ENABLE_LIFTING_MODE={ENABLE_LIFTING_MODE}, "
                f"mode={self.mode}, "
                f"right_arm={self.right_arm is not None}, "
                f"left_arm={self.left_arm is not None}, "
                f"right_omega_initialized={self.right_omega.initialized}"
            )
            return False

        right_pos_local, right_rot = self.right_arm.get_current_ee_pose()
        left_pos_local, left_rot = self.left_arm.get_current_ee_pose()

        right_pos_world = self.local_to_common_world(
            "right",
            right_pos_local,
        )
        left_pos_world = self.local_to_common_world(
            "left",
            left_pos_local,
        )

        center_world = 0.5 * (right_pos_world + left_pos_world)

        self.lift_omega_initial_pos = self.right_omega.pos.copy()
        self.lift_omega_initial_rot = self.right_omega.rot.copy()

        self.lift_right_pos0 = right_pos_local.copy()
        self.lift_left_pos0 = left_pos_local.copy()
        self.lift_right_rot0 = right_rot.copy()
        self.lift_left_rot0 = left_rot.copy()

        self.lift_center_pos0 = center_world.copy()
        self.lift_right_rel_pos = right_pos_world - center_world
        self.lift_left_rel_pos = left_pos_world - center_world

        self.lifting_mode = True

        self.reset_grasp_assist()

        if self.right_arm is not None:
            self.right_arm.rebase_midpoint_reference(
                arm_omega_state=self.right_omega,
                left_omega_state=self.left_omega,
            )

        if self.left_arm is not None:
            self.left_arm.rebase_midpoint_reference(
                arm_omega_state=self.right_omega,
                left_omega_state=self.left_omega,
            )

        self.initialize_grasp_assist_target(
            right_pos=right_pos_local,
            left_pos=left_pos_local,
        )

        distance = float(np.linalg.norm(right_pos_world - left_pos_world))
        grip_axis = self.get_grasp_axis_from_local_positions(
            right_pos_local=right_pos_local,
            left_pos_local=left_pos_local,
        )

        self.get_logger().info(
            "LIFTING MODE ON: "
            f"center_world={self.lift_center_pos0.round(3).tolist()}, "
            f"right_world={right_pos_world.round(3).tolist()}, "
            f"left_world={left_pos_world.round(3).tolist()}, "
            f"right_rel_world={self.lift_right_rel_pos.round(3).tolist()}, "
            f"left_rel_world={self.lift_left_rel_pos.round(3).tolist()}, "
            f"hand_distance={distance:.4f}, "
            f"grip_axis={grip_axis.round(3).tolist()}"
        )

        return True

    def deactivate_lifting_mode(self):
        if not self.lifting_mode:
            return

        self.lifting_mode = False

        if self.right_arm is not None:
            self.right_arm.rebase_omega_mapping_at_current_pose(self.right_omega)

        if self.left_arm is not None:
            self.left_arm.rebase_omega_mapping_at_current_pose(self.left_omega)

        if self.right_omega.gripper_initialized:
            self.right_omega.gripper_initial = self.right_omega.gripper_value
            self.right_omega.gripper_delta = 0.0

        self.reset_grasp_assist()


        if self.right_arm is not None:
            self.right_arm.rebase_midpoint_reference(
                arm_omega_state=self.right_omega,
                left_omega_state=self.left_omega,
            )

        if self.left_arm is not None:
            self.left_arm.rebase_midpoint_reference(
                arm_omega_state=self.left_omega,
                left_omega_state=self.left_omega,
            )

        self.get_logger().info(
            "LIFTING MODE OFF: "
            "rebased EE/Omega references, reset right gripper reference, "
            "reset grasp assist, and rebased midpoint references"
        )

    def update_lifting_targets_from_right_omega(self):
        if not self.lifting_mode:
            return False

        if self.lift_right_pos0 is None or self.lift_left_pos0 is None:
            return False

        if self.lift_right_rot0 is None or self.lift_left_rot0 is None:
            return False

        if self.lift_center_pos0 is None:
            return False

        if self.lift_right_rel_pos is None or self.lift_left_rel_pos is None:
            return False

        if not self.right_omega.initialized:
            return False

        omega_to_fr3_pos, omega_to_fr3_rot, omega_rot_sign = (
            self.right_arm.get_axis_mapping()
        )

        delta_omega = self.right_omega.pos - self.lift_omega_initial_pos
        delta_fr3 = omega_to_fr3_pos @ delta_omega

        center_delta_world = OMEGA_POS_SCALE * delta_fr3
        center_target_world = self.lift_center_pos0 + center_delta_world

        if LIFTING_MODE_USE_RIGHT_OMEGA_ROT:
            R_delta_omega = self.lift_omega_initial_rot.T @ self.right_omega.rot

            rot_vec_omega = pin.log3(R_delta_omega)

            rot_vec_fr3 = omega_to_fr3_rot @ rot_vec_omega

            rot_vec_fr3 = OMEGA_ROT_SCALE * (
                omega_rot_sign * rot_vec_fr3
            )

            R_delta_mapped = pin.exp3(rot_vec_fr3)

            R_object0 = self.lift_right_rot0

            R_world_delta = (
                R_object0
                @ R_delta_mapped
                @ R_object0.T
            )

        else:
            R_world_delta = np.eye(3)

        right_rel_rotated_world = R_world_delta @ self.lift_right_rel_pos
        left_rel_rotated_world = R_world_delta @ self.lift_left_rel_pos

        right_target_world = center_target_world + right_rel_rotated_world
        left_target_world = center_target_world + left_rel_rotated_world

        # IK目標は各FR3ローカル座標なので，world -> localへ戻す。
        right_target_pos = self.common_world_to_local(
            "right",
            right_target_world,
        )
        left_target_pos = self.common_world_to_local(
            "left",
            left_target_world,
        )

        right_target_pos, left_target_pos = self.apply_grasp_assist(
            right_target_pos=right_target_pos,
            left_target_pos=left_target_pos,
        )

        if LIFTING_MODE_USE_RIGHT_OMEGA_ROT:
            right_target_rot = R_world_delta @ self.lift_right_rot0
            left_target_rot = R_world_delta @ self.lift_left_rot0
        else:
            right_target_rot = self.lift_right_rot0.copy()
            left_target_rot = self.lift_left_rot0.copy()

        self.right_arm.set_lifting_target_from_pose(
            right_target_pos,
            right_target_rot,
        )

        self.left_arm.set_lifting_target_from_pose(
            left_target_pos,
            left_target_rot,
        )

        return True

    def update_lifting_mode_state(self):
        button_now = bool(self.right_omega.gripper_button)

        if button_now and not self.lifting_mode:
            self.activate_lifting_mode()

        elif (not button_now) and self.lifting_mode:
            self.deactivate_lifting_mode()

        self.lifting_mode_prev_button = button_now

    def loop(self):
        self.update_lifting_mode_state()

        if self.lifting_mode:
            lifting_ready = self.update_lifting_targets_from_right_omega()

            if lifting_ready:
                self.right_arm.update_mid_target(
                    arm_omega_state=self.right_omega,
                    left_omega_state=self.left_omega,
                )

                self.left_arm.update_mid_target(
                    arm_omega_state=self.right_omega,
                    left_omega_state=self.left_omega,
                )

                (
                    q_right,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = self.right_arm.solve_ik()

                (
                    q_left,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = self.left_arm.solve_ik()

                self.right_arm.publish_q_goal(q_right)
                self.left_arm.publish_q_goal(q_left)

            self.loop_count += 1
            return

        right_grip_close, grip_offset = self.compute_right_grip_close_and_offset()

        grip_axis_world = self.compute_current_between_ee_axis()

        if self.right_arm is not None:
            right_ready = self.right_arm.update_target_from_omega(
                self.right_omega,
                grip_offset=grip_offset,
                grip_axis_world=grip_axis_world,
            )

            if right_ready:
                self.right_arm.update_mid_target(
                    arm_omega_state=self.right_omega,
                    left_omega_state=self.left_omega,
                )

                (
                    q_right,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = self.right_arm.solve_ik()

                self.right_arm.publish_q_goal(q_right)

        if self.left_arm is not None:
            left_ready = self.left_arm.update_target_from_omega(
                self.left_omega,
                grip_offset=grip_offset,
                grip_axis_world=grip_axis_world,
            )

            if left_ready:
                self.left_arm.update_mid_target(
                    arm_omega_state=self.left_omega,
                    left_omega_state=self.left_omega,
                )

                (
                    q_left,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = self.left_arm.solve_ik()

                self.left_arm.publish_q_goal(q_left)

        if self.loop_count % 1000 == 0:
            self.get_logger().info(
                "GRIP POSITION CONTROL: "
                f"close={right_grip_close:.5f}, "
                f"offset={grip_offset*1000.0:.2f} mm, "
                f"axis_world_L_to_R={grip_axis_world.round(3).tolist()}"
            )

        self.loop_count += 1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        type=str,
        default="dual",
        choices=["right", "left", "dual"],
        help="Which arm q_goal to publish: right, left, or dual",
    )

    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)

    node = OmegaToFR3QGoal(mode=args.name)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()