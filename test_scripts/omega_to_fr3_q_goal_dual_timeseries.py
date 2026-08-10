#!/usr/bin/env python3


from __future__ import annotations

import numpy as np
import pinocchio as pin

import omega_to_fr3_q_goal_dual as base
import omega_to_fr3_q_goal_dual_bilateral as bilateral


# ==========================================================
# 既存機能の設定
# ==========================================================
# 左Omegaグリッパから左右両方の中間リンク目標を生成する。
base.ENABLE_RIGHT_MIDPOINT_CONTROL = False
base.ENABLE_LEFT_MIDPOINT_CONTROL = False


# ==========================================================
# 時系列逆運動学の設定
# ==========================================================
# 同時に最適化する将来姿勢数
TSIK_HORIZON = 4

# 1回のsolve_ik()で行うGauss-Newton反復回数
TSIK_MAX_ITER = 3

# 時系列内の仮想時間間隔
TSIK_SEQUENCE_DT = 0.01

# 評価項の重み
TSIK_W_EE_POS = 12.0
TSIK_W_EE_ROT = 5.0
TSIK_W_MID = 3.0
TSIK_W_VEL = 0.35
TSIK_W_ACC = 0.08
TSIK_W_SEED = 0.02

# 終端側の目標を強くする係数
TSIK_TERMINAL_GAIN = 1.5

# 正規方程式のダンピング
TSIK_DAMPING = 1.0e-3

# 1反復で更新する全時系列関節角の最大ノルム
TSIK_MAX_DELTA_NORM = 0.30

# Gauss-Newton更新率
TSIK_ALPHA = 0.75

# 収束判定
TSIK_EPS = 1.0e-5

# 時系列内の隣接姿勢間の関節速度上限
TSIK_MAX_JOINT_VEL_RAD_S = 2.0

# 関節可動限界から確保する余裕
TSIK_JOINT_LIMIT_MARGIN = 0.01

# 診断ログ間隔
TSIK_LOG_EVERY = 1000


class TimeSeriesArmIK(base.ArmIK):
    """
    既存ArmIKの入出力を保ち、
    solve_ik()だけを時系列最適化へ変更する。
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

        self.prev_prev_published_q = self.prev_published_q.copy()
        self.tsik_q_sequence = None
        self.tsik_last_residual_norm = None
        self.tsik_solve_count = 0

        self.node.get_logger().info(
            f"{self.arm_name}: time-series IK enabled, "
            f"horizon={TSIK_HORIZON}, "
            f"iterations={TSIK_MAX_ITER}"
        )

    def reset_timeseries_ik(self):
        """
        Omegaまたは中間リンクの基準を変更したときに、
        前回の時系列解をリセットする。
        """
        self.tsik_q_sequence = None
        self.tsik_last_residual_norm = None

    def rebase_omega_mapping_at_current_pose(
        self,
        omega_state,
    ):
        super().rebase_omega_mapping_at_current_pose(
            omega_state
        )
        self.reset_timeseries_ik()

    def rebase_midpoint_reference(
        self,
        arm_omega_state,
        left_omega_state,
    ):
        super().rebase_midpoint_reference(
            arm_omega_state=arm_omega_state,
            left_omega_state=left_omega_state,
        )
        self.reset_timeseries_ik()

    def make_full_q_from_active(
        self,
        q_active,
    ):
        """
        7関節のベクトルをPinocchioモデル用の
        全関節ベクトルへ戻す。
        """
        q_full = self.q.copy()

        q_full[:7] = np.asarray(
            q_active,
            dtype=float,
        )

        q_full[7] = 0.02
        q_full[8] = 0.02

        return q_full

    def get_normalized_mid_axis(self):
        """
        左右それぞれの中間リンク移動方向を
        単位ベクトルで返す。
        """
        axis = np.asarray(
            self.mid_signed_axis,
            dtype=float,
        )

        axis_norm = float(
            np.linalg.norm(axis)
        )

        if axis_norm < 1.0e-12:
            if self.arm_name == "right":
                return np.array([
                    0.0,
                    -1.0,
                    0.0,
                ])

            return np.array([
                0.0,
                1.0,
                0.0,
            ])

        return axis / axis_norm

    def build_timeseries_targets(
        self,
        q_anchor,
    ):
        """
        現在姿勢から最新のOmega目標までをH分割し、
        各時刻の手先位置・姿勢・中間リンク目標を作る。
        """
        q_anchor_full = (
            self.make_full_q_from_active(
                q_anchor
            )
        )

        pin.forwardKinematics(
            self.model,
            self.data,
            q_anchor_full,
        )

        pin.updateFramePlacements(
            self.model,
            self.data,
        )

        current_ee = self.data.oMf[
            self.frame_id
        ]

        current_pos = (
            current_ee.translation.copy()
        )

        current_rot = (
            current_ee.rotation.copy()
        )

        delta_rot_vec = pin.log3(
            current_rot.T
            @ self.target_rot
        )

        pos_targets = []
        rot_targets = []
        mid_scalar_targets = []

        mid_enabled = (
            self.enable_midpoint_control
            and self.mid_task_enabled_now
            and self.mid_frame_id is not None
            and self.mid_base_pos is not None
        )

        current_mid_scalar = 0.0
        target_mid_scalar = 0.0

        if mid_enabled:
            current_mid_pos = (
                self.data.oMf[
                    self.mid_frame_id
                ].translation.copy()
            )

            mid_axis = (
                self.get_normalized_mid_axis()
            )

            current_mid_scalar = float(
                mid_axis
                @ (
                    current_mid_pos
                    - self.mid_base_pos
                )
            )

            target_mid_scalar = float(
                self.mid_move_amount
            )

        for k in range(TSIK_HORIZON):
            ratio = (
                float(k + 1)
                / float(TSIK_HORIZON)
            )

            pos_target = (
                current_pos
                + ratio
                * (
                    self.target_pos
                    - current_pos
                )
            )

            rot_target = (
                current_rot
                @ pin.exp3(
                    ratio * delta_rot_vec
                )
            )

            pos_targets.append(
                pos_target
            )

            rot_targets.append(
                rot_target
            )

            if mid_enabled:
                mid_target = (
                    current_mid_scalar
                    + ratio
                    * (
                        target_mid_scalar
                        - current_mid_scalar
                    )
                )

                mid_scalar_targets.append(
                    float(mid_target)
                )

            else:
                mid_scalar_targets.append(
                    None
                )

        return (
            pos_targets,
            rot_targets,
            mid_scalar_targets,
        )

    def initialize_timeseries_sequence(
        self,
        q_anchor,
    ):
        """
        前回の最適化結果を1時刻分シフトして
        次回の初期値に使う。

        初回だけ現在姿勢を全時刻へ複製する。
        """
        q_anchor = np.asarray(
            q_anchor,
            dtype=float,
        )

        if (
            self.tsik_q_sequence is None
            or self.tsik_q_sequence.shape
            != (TSIK_HORIZON, 7)
            or not np.all(
                np.isfinite(
                    self.tsik_q_sequence
                )
            )
        ):
            return np.tile(
                q_anchor,
                (TSIK_HORIZON, 1),
            )

        q_seed = np.empty_like(
            self.tsik_q_sequence
        )

        q_seed[:-1] = (
            self.tsik_q_sequence[1:]
        )

        q_seed[-1] = (
            self.tsik_q_sequence[-1]
        )

        # 現在出力している姿勢と、
        # 前回計算した将来姿勢との中間を初期値にする。
        q_seed[0] = (
            0.5 * q_seed[0]
            + 0.5 * q_anchor
        )

        return q_seed

    def project_timeseries_sequence(
        self,
        q_sequence,
        q_anchor,
    ):
        """
        各時刻の関節角に対して、
        関節可動域制限と時系列内の速度制限を適用する。
        """
        q_sequence = np.asarray(
            q_sequence,
            dtype=float,
        ).copy()

        q_anchor = np.asarray(
            q_anchor,
            dtype=float,
        )

        lower = np.asarray(
            self.model.lowerPositionLimit[
                self.active_idx
            ],
            dtype=float,
        )

        upper = np.asarray(
            self.model.upperPositionLimit[
                self.active_idx
            ],
            dtype=float,
        )

        lower_safe = lower.copy()
        upper_safe = upper.copy()

        finite_lower = np.isfinite(
            lower_safe
        )

        finite_upper = np.isfinite(
            upper_safe
        )

        lower_safe[finite_lower] += (
            TSIK_JOINT_LIMIT_MARGIN
        )

        upper_safe[finite_upper] -= (
            TSIK_JOINT_LIMIT_MARGIN
        )

        max_step = (
            TSIK_MAX_JOINT_VEL_RAD_S
            * TSIK_SEQUENCE_DT
        )

        previous = q_anchor.copy()

        for k in range(TSIK_HORIZON):
            q_k = q_sequence[k].copy()

            q_k = np.where(
                finite_lower,
                np.maximum(
                    q_k,
                    lower_safe,
                ),
                q_k,
            )

            q_k = np.where(
                finite_upper,
                np.minimum(
                    q_k,
                    upper_safe,
                ),
                q_k,
            )

            dq_k = np.clip(
                q_k - previous,
                -max_step,
                max_step,
            )

            q_k = previous + dq_k

            q_k = np.where(
                finite_lower,
                np.maximum(
                    q_k,
                    lower_safe,
                ),
                q_k,
            )

            q_k = np.where(
                finite_upper,
                np.minimum(
                    q_k,
                    upper_safe,
                ),
                q_k,
            )

            q_sequence[k] = q_k
            previous = q_k

        return q_sequence

    @staticmethod
    def append_weighted_rows(
        rows,
        rhs,
        row_local,
        rhs_local,
        weight,
    ):
        """
        重み付き最小二乗問題の行ブロックを追加する。
        """
        if weight <= 0.0:
            return

        scale = float(
            np.sqrt(weight)
        )

        rows.append(
            scale
            * np.asarray(
                row_local,
                dtype=float,
            )
        )

        rhs.append(
            scale
            * np.asarray(
                rhs_local,
                dtype=float,
            ).reshape(-1)
        )

    def build_timeseries_linear_system(
        self,
        q_sequence,
        q_anchor,
        q_previous,
        q_seed,
        pos_targets,
        rot_targets,
        mid_scalar_targets,
    ):
        """
        全時刻を同時に扱う線形化最小二乗問題

            A ΔQ = b

        を構成する。

        評価する項:
          1. 手先位置誤差
          2. 手先姿勢誤差
          3. 中間リンク軸方向誤差
          4. 隣接関節角差
          5. 関節角の2階差分
          6. ウォームスタート解との差
        """
        horizon = TSIK_HORIZON
        dof = 7
        nvar = horizon * dof

        rows = []
        rhs = []

        identity = np.eye(dof)

        # ==================================================
        # 各時刻の手先タスク・中間リンクタスク
        # ==================================================
        for k in range(horizon):
            q_full = (
                self.make_full_q_from_active(
                    q_sequence[k]
                )
            )

            pin.forwardKinematics(
                self.model,
                self.data,
                q_full,
            )

            pin.updateFramePlacements(
                self.model,
                self.data,
            )

            current_ee = (
                self.data.oMf[
                    self.frame_id
                ]
            )

            pos_err = (
                pos_targets[k]
                - current_ee.translation
            )

            rot_err = pin.log3(
                rot_targets[k]
                @ current_ee.rotation.T
            )

            J6 = pin.computeFrameJacobian(
                self.model,
                self.data,
                q_full,
                self.frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )

            J6 = J6[
                :,
                self.active_idx,
            ]

            ratio = (
                float(k + 1)
                / float(horizon)
            )

            terminal_scale = (
                1.0
                + TSIK_TERMINAL_GAIN
                * ratio
                * ratio
            )

            # ----------------------------------------------
            # 手先位置
            # ----------------------------------------------
            pos_row = np.zeros(
                (3, nvar)
            )

            pos_row[
                :,
                k*dof:(k+1)*dof,
            ] = J6[:3, :]

            self.append_weighted_rows(
                rows=rows,
                rhs=rhs,
                row_local=pos_row,
                rhs_local=pos_err,
                weight=(
                    TSIK_W_EE_POS
                    * terminal_scale
                ),
            )

            # ----------------------------------------------
            # 手先姿勢
            # ----------------------------------------------
            if base.EE_TASK_MODE == "pose":
                rot_row = np.zeros(
                    (3, nvar)
                )

                rot_row[
                    :,
                    k*dof:(k+1)*dof,
                ] = J6[3:, :]

                self.append_weighted_rows(
                    rows=rows,
                    rhs=rhs,
                    row_local=rot_row,
                    rhs_local=rot_err,
                    weight=(
                        TSIK_W_EE_ROT
                        * terminal_scale
                    ),
                )

            # ----------------------------------------------
            # 中間リンク
            # ----------------------------------------------
            if (
                mid_scalar_targets[k]
                is not None
                and self.mid_frame_id
                is not None
                and self.mid_base_pos
                is not None
            ):
                mid_axis = (
                    self.get_normalized_mid_axis()
                )

                current_mid_pos = (
                    self.data.oMf[
                        self.mid_frame_id
                    ].translation.copy()
                )

                current_mid_scalar = float(
                    mid_axis
                    @ (
                        current_mid_pos
                        - self.mid_base_pos
                    )
                )

                mid_err = float(
                    mid_scalar_targets[k]
                    - current_mid_scalar
                )

                mid_err = float(
                    np.clip(
                        mid_err,
                        -base.MID_ERR_MAX,
                        base.MID_ERR_MAX,
                    )
                )

                J_mid6 = (
                    pin.computeFrameJacobian(
                        self.model,
                        self.data,
                        q_full,
                        self.mid_frame_id,
                        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                    )
                )

                J_mid = J_mid6[
                    :3,
                    self.active_idx,
                ]

                J_mid_axis = (
                    mid_axis.reshape(1, 3)
                    @ J_mid
                )

                mid_row = np.zeros(
                    (1, nvar)
                )

                mid_row[
                    :,
                    k*dof:(k+1)*dof,
                ] = J_mid_axis

                self.append_weighted_rows(
                    rows=rows,
                    rhs=rhs,
                    row_local=mid_row,
                    rhs_local=np.array([
                        base.K_MID * mid_err
                    ]),
                    weight=(
                        TSIK_W_MID
                        * terminal_scale
                    ),
                )

        # ==================================================
        # 1階差分
        # 隣接する関節姿勢を近づける
        # ==================================================
        for k in range(horizon):
            vel_row = np.zeros(
                (dof, nvar)
            )

            vel_row[
                :,
                k*dof:(k+1)*dof,
            ] = identity

            if k == 0:
                vel_err = (
                    q_anchor
                    - q_sequence[k]
                )

            else:
                vel_row[
                    :,
                    (k-1)*dof:k*dof,
                ] = -identity

                vel_err = -(
                    q_sequence[k]
                    - q_sequence[k-1]
                )

            self.append_weighted_rows(
                rows=rows,
                rhs=rhs,
                row_local=vel_row,
                rhs_local=vel_err,
                weight=TSIK_W_VEL,
            )

        # ==================================================
        # 2階差分
        # 関節速度の急変を抑える
        # ==================================================
        for k in range(horizon):
            acc_row = np.zeros(
                (dof, nvar)
            )

            if k == 0:
                acc_row[
                    :,
                    0:dof,
                ] = identity

                acc_value = (
                    q_sequence[0]
                    - 2.0 * q_anchor
                    + q_previous
                )

            elif k == 1:
                acc_row[
                    :,
                    dof:2*dof,
                ] = identity

                acc_row[
                    :,
                    0:dof,
                ] = -2.0 * identity

                acc_value = (
                    q_sequence[1]
                    - 2.0
                    * q_sequence[0]
                    + q_anchor
                )

            else:
                acc_row[
                    :,
                    k*dof:(k+1)*dof,
                ] = identity

                acc_row[
                    :,
                    (k-1)*dof:k*dof,
                ] = (
                    -2.0 * identity
                )

                acc_row[
                    :,
                    (k-2)*dof:(k-1)*dof,
                ] = identity

                acc_value = (
                    q_sequence[k]
                    - 2.0
                    * q_sequence[k-1]
                    + q_sequence[k-2]
                )

            self.append_weighted_rows(
                rows=rows,
                rhs=rhs,
                row_local=acc_row,
                rhs_local=-acc_value,
                weight=TSIK_W_ACC,
            )

        # ==================================================
        # ウォームスタート解との差
        # ==================================================
        for k in range(horizon):
            seed_row = np.zeros(
                (dof, nvar)
            )

            seed_row[
                :,
                k*dof:(k+1)*dof,
            ] = identity

            seed_err = (
                q_seed[k]
                - q_sequence[k]
            )

            self.append_weighted_rows(
                rows=rows,
                rhs=rhs,
                row_local=seed_row,
                rhs_local=seed_err,
                weight=TSIK_W_SEED,
            )

        A = np.vstack(rows)
        b = np.concatenate(rhs)

        return A, b

    def compute_timeseries_diagnostics(self):
        """
        元のsolve_ik()と同じ形式の診断値を返す。
        """
        pin.forwardKinematics(
            self.model,
            self.data,
            self.q,
        )

        pin.updateFramePlacements(
            self.model,
            self.data,
        )

        current_se3 = (
            self.data.oMf[
                self.frame_id
            ]
        )

        (
            _,
            _,
            pos_err,
            rot_err,
        ) = (
            self.compute_task_error_and_jacobian(
                current_se3
            )
        )

        mid_pos = None
        mid_target = None
        mid_err_vec = None

        if (
            self.enable_midpoint_control
            and self.mid_task_enabled_now
            and self.mid_frame_id
            is not None
            and self.mid_base_pos
            is not None
        ):
            mid_axis = (
                self.get_normalized_mid_axis()
            )

            mid_pos = (
                self.data.oMf[
                    self.mid_frame_id
                ].translation.copy()
            )

            current_scalar = float(
                mid_axis
                @ (
                    mid_pos
                    - self.mid_base_pos
                )
            )

            target_scalar = float(
                self.mid_move_amount
            )

            scalar_err = float(
                target_scalar
                - current_scalar
            )

            scalar_err = float(
                np.clip(
                    scalar_err,
                    -base.MID_ERR_MAX,
                    base.MID_ERR_MAX,
                )
            )

            mid_target = (
                self.mid_base_pos
                + mid_axis
                * target_scalar
            )

            mid_err_vec = (
                mid_axis
                * scalar_err
            )

        return (
            pos_err,
            rot_err,
            mid_pos,
            mid_target,
            mid_err_vec,
        )

    def solve_ik(self):
        """
        短区間の時系列逆運動学。

        Omega入力から作られた手先目標と中間リンク目標は変更せず、
        H個の将来関節姿勢をまとめて最適化する。

        q_goalへ送るのは時系列の最初の姿勢で、
        次のループで再度、時系列全体を最適化する。
        """
        q_anchor = (
            self.q[:7].copy()
        )

        q_previous = (
            self.prev_prev_published_q.copy()
        )

        q_seed = (
            self.initialize_timeseries_sequence(
                q_anchor
            )
        )

        q_sequence = (
            self.project_timeseries_sequence(
                q_sequence=q_seed,
                q_anchor=q_anchor,
            )
        )

        (
            pos_targets,
            rot_targets,
            mid_scalar_targets,
        ) = (
            self.build_timeseries_targets(
                q_anchor
            )
        )

        last_residual_norm = None

        for _ in range(
            TSIK_MAX_ITER
        ):
            A, b = (
                self.build_timeseries_linear_system(
                    q_sequence=q_sequence,
                    q_anchor=q_anchor,
                    q_previous=q_previous,
                    q_seed=q_seed,
                    pos_targets=pos_targets,
                    rot_targets=rot_targets,
                    mid_scalar_targets=mid_scalar_targets,
                )
            )

            last_residual_norm = float(
                np.linalg.norm(b)
            )

            normal_matrix = (
                A.T @ A
                + (
                    TSIK_DAMPING ** 2
                )
                * np.eye(
                    A.shape[1]
                )
            )

            normal_rhs = A.T @ b

            try:
                delta_flat = (
                    np.linalg.solve(
                        normal_matrix,
                        normal_rhs,
                    )
                )

            except np.linalg.LinAlgError:
                delta_flat = (
                    np.linalg.lstsq(
                        normal_matrix,
                        normal_rhs,
                        rcond=None,
                    )[0]
                )

            delta_norm = float(
                np.linalg.norm(
                    delta_flat
                )
            )

            if (
                delta_norm
                > TSIK_MAX_DELTA_NORM
            ):
                delta_flat *= (
                    TSIK_MAX_DELTA_NORM
                    / max(
                        delta_norm,
                        1.0e-12,
                    )
                )

            q_candidate = (
                q_sequence
                + TSIK_ALPHA
                * delta_flat.reshape(
                    TSIK_HORIZON,
                    7,
                )
            )

            q_candidate = (
                self.project_timeseries_sequence(
                    q_sequence=q_candidate,
                    q_anchor=q_anchor,
                )
            )

            update_norm = float(
                np.linalg.norm(
                    q_candidate
                    - q_sequence
                )
            )

            q_sequence = q_candidate

            if update_norm < TSIK_EPS:
                break

        self.tsik_q_sequence = (
            q_sequence.copy()
        )

        self.tsik_last_residual_norm = (
            last_residual_norm
        )

        self.tsik_solve_count += 1

        q_next = q_sequence[0].copy()

        self.q[:7] = q_next
        self.q[7] = 0.02
        self.q[8] = 0.02

        if (
            TSIK_LOG_EVERY > 0
            and self.tsik_solve_count
            % TSIK_LOG_EVERY
            == 0
        ):
            self.node.get_logger().info(
                f"{self.arm_name}: "
                f"time-series IK, "
                f"horizon={TSIK_HORIZON}, "
                f"residual="
                f"{self.tsik_last_residual_norm:.6f}"
            )

        (
            pos_err,
            rot_err,
            mid_pos,
            mid_target,
            mid_err,
        ) = (
            self.compute_timeseries_diagnostics()
        )

        return (
            self.q,
            pos_err,
            rot_err,
            mid_pos,
            mid_target,
            mid_err,
        )

    def publish_q_goal(
        self,
        q,
    ):
        """
        既存のpublish処理を使用し、
        時系列最適化で使う過去姿勢も更新する。
        """
        previous_published_before_update = (
            self.prev_published_q.copy()
        )

        result = (
            super().publish_q_goal(q)
        )

        self.prev_prev_published_q = (
            previous_published_before_update
        )

        if self.tsik_q_sequence is not None:
            self.tsik_q_sequence[0] = (
                self.prev_published_q.copy()
            )

        return result


# ==========================================================
# 重要
# ==========================================================
# OmegaToFR3QGoal.__init__() が左右ArmIKを生成するときに、
# このTimeSeriesArmIKを使用させる。
base.ArmIK = TimeSeriesArmIK


# ==========================================================
# バイラテラル版のmainを使用する
# ==========================================================
# これにより、
#   --force-mode
#   --force-gain-z
#   --force-sign-z
#   --force-limit
# などの引数が有効になる。
if __name__ == "__main__":
    bilateral.main()