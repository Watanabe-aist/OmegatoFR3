#!/usr/bin/env python3
import pickle
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin


# ================================
# 0) Settings
# ================================
data_dir = Path("/home/watanaberyuto/ws_backup/raw_data/exp0619_50times_Omega")
urdf_path = Path("/home/watanaberyuto/ws/src/factr_teleop-main/src/factr_teleop/factr_teleop/urdf/factr_teleop_franka.urdf")

side = "left"   # "right" or "left"
frame_name = "link_7"

state_topic = f"/franka/{side}/obs_franka_state"
torque_topic = f"/franka/{side}/obs_franka_torque"

csv_path = Path(f"estimated_force_zero_overlay_{side}.csv")
png_path = Path(f"estimated_force_zero_overlay_mean_{side}.png")


# ================================
# 1) Load robot model
# ================================
model = pin.buildModelFromUrdf(str(urdf_path))
data = model.createData()

frame_id = model.getFrameId(frame_name)
if frame_id == len(model.frames):
    print("Available frames:")
    for f in model.frames:
        print(f.name)
    raise ValueError(f"Frame '{frame_name}' not found")


def estimate_wrench(q, tau_ext):
    q = np.asarray(q, dtype=float)
    tau_ext = np.asarray(tau_ext, dtype=float)

    pin.forwardKinematics(model, data, q)
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)

    J = pin.getFrameJacobian(
        model,
        data,
        frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )

    wrench = np.linalg.pinv(J.T) @ tau_ext
    return wrench


# ================================
# 2) Process episodes
# ================================
files = sorted(data_dir.glob("ep_*.pkl"))
print("num episodes:", len(files))

all_episode_times = []
all_episode_force_norms = []

plt.figure(figsize=(10, 6))

with open(csv_path, "w", newline="") as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow([
        "episode", "time",
        "Fx_zero", "Fy_zero", "Fz_zero",
        "force_norm_zero",
        "Mx_zero", "My_zero", "Mz_zero",
    ])

    for ep_idx, pkl_path in enumerate(files):
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)

        if state_topic not in d["data"] or torque_topic not in d["data"]:
            print(f"skip {pkl_path.name}: topic not found")
            continue

        q_list = d["data"][state_topic]
        tau_list = d["data"][torque_topic]
        q_ts = d["timestamps"][state_topic]
        tau_ts = d["timestamps"][torque_topic]

        if len(q_list) == 0 or len(tau_list) == 0:
            print(f"skip {pkl_path.name}: empty data")
            continue

        q_ts = np.asarray(q_ts, dtype=np.int64)
        tau_ts = np.asarray(tau_ts, dtype=np.int64)

        ep_t0 = tau_ts[0]

        ep_t = []
        ep_force_norm = []

        wrench_offset = None

        for i, tau_time in enumerate(tau_ts):
            q_idx = np.argmin(np.abs(q_ts - tau_time))

            q = np.asarray(q_list[q_idx], dtype=float)
            tau_ext = np.asarray(tau_list[i], dtype=float)

            wrench = estimate_wrench(q, tau_ext)

            if wrench_offset is None:
                wrench_offset = wrench.copy()

            wrench_zero = wrench - wrench_offset

            Fx, Fy, Fz = wrench_zero[:3]
            Mx, My, Mz = wrench_zero[3:]

            # ここが重要：灰色も平均もこの同じ force_norm を使う
            force_norm = np.linalg.norm(wrench_zero[:3])

            t = (tau_time - ep_t0) * 1e-9

            writer.writerow([
                ep_idx, t,
                Fx, Fy, Fz,
                force_norm,
                Mx, My, Mz,
            ])

            ep_t.append(t)
            ep_force_norm.append(force_norm)

        if len(ep_t) > 0:
            ep_t = np.asarray(ep_t)
            ep_force_norm = np.asarray(ep_force_norm)

            # 念のため force_norm 自体もゼロスタート化
            ep_force_norm = ep_force_norm - ep_force_norm[0]

            # 各episodeを薄く描画
            plt.plot(
                ep_t,
                ep_force_norm,
                alpha=0.25,
                linewidth=1.0,
                color="gray"
            )

            all_episode_times.append(ep_t)
            all_episode_force_norms.append(ep_force_norm)

        print(f"processed {pkl_path.name}")


# ================================
# 3) Average Plot
# ================================
if len(all_episode_force_norms) > 0:
    min_len = min(len(x) for x in all_episode_force_norms)

    aligned_force_norm = np.array([
        x[:min_len] for x in all_episode_force_norms
    ])

    aligned_time = all_episode_times[0][:min_len]

    # ここが重要：mean(||F||)
    mean_force = np.mean(aligned_force_norm, axis=0)

    plt.plot(
        aligned_time,
        mean_force,
        linewidth=4,
        color="red",
        label="Mean force norm"
    )


# ================================
# 4) Save Plot
# ================================
plt.xlabel("Time from episode start [s]")
plt.ylabel("Estimated force norm [N]")
plt.title(f"Zero-Start Estimated External Force Norm Overlay + Mean ({side})")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(png_path, dpi=300)

print(f"Saved CSV:  {csv_path}")
print(f"Saved plot: {png_path}")

plt.show()


# #!/usr/bin/env python3
# import pickle
# import csv
# from pathlib import Path

# import numpy as np
# import matplotlib.pyplot as plt
# import pinocchio as pin


# # ================================
# # 0) Settings
# # ================================
# data_dir = Path("/home/watanaberyuto/ws_backup/raw_data/exp0501_mix_0410_0430narrow")
# urdf_path = Path("/home/watanaberyuto/ws/src/factr_teleop-main/src/factr_teleop/factr_teleop/urdf/factr_teleop_franka.urdf")

# side = "right"   # "right" or "left"
# frame_name = "link_7"

# state_topic = f"/franka/{side}/obs_franka_state"
# torque_topic = f"/franka/{side}/obs_franka_torque"

# csv_path = Path(f"estimated_force_zero_overlay_{side}.csv")
# png_path = Path(f"estimated_force_zero_overlay_mean_{side}.png")


# # ================================
# # 1) Load robot model
# # ================================
# model = pin.buildModelFromUrdf(str(urdf_path))
# data = model.createData()

# frame_id = model.getFrameId(frame_name)
# if frame_id == len(model.frames):
#     print("Available frames:")
#     for f in model.frames:
#         print(f.name)
#     raise ValueError(f"Frame '{frame_name}' not found")


# def estimate_wrench(q, tau_ext):
#     q = np.asarray(q, dtype=float)
#     tau_ext = np.asarray(tau_ext, dtype=float)

#     pin.forwardKinematics(model, data, q)
#     pin.computeJointJacobians(model, data, q)
#     pin.updateFramePlacements(model, data)

#     J = pin.getFrameJacobian(
#         model,
#         data,
#         frame_id,
#         pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
#     )

#     wrench = np.linalg.pinv(J.T) @ tau_ext
#     return wrench


# # ================================
# # 2) Process episodes
# # ================================
# files = sorted(data_dir.glob("ep_*.pkl"))
# print("num episodes:", len(files))

# all_episode_times = []
# all_episode_force_norms = []

# plt.figure(figsize=(10, 6))

# with open(csv_path, "w", newline="") as f_csv:
#     writer = csv.writer(f_csv)
#     writer.writerow([
#         "episode", "time",
#         "Fx_zero", "Fy_zero", "Fz_zero",
#         "force_norm_zero",
#         "Mx_zero", "My_zero", "Mz_zero",
#     ])

#     for ep_idx, pkl_path in enumerate(files):
#         with open(pkl_path, "rb") as f:
#             d = pickle.load(f)

#         if state_topic not in d["data"] or torque_topic not in d["data"]:
#             print(f"skip {pkl_path.name}: topic not found")
#             continue

#         q_list = d["data"][state_topic]
#         tau_list = d["data"][torque_topic]
#         q_ts = d["timestamps"][state_topic]
#         tau_ts = d["timestamps"][torque_topic]

#         if len(q_list) == 0 or len(tau_list) == 0:
#             print(f"skip {pkl_path.name}: empty data")
#             continue

#         q_ts = np.asarray(q_ts, dtype=np.int64)
#         tau_ts = np.asarray(tau_ts, dtype=np.int64)

#         ep_t0 = tau_ts[0]

#         ep_t = []
#         ep_force_norm = []

#         wrench_offset = None

#         for i, tau_time in enumerate(tau_ts):
#             q_idx = np.argmin(np.abs(q_ts - tau_time))

#             q = np.asarray(q_list[q_idx], dtype=float)
#             tau_ext = np.asarray(tau_list[i], dtype=float)

#             wrench = estimate_wrench(q, tau_ext)

#             if wrench_offset is None:
#                 wrench_offset = wrench.copy()

#             wrench_zero = wrench - wrench_offset

#             Fx, Fy, Fz = wrench_zero[:3]
#             Mx, My, Mz = wrench_zero[3:]
#             force_norm = np.linalg.norm(wrench_zero[:3])

#             t = (tau_time - ep_t0) * 1e-9

#             writer.writerow([
#                 ep_idx, t,
#                 Fx, Fy, Fz,
#                 force_norm,
#                 Mx, My, Mz,
#             ])

#             ep_t.append(t)
#             ep_force_norm.append(force_norm)

#         if len(ep_t) > 0:
#             ep_t = np.array(ep_t)
#             ep_force_norm = np.array(ep_force_norm)

#             # 各episodeを薄く描画
#             plt.plot(
#                 ep_t,
#                 ep_force_norm,
#                 alpha=0.25,
#                 linewidth=1.0,
#                 color="gray"
#             )

#             all_episode_times.append(ep_t)
#             all_episode_force_norms.append(ep_force_norm)

#         print(f"processed {pkl_path.name}")


# # ================================
# # 3) Average Plot
# # ================================
# if len(all_episode_force_norms) > 0:

#     min_len = min(len(x) for x in all_episode_force_norms)

#     aligned_force = []

#     for f in all_episode_force_norms:

#         f = np.array(f[:min_len])

#         # 念のため平均前にゼロスタート化
#         f = f - f[0]

#         aligned_force.append(f)

#     aligned_force = np.array(aligned_force)

#     aligned_time = all_episode_times[0][:min_len]

#     mean_force = np.mean(aligned_force, axis=0)

#     plt.plot(
#         aligned_time,
#         mean_force,
#         linewidth=4,
#         color="red",
#         label="Mean force norm"
#     )

# # ================================
# # 4) Save Plot
# # ================================
# plt.xlabel("Time from episode start [s]")
# plt.ylabel("Estimated force norm [N]")
# plt.title(f"Zero-Start Estimated External Force Norm Overlay + Mean ({side})")
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig(png_path, dpi=300)

# print(f"Saved CSV:  {csv_path}")
# print(f"Saved plot: {png_path}")

# plt.show()
