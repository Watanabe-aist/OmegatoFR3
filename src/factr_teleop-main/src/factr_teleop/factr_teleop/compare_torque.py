#!/usr/bin/env python3
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


# ================================
# 0) Settings
# ================================
side = "right"   # "left" or "right"
x_max = 8.0

dataset_torque_csv = Path(f"joint_torque_overlay_{side}.csv")
dataset_force_csv = Path(f"estimated_force_zero_overlay_{side}.csv")
rollout_csv = Path(f"rollout_state_torque_force_{side}.csv")

output_dir = Path(f"compare_rollout_dataset_{side}")
output_dir.mkdir(exist_ok=True, parents=True)


# ================================
# Utility
# ================================
def load_episode_csv(csv_path, value_columns):
    episodes = defaultdict(list)

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row["episode"])
            t = float(row["time"])
            values = [float(row[col]) for col in value_columns]
            episodes[ep].append([t, *values])

    times = []
    values_list = []

    for ep, rows in episodes.items():
        arr = np.asarray(rows, dtype=float)
        times.append(arr[:, 0])
        values_list.append(arr[:, 1:])

    min_len = min(len(x) for x in values_list)

    aligned_values = np.array([x[:min_len] for x in values_list])
    aligned_time = times[0][:min_len]

    mean_values = np.mean(aligned_values, axis=0)

    return aligned_time, aligned_values, mean_values


def load_rollout_csv(csv_path, value_columns):
    rows = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["time"])
            values = [float(row[col]) for col in value_columns]
            rows.append([t, *values])

    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1:]


def zero_start_array(arr):
    """
    arr shape:
      [T] or [N, T] or [N, T, 1]
    最初の値を引いてゼロスタートにする
    """
    arr = np.asarray(arr, dtype=float)

    if arr.ndim == 1:
        return arr - arr[0]

    if arr.ndim == 2:
        return arr - arr[:, [0]]

    if arr.ndim == 3:
        return arr - arr[:, [0], :]

    raise ValueError(f"Unsupported shape: {arr.shape}")


# ================================
# 1) Load torque data
# ================================
tau_cols = [f"tau{i}" for i in range(1, 8)]

dataset_t, dataset_tau_all, dataset_tau_mean = load_episode_csv(
    dataset_torque_csv,
    tau_cols
)

rollout_t, rollout_tau = load_rollout_csv(
    rollout_csv,
    tau_cols
)


# ================================
# 2) Load force norm data
# ================================
dataset_force_t, dataset_force_all, dataset_force_mean = load_episode_csv(
    dataset_force_csv,
    ["force_norm_zero"]
)

rollout_force_t, rollout_force = load_rollout_csv(
    rollout_csv,
    ["force_norm"]
)

# force normだけゼロスタート化
dataset_force_all = zero_start_array(dataset_force_all)
dataset_force_mean = np.mean(dataset_force_all, axis=0)

rollout_force = zero_start_array(rollout_force[:, 0])  # [T]


# ================================
# 3) Plot joint torque comparison
# ================================
for j in range(7):
    plt.figure(figsize=(10, 5))

    # dataset 50回分
    for ep_idx in range(dataset_tau_all.shape[0]):
        plt.plot(
            dataset_t,
            dataset_tau_all[ep_idx, :, j],
            color="gray",
            alpha=0.15,
            linewidth=0.7
        )

    # dataset平均
    plt.plot(
        dataset_t,
        dataset_tau_mean[:, j],
        color="blue",
        linewidth=3,
        label=f"Dataset mean tau{j+1}"
    )

    # rollout
    plt.plot(
        rollout_t,
        rollout_tau[:, j],
        color="red",
        linewidth=2,
        label=f"Rollout tau{j+1}"
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Joint torque [Nm]")
    plt.title(f"Joint {j+1} Torque Comparison ({side})")
    plt.xlim(0, x_max)
    plt.grid()
    plt.legend()
    plt.tight_layout()

    png_path = output_dir / f"compare_tau{j+1}_{side}.png"
    plt.savefig(png_path, dpi=300)
    plt.close()

    print(f"Saved: {png_path}")


# ================================
# 4) Plot all joint torques in one figure
# ================================
plt.figure(figsize=(14, 10))

for j in range(7):
    plt.subplot(4, 2, j + 1)

    for ep_idx in range(dataset_tau_all.shape[0]):
        plt.plot(
            dataset_t,
            dataset_tau_all[ep_idx, :, j],
            color="gray",
            alpha=0.12,
            linewidth=0.5
        )

    plt.plot(
        dataset_t,
        dataset_tau_mean[:, j],
        color="blue",
        linewidth=2,
        label="Dataset mean"
    )

    plt.plot(
        rollout_t,
        rollout_tau[:, j],
        color="red",
        linewidth=1.5,
        label="Rollout"
    )

    plt.title(f"tau{j+1}")
    plt.xlabel("Time [s]")
    plt.ylabel("Nm")
    plt.xlim(0, x_max)
    plt.grid()

    if j == 0:
        plt.legend()

plt.tight_layout()
all_tau_png = output_dir / f"compare_all_joint_torques_{side}.png"
plt.savefig(all_tau_png, dpi=300)
plt.close()

print(f"Saved: {all_tau_png}")


# ================================
# 5) Plot force norm comparison
# ================================
plt.figure(figsize=(10, 5))

# dataset 50回分 force norm
for ep_idx in range(dataset_force_all.shape[0]):
    plt.plot(
        dataset_force_t,
        dataset_force_all[ep_idx, :, 0],
        color="gray",
        alpha=0.15,
        linewidth=0.7
    )

# dataset平均 force norm
plt.plot(
    dataset_force_t,
    dataset_force_mean[:, 0],
    color="blue",
    linewidth=3,
    label="Dataset mean force norm"
)

# rollout force norm
plt.plot(
    rollout_force_t,
    rollout_force,
    color="red",
    linewidth=2,
    label="Rollout force norm"
)

plt.xlabel("Time [s]")
plt.ylabel("Estimated force norm [N]")
plt.title(f"Zero-Start Estimated Force Norm Comparison ({side})")
plt.xlim(0, x_max)
plt.grid()
plt.legend()
plt.tight_layout()

force_png = output_dir / f"compare_force_norm_zero_start_{side}.png"
plt.savefig(force_png, dpi=300)
plt.close()

print(f"Saved: {force_png}")

print("\nDone.")
print(f"Output directory: {output_dir}")


# #!/usr/bin/env python3
# import csv
# from pathlib import Path
# from collections import defaultdict

# import numpy as np
# import matplotlib.pyplot as plt


# # ================================
# # 0) Settings
# # ================================
# side = "left"   # "left" or "right"

# dataset_csv = Path(f"joint_torque_overlay_{side}.csv")
# rollout_csv = Path(f"rollout_joint_torque_{side}.csv")

# output_dir = Path(f"compare_joint_torque_{side}")
# output_dir.mkdir(exist_ok=True, parents=True)


# # ================================
# # 1) Load dataset CSV
# # ================================
# episodes = defaultdict(list)

# with open(dataset_csv, "r") as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         ep = int(row["episode"])
#         t = float(row["time"])
#         tau = [float(row[f"tau{j}"]) for j in range(1, 8)]
#         episodes[ep].append([t, *tau])

# dataset_times = []
# dataset_torques = []

# for ep, rows in episodes.items():
#     arr = np.asarray(rows, dtype=float)
#     dataset_times.append(arr[:, 0])
#     dataset_torques.append(arr[:, 1:8])

# min_len = min(len(x) for x in dataset_torques)

# aligned_dataset_torque = np.array([
#     x[:min_len] for x in dataset_torques
# ])  # [N_episode, T, 7]

# dataset_time = dataset_times[0][:min_len]
# dataset_mean = np.mean(aligned_dataset_torque, axis=0)
# dataset_std = np.std(aligned_dataset_torque, axis=0)


# # ================================
# # 2) Load rollout CSV
# # ================================
# rollout_rows = []

# with open(rollout_csv, "r") as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         t = float(row["time"])
#         tau = [float(row[f"tau{j}"]) for j in range(1, 8)]
#         rollout_rows.append([t, *tau])

# rollout_arr = np.asarray(rollout_rows, dtype=float)
# rollout_time = rollout_arr[:, 0]
# rollout_torque = rollout_arr[:, 1:8]


# # ================================
# # 3) Plot each joint separately
# # ================================
# for j in range(7):
#     plt.figure(figsize=(10, 5))

#     # 50回分の各episodeを薄く表示
#     for ep_idx in range(aligned_dataset_torque.shape[0]):
#         plt.plot(
#             dataset_time,
#             aligned_dataset_torque[ep_idx, :, j],
#             color="gray",
#             alpha=0.15,
#             linewidth=0.7
#         )

#     # dataset平均
#     plt.plot(
#         dataset_time,
#         dataset_mean[:, j],
#         linewidth=3,
#         color="blue",
#         label=f"Dataset mean tau{j+1}"
#     )

#     # dataset ±1 std
#     plt.fill_between(
#         dataset_time,
#         dataset_mean[:, j] - dataset_std[:, j],
#         dataset_mean[:, j] + dataset_std[:, j],
#         color="blue",
#         alpha=0.2,
#         label="Dataset ±1 std"
#     )

#     # rollout
#     plt.plot(
#         rollout_time,
#         rollout_torque[:, j],
#         linewidth=2,
#         color="red",
#         label=f"Rollout tau{j+1}"
#     )

#     plt.xlabel("Time [s]")
#     plt.ylabel("Joint torque [Nm]")
#     plt.title(f"Joint {j+1} Torque Comparison ({side})")
#     plt.xlim(0, 8)
#     plt.grid()
#     plt.legend()
#     plt.tight_layout()

#     png_path = output_dir / f"compare_tau{j+1}_{side}.png"
#     plt.savefig(png_path, dpi=300)
#     plt.close()

#     print(f"Saved: {png_path}")


# # ================================
# # 4) All joints in one figure
# # ================================
# plt.figure(figsize=(14, 10))

# for j in range(7):
#     plt.subplot(4, 2, j + 1)

#     # 50回分の各episodeを薄く表示
#     for ep_idx in range(aligned_dataset_torque.shape[0]):
#         plt.plot(
#             dataset_time,
#             aligned_dataset_torque[ep_idx, :, j],
#             color="gray",
#             alpha=0.12,
#             linewidth=0.5
#         )

#     plt.plot(
#         dataset_time,
#         dataset_mean[:, j],
#         linewidth=2,
#         color="blue",
#         label="Dataset mean"
#     )

#     plt.plot(
#         rollout_time,
#         rollout_torque[:, j],
#         linewidth=1.5,
#         color="red",
#         label="Rollout"
#     )

#     plt.title(f"tau{j+1}")
#     plt.xlabel("Time [s]")
#     plt.ylabel("Nm")
#     plt.xlim(0, 8)
#     plt.grid()

#     if j == 0:
#         plt.legend()
# plt.tight_layout()
# all_png = output_dir / f"compare_all_joints_{side}.png"
# plt.savefig(all_png, dpi=300)
# plt.show()

# print(f"Saved: {all_png}")