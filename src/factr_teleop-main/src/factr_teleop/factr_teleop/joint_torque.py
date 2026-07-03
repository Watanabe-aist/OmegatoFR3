#!/usr/bin/env python3
import pickle
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ================================
# 0) Settings
# ================================
data_dir = Path("/home/watanaberyuto/ws_backup/raw_data/exp0430_25times_narrow_box")

side = "right"   # "right" or "left"
torque_topic = f"/franka/{side}/obs_franka_torque"

csv_path = Path(f"joint_torque_overlay_{side}.csv")
png_path = Path(f"joint_torque_overlay_mean_{side}.png")


# ================================
# 1) Process episodes
# ================================
files = sorted(data_dir.glob("ep_*.pkl"))
print("num episodes:", len(files))

all_episode_times = []
all_episode_torques = []

plt.figure(figsize=(12, 7))

with open(csv_path, "w", newline="") as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow([
        "episode", "time",
        "tau1", "tau2", "tau3", "tau4", "tau5", "tau6", "tau7"
    ])

    for ep_idx, pkl_path in enumerate(files):
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)

        if torque_topic not in d["data"]:
            print(f"skip {pkl_path.name}: topic not found")
            continue

        tau_list = d["data"][torque_topic]
        tau_ts = d["timestamps"][torque_topic]

        if len(tau_list) == 0:
            print(f"skip {pkl_path.name}: empty torque data")
            continue

        tau_ts = np.asarray(tau_ts, dtype=np.int64)
        tau_array = np.asarray(tau_list, dtype=float)  # shape = [T, 7]

        ep_t0 = tau_ts[0]
        ep_t = (tau_ts - ep_t0) * 1e-9

        for t, tau in zip(ep_t, tau_array):
            writer.writerow([ep_idx, t, *tau.tolist()])

        # 各episodeを薄く描画
        for j in range(7):
            plt.plot(
                ep_t,
                tau_array[:, j],
                alpha=0.15,
                linewidth=0.8,
                color="gray"
            )

        all_episode_times.append(ep_t)
        all_episode_torques.append(tau_array)

        print(f"processed {pkl_path.name}")


# ================================
# 2) Average Plot for each joint
# ================================
if len(all_episode_torques) > 0:
    min_len = min(len(x) for x in all_episode_torques)

    aligned_torques = np.array([
        x[:min_len] for x in all_episode_torques
    ])  # shape = [N_episode, T, 7]

    aligned_time = all_episode_times[0][:min_len]

    mean_torque = np.mean(aligned_torques, axis=0)  # [T, 7]
    std_torque = np.std(aligned_torques, axis=0)    # [T, 7]

    for j in range(7):
        plt.plot(
            aligned_time,
            mean_torque[:, j],
            linewidth=2.5,
            label=f"Mean tau{j+1}"
        )

        # 標準偏差も見たい場合はコメント解除
        # plt.fill_between(
        #     aligned_time,
        #     mean_torque[:, j] - std_torque[:, j],
        #     mean_torque[:, j] + std_torque[:, j],
        #     alpha=0.15
        # )


# ================================
# 3) Save Plot
# ================================
plt.xlabel("Time from episode start [s]")
plt.ylabel("Joint torque [Nm]")
plt.title(f"Measured Joint Torques Overlay + Mean ({side})")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(png_path, dpi=300)

print(f"Saved CSV:  {csv_path}")
print(f"Saved plot: {png_path}")

plt.show()