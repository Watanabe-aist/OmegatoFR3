#!/usr/bin/env python3
import time
import csv
import numpy as np
import matplotlib.pyplot as plt

from pylibfranka import Robot

# ============================
# パラメータ
# ============================
IP = "192.168.2.122"
DURATION = 20.0   # 計測時間 [s]
CALIB_SAMPLES = 300   # ゼロ点補正用サンプル数
CSV_PATH = "force_log.csv"

# ============================
# メイン処理
# ============================
def main():

    print("=== Connecting to Franka ===")
    robot = Robot(IP)

    # ======================
    # 1) ゼロ点補正（オフセット計算）
    # ======================
    print("=== Calibrating offset... ===")

    offset_list = []

    for _ in range(CALIB_SAMPLES):
        state = robot.read_once()
        F_raw = np.array(state.K_F_ext_hat_K)[0:3]
        offset_list.append(F_raw)
        time.sleep(0.005)

    offset = np.mean(np.vstack(offset_list), axis=0)
    print("Offset (Fx, Fy, Fz):", offset)

    # ======================
    # 2) 計測フェーズ
    # ======================
    print("=== Start measuring external force ===")

    t0 = time.time()
    log_t = []
    log_fx = []
    log_fy = []
    log_fz = []

    while True:
        t = time.time() - t0
        if t > DURATION:
            break

        state = robot.read_once()
        F_raw = np.array(state.K_F_ext_hat_K)[0:3]

        # オフセット補正後
        F = F_raw - offset

        log_t.append(t)
        log_fx.append(F[0])
        log_fy.append(F[1])
        log_fz.append(F[2])

        time.sleep(0.005)

    # ======================
    # 3) CSV 保存
    # ======================
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "Fx", "Fy", "Fz"])
        for t, fx, fy, fz in zip(log_t, log_fx, log_fy, log_fz):
            writer.writerow([t, fx, fy, fz])

    print(f"Saved CSV → {CSV_PATH}")

    # ======================
    # 4) プロット
    # ======================
    plt.figure(figsize=(10, 7))
    plt.subplot(3,1,1)
    plt.plot(log_t, log_fx); plt.ylabel("Fx [N]"); plt.grid()
    plt.subplot(3,1,2)
    plt.plot(log_t, log_fy); plt.ylabel("Fy [N]"); plt.grid()
    plt.subplot(3,1,3)
    plt.plot(log_t, log_fz); plt.ylabel("Fz [N]"); plt.xlabel("Time [s]"); plt.grid()
    plt.suptitle("External Force (offset corrected)")
    plt.show()

    print("=== Finished ===")

if __name__ == "__main__":
    main()

