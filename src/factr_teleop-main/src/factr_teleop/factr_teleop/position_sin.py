#!/usr/bin/env python3
import numpy as np
import time
import csv
import matplotlib.pyplot as plt

from python_utils.zmq_messenger import ZMQPublisher, ZMQSubscriber
from python_utils.global_configs import (
    franka_right_real_zmq_addresses
)

#==============================
# 実験パラメータ
#==============================
TARGET_JOINT = 3               # 4番目の関節を動作
AMPLITUDE = 0.3                # [rad]
FREQ = 0.2                     # [Hz]
DURATION = 20.0                # [s]
SEND_RATE = 100                # [Hz]

#==============================
# ZMQ setup
#==============================
# Franka へ指令角送信
cmd_pub = ZMQPublisher(franka_right_real_zmq_addresses["joint_pos_cmd_pub"])
# Franka 現在角度の購読
joint_state_sub = ZMQSubscriber(franka_right_real_zmq_addresses["joint_state_sub"])


# wait for data
print("Waiting for joint states...")
while joint_state_sub.message is None:
    time.sleep(0.01)

# 初期値
q0 = np.array(joint_state_sub.message)
print("Initial q:", q0)

#==============================
# ログ用バッファ
#==============================
log_time = []
log_q_cmd = []
log_q_meas = []

#==============================
# 実験開始
#==============================
print("=== Start joint tracking experiment (ZMQ impedance) ===")
start_time = time.time()

dt = 1.0 / SEND_RATE
next_send = time.time()

try:
    while True:
        t = time.time() - start_time
        if t > DURATION:
            break

        #----- 指令生成 -----
        q_cmd = q0.copy()
        q_cmd[TARGET_JOINT] = q0[TARGET_JOINT] + AMPLITUDE * np.sin(2*np.pi*FREQ * t)

        #----- Franka の現在関節角を取得 -----
        q_meas = np.array(joint_state_sub.message)

        #----- ZMQ で指令角送信 -----
        cmd_pub.send_message(q_cmd)


        # ----- ログ -----
        log_time.append(t)

        # 指令・実測ともに初期姿勢を0として正規化
        log_q_cmd.append(q_cmd[TARGET_JOINT] - q0[TARGET_JOINT])
        log_q_meas.append(q_meas[TARGET_JOINT] - q0[TARGET_JOINT])

        # 整合した周期で送信
        next_send += dt
        sleep_t = next_send - time.time()
        if sleep_t > 0:
            time.sleep(sleep_t)

except KeyboardInterrupt:
    print("Interrupted.")

#==============================
# CSV 保存
#==============================
csv_path = "joint_tracking_log.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "q_cmd", "q_meas"])
    for t, qc, qm in zip(log_time, log_q_cmd, log_q_meas):
        writer.writerow([t, qc, qm])

print("Saved CSV:", csv_path)

#==============================
# プロット
#==============================
plt.figure(figsize=(10,5))
plt.plot(log_time, log_q_cmd, label="Command")
plt.plot(log_time, log_q_meas, label="Measured")
plt.xlabel("Time [s]")
plt.ylabel(f"Joint {TARGET_JOINT} angle [rad]")
plt.legend()
plt.grid()
plt.title("Joint Tracking (ZMQ impedance controller)")
plt.show()


