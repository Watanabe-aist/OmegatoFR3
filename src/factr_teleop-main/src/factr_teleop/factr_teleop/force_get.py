#!/usr/bin/env python3
import time
import csv
import math
import matplotlib.pyplot as plt
from python_utils.zmq_messenger import ZMQSubscriber
from python_utils.global_configs import franka_right_real_zmq_addresses


# ================================
# Parameters
# ================================
WARMUP_TIME = 20.0       # [s] 起動後の安定待ち時間
LOG_RATE = 100           # [Hz]


# ================================
# ZMQ Subscribe
# ================================
force_sub = ZMQSubscriber(
    franka_right_real_zmq_addresses["force_ext_pub"]
)

print(f"Listening on: {franka_right_real_zmq_addresses['force_ext_pub']}")
print(f"System warm-up for {WARMUP_TIME} seconds...")


# ================================
# 1) Warm-up phase (DO NOTHING)
# ================================
t_start = time.time()
while time.time() - t_start < WARMUP_TIME:
    time.sleep(0.01)

print("Warm-up finished. Start logging.\n")


# ================================
# 2) Buffers
# ================================
time_list = []
F_norm_list = []


# ================================
# 3) CSV setup
# ================================
csv_path = "force_norm_log_right.csv"
csv_file = open(csv_path, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["time", "Fx", "Fy", "Fz", "F_norm"])

t0 = time.time()


# ================================
# 4) Main logging loop
# ================================
try:
    while True:
        F = force_sub.message
        if F is None:
            time.sleep(0.005)
            continue

        Fx, Fy, Fz = F[:3]

        # Force norm
        F_norm = math.sqrt(Fx**2 + Fy**2 + Fz**2)

        t = time.time() - t0

        # CSV
        writer.writerow([t, Fx, Fy, Fz, F_norm])

        # Buffer
        time_list.append(t)
        F_norm_list.append(F_norm)

        print(f"[{t:6.2f}s] |F| = {F_norm:6.3f} N")

        time.sleep(1.0 / LOG_RATE)

except KeyboardInterrupt:
    print("\nStopping logging...")
    csv_file.close()


# ================================
# 5) Plot
# ================================
plt.figure(figsize=(10, 5))
plt.plot(time_list, F_norm_list, label=r"$\|F\|$")
plt.xlabel("Time [s]")
plt.ylabel("Force Norm [N]")
plt.title("External Force Norm (after warm-up)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
