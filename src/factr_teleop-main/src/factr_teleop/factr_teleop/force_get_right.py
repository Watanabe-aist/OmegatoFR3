#APIからエンドエフェクタ先の力を取得して保存
# #!/usr/bin/env python3
# import time
# import csv
# from pathlib import Path
# import matplotlib.pyplot as plt

# from python_utils.zmq_messenger import ZMQSubscriber
# from python_utils.global_configs import franka_right_real_zmq_addresses


# # ================================
# # 0) Save paths
# # ================================
# save_dir = Path(".")
# csv_path = save_dir / "force_log_right.csv"
# png_path = save_dir / "force_log_right.png"


# # ================================
# # 1) Subscribe
# # ================================
# force_sub = ZMQSubscriber(franka_right_real_zmq_addresses["force_ext_pub"])

# print(f"Listening on: {franka_right_real_zmq_addresses['force_ext_pub']}")
# print("Waiting for first message to calibrate offset...")


# # ================================
# # 2) Wait first message for ZERO offset
# # ================================
# offset = None
# while True:
#     F = force_sub.message
#     if F is not None:
#         Fx0, Fy0, Fz0 = F[:3]
#         offset = (Fx0, Fy0, Fz0)
#         print(f"Offset calibrated: Fx={Fx0:.3f}, Fy={Fy0:.3f}, Fz={Fz0:.3f}\n")
#         break
#     time.sleep(0.005)


# # ================================
# # 3) Buffers
# # ================================
# time_list = []
# Fx_list, Fy_list, Fz_list = [], [], []


# # ================================
# # 4) CSV
# # ================================
# csv_file = open(csv_path, "w", newline="")
# writer = csv.writer(csv_file)
# writer.writerow(["time", "Fx", "Fy", "Fz"])

# t0 = time.time()

# print("Start logging... Press Ctrl+C to stop.\n")


# # ================================
# # 5) Main loop
# # ================================
# try:
#     while True:
#         F = force_sub.message
#         if F is None:
#             time.sleep(0.005)
#             continue

#         Fx_raw, Fy_raw, Fz_raw = F[:3]

#         # Zero-start correction
#         Fx = Fx_raw - offset[0]
#         Fy = Fy_raw - offset[1]
#         Fz = Fz_raw - offset[2]

#         t = time.time() - t0

#         # Save CSV
#         writer.writerow([t, Fx, Fy, Fz])

#         # Store for plotting
#         time_list.append(t)
#         Fx_list.append(Fx)
#         Fy_list.append(Fy)
#         Fz_list.append(Fz)

#         print(f"[{t:.2f}s] Fx={Fx:.2f}, Fy={Fy:.2f}, Fz={Fz:.2f}")

#         time.sleep(0.01)

# except KeyboardInterrupt:
#     print("\nStopping logging...")

# finally:
#     csv_file.close()

#     # ================================
#     # 6) Save Plot
#     # ================================
#     if len(time_list) > 0:
#         plt.figure(figsize=(10, 6))
#         plt.plot(time_list, Fx_list, label="Fx")
#         plt.plot(time_list, Fy_list, label="Fy")
#         plt.plot(time_list, Fz_list, label="Fz")

#         plt.xlabel("Time [s]")
#         plt.ylabel("Force [N]")
#         plt.title("Franka External Right Force (Zero-Start, Fx/Fy/Fz)")
#         plt.grid()
#         plt.legend()
#         plt.tight_layout()

#         plt.savefig(png_path, dpi=300)
#         print(f"Saved CSV:  {csv_path}")
#         print(f"Saved plot: {png_path}")

#         plt.show()
#     else:
#         print("No data recorded, so no plot was saved.")


#各関節トルクから把持部分の力を推定
#!/usr/bin/env python3
# import time
# import csv
# from pathlib import Path

# import numpy as np
# import matplotlib.pyplot as plt
# import pinocchio as pin

# from python_utils.zmq_messenger import ZMQSubscriber
# from python_utils.global_configs import franka_right_real_zmq_addresses


# # ================================
# # 0) Settings
# # ================================
# side = "right"

# urdf_path = Path(
#     "/home/watanaberyuto/ws/src/factr_teleop-main/src/factr_teleop/factr_teleop/urdf/factr_teleop_franka.urdf"
# )

# frame_name = "link_7"

# csv_path = Path(f"estimated_force_from_torque_{side}.csv")
# png_path = Path(f"estimated_force_from_torque_{side}.png")


# # ================================
# # 1) ZMQ Subscribe
# # ================================
# zmq_addresses = franka_right_real_zmq_addresses

# q_sub = ZMQSubscriber(zmq_addresses["joint_state_sub"])
# tau_sub = ZMQSubscriber(zmq_addresses["joint_torque_sub"])

# print("Waiting for joint state and external joint torque...")


# # ================================
# # 2) Load Pinocchio model
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
# # 3) Wait first valid data and offset
# # ================================
# offset = None

# while True:
#     q = q_sub.message
#     tau_ext = tau_sub.message

#     if q is not None and tau_ext is not None:
#         q = np.asarray(q[:7], dtype=float)
#         tau_ext = np.asarray(tau_ext[:7], dtype=float)

#         wrench0 = estimate_wrench(q, tau_ext)
#         offset = wrench0.copy()

#         print("Offset calibrated:")
#         print(f"Fx={offset[0]:.3f}, Fy={offset[1]:.3f}, Fz={offset[2]:.3f}")
#         break

#     time.sleep(0.005)


# # ================================
# # 4) Buffers
# # ================================
# time_list = []
# Fx_list, Fy_list, Fz_list = [], [], []
# force_norm_list = []


# # ================================
# # 5) CSV
# # ================================
# csv_file = open(csv_path, "w", newline="")
# writer = csv.writer(csv_file)
# writer.writerow(["time", "Fx", "Fy", "Fz", "force_norm", "Mx", "My", "Mz"])

# t0 = time.time()

# print("Start logging estimated force from joint torque... Press Ctrl+C to stop.")


# # ================================
# # 6) Main loop
# # ================================
# try:
#     while True:
#         q = q_sub.message
#         tau_ext = tau_sub.message

#         if q is None or tau_ext is None:
#             time.sleep(0.005)
#             continue

#         q = np.asarray(q[:7], dtype=float)
#         tau_ext = np.asarray(tau_ext[:7], dtype=float)

#         wrench = estimate_wrench(q, tau_ext)
#         wrench_zero = wrench - offset

#         Fx, Fy, Fz = wrench_zero[:3]
#         Mx, My, Mz = wrench_zero[3:]

#         force_norm = np.linalg.norm(wrench_zero[:3])

#         t = time.time() - t0

#         writer.writerow([t, Fx, Fy, Fz, force_norm, Mx, My, Mz])

#         time_list.append(t)
#         Fx_list.append(Fx)
#         Fy_list.append(Fy)
#         Fz_list.append(Fz)
#         force_norm_list.append(force_norm)

#         print(
#             f"[{t:.2f}s] "
#             f"Fx={Fx:.2f}, Fy={Fy:.2f}, Fz={Fz:.2f}, "
#             f"|F|={force_norm:.2f}"
#         )

#         time.sleep(0.01)

# except KeyboardInterrupt:
#     print("\nStopping logging...")

# finally:
#     csv_file.close()

#     if len(time_list) > 0:
#         plt.figure(figsize=(10, 6))

#         plt.plot(time_list, Fx_list, label="Fx")
#         plt.plot(time_list, Fy_list, label="Fy")
#         plt.plot(time_list, Fz_list, label="Fz")
#         plt.plot(time_list, force_norm_list, label="Force norm")

#         plt.xlabel("Time [s]")
#         plt.ylabel("Estimated force [N]")
#         plt.title(f"Estimated External Force from Joint Torque ({side})")
#         plt.grid()
#         plt.legend()
#         plt.tight_layout()

#         plt.savefig(png_path, dpi=300)

#         print(f"Saved CSV:  {csv_path}")
#         print(f"Saved plot: {png_path}")

#         plt.show()
#     else:
#         print("No data recorded.")

# #各関節のトルクを計測し別々に保存
# #!/usr/bin/env python3
# import time
# import csv
# from pathlib import Path

# import numpy as np
# import matplotlib.pyplot as plt

# from python_utils.zmq_messenger import ZMQSubscriber
# from python_utils.global_configs import (
#     franka_left_real_zmq_addresses,
#     franka_right_real_zmq_addresses,
# )


# # ================================
# # 0) Settings
# # ================================
# side = "right"   # "left" or "right"

# if side == "left":
#     zmq_addresses = franka_left_real_zmq_addresses
# elif side == "right":
#     zmq_addresses = franka_right_real_zmq_addresses
# else:
#     raise ValueError("side must be 'left' or 'right'")

# csv_path = Path(f"rollout_joint_torque_{side}.csv")
# png_path = Path(f"rollout_joint_torque_{side}.png")


# # ================================
# # 1) Subscribe
# # ================================
# tau_sub = ZMQSubscriber(zmq_addresses["joint_torque_sub"])

# print(f"Listening joint torque: {zmq_addresses['joint_torque_sub']}")
# print("Start logging rollout joint torques... Press Ctrl+C to stop.")


# # ================================
# # 2) Buffers
# # ================================
# time_list = []
# tau_lists = [[] for _ in range(7)]


# # ================================
# # 3) CSV
# # ================================
# csv_file = open(csv_path, "w", newline="")
# writer = csv.writer(csv_file)
# writer.writerow(["time", "tau1", "tau2", "tau3", "tau4", "tau5", "tau6", "tau7"])

# t0 = time.time()


# # ================================
# # 4) Main loop
# # ================================
# try:
#     while True:
#         tau = tau_sub.message

#         if tau is None:
#             time.sleep(0.005)
#             continue

#         tau = np.asarray(tau[:7], dtype=float)
#         t = time.time() - t0

#         writer.writerow([t, *tau.tolist()])

#         time_list.append(t)
#         for j in range(7):
#             tau_lists[j].append(tau[j])

#         print(
#             f"[{t:.2f}s] "
#             + " ".join([f"tau{j+1}={tau[j]:.3f}" for j in range(7)])
#         )

#         time.sleep(0.01)

# except KeyboardInterrupt:
#     print("\nStopping logging...")

# finally:
#     csv_file.close()

#     if len(time_list) > 0:
#         plt.figure(figsize=(12, 7))

#         for j in range(7):
#             plt.plot(time_list, tau_lists[j], label=f"tau{j+1}")

#         plt.xlabel("Time [s]")
#         plt.ylabel("Joint torque [Nm]")
#         plt.title(f"Rollout Joint Torque ({side})")
#         plt.grid()
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(png_path, dpi=300)

#         print(f"Saved CSV:  {csv_path}")
#         print(f"Saved plot: {png_path}")

#         plt.show()
#     else:
#         print("No data recorded.")


#複合
#!/usr/bin/env python3
import time
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin

from python_utils.zmq_messenger import ZMQSubscriber
from python_utils.global_configs import (
    franka_left_real_zmq_addresses,
    franka_right_real_zmq_addresses,
)


# ================================
# 0) Settings
# ================================
side = "right"   # "left" or "right"

urdf_path = Path(
    "/home/watanaberyuto/ws/src/factr_teleop-main/src/factr_teleop/factr_teleop/urdf/factr_teleop_franka.urdf"
)

frame_name = "link_7"

if side == "left":
    zmq_addresses = franka_left_real_zmq_addresses
elif side == "right":
    zmq_addresses = franka_right_real_zmq_addresses
else:
    raise ValueError("side must be 'left' or 'right'")

csv_path = Path(f"rollout_state_torque_force_{side}.csv")
png_tau_path = Path(f"rollout_joint_torque_{side}.png")
png_force_path = Path(f"rollout_estimated_force_norm_{side}.png")


# ================================
# 1) Subscribe
# ================================
q_sub = ZMQSubscriber(zmq_addresses["joint_state_sub"])
tau_sub = ZMQSubscriber(zmq_addresses["joint_torque_sub"])

print(f"Listening joint state:  {zmq_addresses['joint_state_sub']}")
print(f"Listening joint torque: {zmq_addresses['joint_torque_sub']}")
print("Waiting for first q and tau...")


# ================================
# 2) Load Pinocchio model
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
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )

    wrench = np.linalg.pinv(J.T) @ tau_ext
    return wrench


# ================================
# 3) Wait first valid data
# ================================
while True:
    q_msg = q_sub.message
    tau_msg = tau_sub.message

    if q_msg is not None and tau_msg is not None:
        print("Received first q and tau. Start logging.")
        break

    time.sleep(0.005)


# ================================
# 4) Buffers
# ================================
time_list = []

q_lists = [[] for _ in range(7)]
tau_lists = [[] for _ in range(7)]

Fx_list, Fy_list, Fz_list = [], [], []
Mx_list, My_list, Mz_list = [], [], []
force_norm_list = []


# ================================
# 5) CSV
# ================================
csv_file = open(csv_path, "w", newline="")
writer = csv.writer(csv_file)

writer.writerow([
    "time",
    "q1", "q2", "q3", "q4", "q5", "q6", "q7",
    "tau1", "tau2", "tau3", "tau4", "tau5", "tau6", "tau7",
    "Fx", "Fy", "Fz", "force_norm",
    "Mx", "My", "Mz",
])

t0 = time.time()

print("Start logging rollout q, tau, estimated force... Press Ctrl+C to stop.")


# ================================
# 6) Main loop
# ================================
try:
    while True:
        q_msg = q_sub.message
        tau_msg = tau_sub.message

        if q_msg is None or tau_msg is None:
            time.sleep(0.005)
            continue

        q = np.asarray(q_msg[:7], dtype=float)
        tau = np.asarray(tau_msg[:7], dtype=float)

        wrench = estimate_wrench(q, tau)

        Fx, Fy, Fz = wrench[:3]
        Mx, My, Mz = wrench[3:]
        force_norm = np.linalg.norm(wrench[:3])

        t = time.time() - t0

        writer.writerow([
            t,
            *q.tolist(),
            *tau.tolist(),
            Fx, Fy, Fz, force_norm,
            Mx, My, Mz,
        ])

        time_list.append(t)

        for j in range(7):
            q_lists[j].append(q[j])
            tau_lists[j].append(tau[j])

        Fx_list.append(Fx)
        Fy_list.append(Fy)
        Fz_list.append(Fz)
        Mx_list.append(Mx)
        My_list.append(My)
        Mz_list.append(Mz)
        force_norm_list.append(force_norm)

        print(
            f"[{t:.2f}s] "
            + " ".join([f"tau{j+1}={tau[j]:.3f}" for j in range(7)])
            + f" | |F|={force_norm:.2f}"
        )

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopping logging...")

finally:
    csv_file.close()

    if len(time_list) > 0:
        # ================================
        # 7) Plot joint torques
        # ================================
        plt.figure(figsize=(12, 7))
        for j in range(7):
            plt.plot(time_list, tau_lists[j], label=f"tau{j+1}")

        plt.xlabel("Time [s]")
        plt.ylabel("Joint torque [Nm]")
        plt.title(f"Rollout Joint Torque ({side})")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_tau_path, dpi=300)

        # ================================
        # 8) Plot estimated force norm
        # ================================
        plt.figure(figsize=(10, 5))
        plt.plot(time_list, force_norm_list, linewidth=2, label="Estimated force norm")

        plt.xlabel("Time [s]")
        plt.ylabel("Estimated force norm [N]")
        plt.title(f"Rollout Estimated Force Norm from Joint Torque ({side})")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_force_path, dpi=300)

        print(f"Saved CSV:        {csv_path}")
        print(f"Saved tau plot:   {png_tau_path}")
        print(f"Saved force plot: {png_force_path}")

        plt.show()
    else:
        print("No data recorded.")