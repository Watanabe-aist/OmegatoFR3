#!/usr/bin/env python3
import numpy as np
import time

from python_utils.zmq_messenger import ZMQPublisher, ZMQSubscriber
from python_utils.global_configs import franka_right_real_zmq_addresses


cmd_pub = ZMQPublisher(franka_right_real_zmq_addresses["joint_pos_cmd_pub"])
joint_state_sub = ZMQSubscriber(franka_right_real_zmq_addresses["joint_state_sub"])

print("Waiting for joint state...")

q0 = None
while q0 is None:
    msg = joint_state_sub.message
    if msg is not None:
        q = np.array(msg[:7], dtype=float)

        if q.shape == (7,) and np.all(np.isfinite(q)):
            q0 = q.copy()
            print("Hold position:", q0)
            break

        print("Invalid joint state:", q)

    time.sleep(0.01)

print("Start holding... Ctrl+C to stop.")

try:
    while True:
        cmd_pub.send_message(q0)
        time.sleep(0.01)  # 100Hzくらいで送る
except KeyboardInterrupt:
    print("\nStopped.")