import ctypes
import time
from ctypes import c_double

LIB = "/home/watanaberyuto/franka_ros2_ws/omega/omega_hatptic_device/omega_hatptic_device-main/lib/libdhd.so.3.14.0"

print("Loading:", LIB)

dhd = ctypes.CDLL(LIB)

# return type
dhd.dhdOpen.restype = ctypes.c_int
dhd.dhdClose.restype = ctypes.c_int
dhd.dhdGetDeviceCount.restype = ctypes.c_int
dhd.dhdGetPosition.restype = ctypes.c_int
dhd.dhdErrorGetLastStr.restype = ctypes.c_char_p
dhd.dhdEnableForce.restype = ctypes.c_int

# argument type
dhd.dhdEnableForce.argtypes = [ctypes.c_ubyte]
dhd.dhdGetPosition.argtypes = [
    ctypes.POINTER(c_double),
    ctypes.POINTER(c_double),
    ctypes.POINTER(c_double),
]

print("device count =", dhd.dhdGetDeviceCount())

ret = dhd.dhdOpen()
print("dhdOpen =", ret)

if ret < 0:
    print("DHD open error:", dhd.dhdErrorGetLastStr().decode())
    raise RuntimeError("Failed to open Omega")

# 力出力はOFFにして安全に位置だけ読む
ret = dhd.dhdEnableForce(0)
print("dhdEnableForce(0) =", ret)

x = c_double()
y = c_double()
z = c_double()

try:
    while True:
        ret = dhd.dhdGetPosition(
            ctypes.byref(x),
            ctypes.byref(y),
            ctypes.byref(z),
        )

        if ret < 0:
            print("getPosition error:", dhd.dhdErrorGetLastStr().decode())
        else:
            print(
                f"x={x.value:.4f} "
                f"y={y.value:.4f} "
                f"z={z.value:.4f} "
                f"ret={ret}"
            )

        time.sleep(0.05)

except KeyboardInterrupt:
    pass

finally:
    dhd.dhdClose()
    print("closed")
