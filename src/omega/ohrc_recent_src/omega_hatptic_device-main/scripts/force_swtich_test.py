#!/usr/bin/env python
# license removed for brevity
from ftplib import ftpcp
from time import time
import rospy
from geometry_msgs.msg import Wrench
import math


def talker():
    pub = rospy.Publisher('/omega_driver/right/cmd_force', Wrench, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(2000) # 10hz

    wrench = Wrench()
    wrench.force.z = 3.0

    r = rospy.Rate(1.0/3.0)
    
    t0 = rospy.Time.now().to_sec()
    state = 1
    cout = 0
    t_mod_prev = 0.0

    T = 2.5
    t1 = rospy.Time.now().to_sec()

    f_A = 0.0
    f_B = -10.0

    mode = "spline5"
    while not rospy.is_shutdown():
 
        t_mod =(rospy.Time.now().to_sec() - t0) % 3.0

        if t_mod_prev > t_mod:
            state *= -1
            t1 = rospy.Time.now().to_sec()
        if state == 1:
            f_t = f_B
            f_0 = f_A
        else:
            f_t = f_A
            f_0 = f_B
        
        t = (rospy.Time.now().to_sec() - t1)
        if t > T:
            t = T

        s = t/T
        s2 = s*s
        s3 = s2*s
        s4 = s3*s
        s5 = s4*s

        if mode == "step":
            wrench.force.z = f_t
        elif mode == "linear":
            wrench.force.z = (f_t-f_0)*t/T+f_0
        elif mode == "spline3":
            a0 = f_0
            a1 = 0.0
            a2 = 3.0
            a3 = -2.0
            wrench.force.z = a0 + (f_t-f_0) *(a1*s +  a2 * s2 + a3 * s3)
        elif mode == "spline5":
            a0 = f_0
            a1 = 0.0
            a2 = 0.0
            a3 = 10.0
            a4 = -15.0
            a5 = 6.0
            wrench.force.z = a0 + (f_t-f_0) *(a1*s + a2 * s2 + a3 * s3 + a4 * s4 + a5 * s5)

        
        t_mod_prev = t_mod
        pub.publish(wrench)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass