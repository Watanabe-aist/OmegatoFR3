#!/usr/bin/env python
# license removed for brevity
from ftplib import ftpcp
from time import time
import rospy
from geometry_msgs.msg import Pose
import math


def talker():
    pub = rospy.Publisher('/omega_driver/left/cmd_pose', Pose, queue_size=10)
    pub2 = rospy.Publisher('/omega_driver/right/cmd_pose', Pose, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(500) # 10hz

    pose = Pose()

    rospy.sleep(3.0)
    
    t0 = rospy.Time.now().to_sec()
    while not rospy.is_shutdown():
        t = (rospy.Time.now().to_sec() - t0)
        f = 0.1*(1.0+t/10.0)
        # print(t)
        # pose.position.x = -(0.02*math.cos(2.0*math.pi*f*t)-0.02)
        pose.position.x = (0.01*math.cos(2.0*math.pi*f*t)-0.01)
        pub.publish(pose)
        pub2.publish(pose)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass