#!/usr/bin/env python

from time import sleep
import rospy
from geometry_msgs.msg import Twist, Vector3

def talker():
    # ROS节点初始化
    rospy.init_node('talker', anonymous=True)

    # 创建一个Publisher，发布名为cmd_vel的topic，消息类型为geometry_msgs/Twist
    chatter_pub = rospy.Publisher('cmd_vel9185', Twist, queue_size=10)
    
    # 设置循环的频率
    loop_rate = rospy.Rate(10)

    for count in range(10):
        # 设置需要发布的速度大小
        twist = Twist()
        twist.linear.x = -0.2
        twist.linear.y = 0.0
        # twist.angular = Vector3(0, 0, 0)

        # 将设置好的速度发布出去
        chatter_pub.publish(twist)


        # 按照循环频率延时
        loop_rate.sleep()

    # # # 使用rospy.spin()等待回调函数
    # rospy.spinOnce()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
