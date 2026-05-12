__author__ = 'duguguang'

from nokov.nokovsdk import *
import numpy as np
import torch
import rospy
from geometry_msgs.msg import Twist,Vector3
import sys, getopt
from math import *

preFrmNo = 0
curFrmNo = 0
frame_cnt = 0

class MinimalPublisher():
    def __init__(self):
        self.targetx = 0.85
        self.targety = -0.85
        self.observation_space = 3

        self.publisher102 = rospy.Publisher('cmd_vel102',Twist,queue_size=10)
        self.publisher100 = rospy.Publisher('cmd_vel100',Twist,queue_size=10)
        self.publisher83 = rospy.Publisher('cmd_vel83',Twist,  queue_size=10)
        self.publisher101 = rospy.Publisher('cmd_vel101',Twist,queue_size=10)
        self.publisher120 = rospy.Publisher('cmd_vel',Twist,   queue_size=10)   #小车名字为com_vel
        self.publisher103 = rospy.Publisher('cmd_vel103',Twist,queue_size=10)
        # self.publisher51 = self.create_publisher(Twist, 'cmd_vel51', 10)
        # self.publisher94 = self.create_publisher(Twist, 'cmd_vel94', 10)
        # self.publisher_led1 = self.create_publisher(Int32, 'led1', 10)
        # self.publisher_led3 = self.create_publisher(Int32, 'led3', 10)
        # self.publisher_led5 = self.create_publisher(Int32, 'led5', 10)
        # self.publisher_led7 = self.create_publisher(Int32, 'led7', 10)
        # self.robot = [robot.getFromDef("e-puck" + str(i)) for i in range(5)]
        # self.trans = []
        # self.rot = []
        # for i in range(5):
        #     self.trans.append(self.robot[i].getField("translation"))
        #     self.rot.append(self.robot[i].getField("rotation"))
        
        # timer_period = 0.1  # seconds # TODO：不知道干什么的
        # self.timer = self.create_timer(timer_period, self.timer_callback)

        self.cnt = 0  # 控制step數量
        self.stop = False  # 用于正常退出程序
        # coordinates = get_current_coordinates() # TODO：要开
        # pos100 = coordinates['epk100']
        # pos101 = coordinates['epk101']
        # pos102 = coordinates['epk102']
        # pos83 = coordinates['epk83']
        # pos120 = coordinates['epk120']

        # self.trans[0].setSFVec3f([pos100[0] / 1000 - 0.327, pos100[1] / 1000 - 0.292, 0])
        # self.trans[1].setSFVec3f([pos101[0] / 1000 - 0.327, pos101[1] / 1000 - 0.292, 0])
        # self.trans[2].setSFVec3f([pos102[0] / 1000 - 0.327, pos102[1] / 1000 - 0.292, 0])
        # self.trans[3].setSFVec3f([pos83[0] / 1000 - 0.327, pos83[1] / 1000 - 0.292, 0])
        # self.trans[4].setSFVec3f([pos120[0] / 1000 - 0.327, pos120[1] / 1000 - 0.292, 0])
        # self.rot[0].setSFRotation([0, 0, 1, pos100[2]])
        # self.rot[1].setSFRotation([0, 0, 1, pos101[2]])
        # self.rot[2].setSFRotation([0, 0, 1, pos102[2]])
        # self.rot[3].setSFRotation([0, 0, 1, pos83[2]])
        # self.rot[4].setSFRotation([0, 0, 1, pos120[2]])
        # step()


    def get_observations(self,coordinates):
        # positions_x = np.array([normalizeToRange(self.robot[i].getPosition()[0], -0.97, 0.97, -1.0, 1.0)
        #                         for i in range(self.num_robots)])
        #
        # positions_y = np.array([normalizeToRange(self.robot[i].getPosition()[1], -0.97, 0.97, -1.0, 1.0)
        #                         for i in range(self.num_robots)])
        #
        # velocity_x = np.array([normalizeToRange(self.robot[i].getVelocity()[0], -0.15, 0.15, -1.0, 1.0)
        #                        for i in range(self.num_robots)])
        # velocity_y = np.array([normalizeToRange(self.robot[i].getVelocity()[1], -0.15, 0.15, -1.0, 1.0)
        #                        for i in range(self.num_robots)])
        # self.distance = np.empty((self.num_robots, 1), float)
        # for i in range(self.num_robots):
        #     dx = positions_x[i] - self.targetx
        #     dy = positions_y[i] - self.targety
        #     self.distance[i] = np.sqrt(dx * dx + dy * dy)
        #
        #
        # self.observations = np.empty((self.num_robots, self.observation_space), float)
        #
        # for i in range(self.num_robots):
        #     self.observations[i] = np.append(self.targetx - positions_x[i],
        #                                      [self.targety - positions_y[i],
        #                                       self.robot[i].getField("rotation").getSFRotation()[3] *
        #                                       self.robot[i].getField("rotation").getSFRotation()[2],
        #                                       ])
        observations = []
        pos100 = coordinates['epk100']
        pos101 = coordinates['epk101']
        pos102 = coordinates['epk102']
        pos83 = coordinates['epk83']
        pos120 = coordinates['epk120']
        observations = np.array([pos100, pos101, pos102, pos83, pos120])
        observations[:, 0] = observations[:, 0] / 1000 - 0.327 - self.targetx
        observations[:, 1] = observations[:, 1] / 1000 - 0.292 - self.targety
        observations[:, 2] = observations[:, 2] * np.pi / 180

        return observations
    
    def timer_callback(self,coordinates):
        # obs = self.get_observations(coordinates) #TODO:要换算出来observation,之后连模型
        # print(obs)
        act0 = [0.1,0.1] #TODO:将observation输入model中,通过model获取action
        act0 = torch.tanh(torch.tensor(act0)).numpy()
        act1 = [0.1,0.1]
        act1 = torch.tanh(torch.tensor(act1)).numpy()
        act2 = [0.1,0.1]
        act2 = torch.tanh(torch.tensor(act2)).numpy()
        act3 = [0.1,0.1]
        act3 = torch.tanh(torch.tensor(act3)).numpy()
        act4 = [0.1,0.1]
        act4 = torch.tanh(torch.tensor(act4)).numpy()
        act = [act0, act1, act2, act3, act4]

        # self.handle_emitter(act)
        # step()
        print("act:{}".format(act1))

        self.cnt += 1
        msgs = []
        for i in range(5):
            msg = Twist()
            # linear, angular = vel_transformation(act[i])
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = 0.0 
            msg.linear.x = -0.1
            msg.linear.y = 0.1
            msg.linear.z = 0.1
            msgs.append(msg)

        if self.cnt > 500:
            for i in range(5):
                msgs[i].angular.z = 0.0
                msgs[i].linear.x = 0.0

        print("linear:{}, angular:{}".format(msgs[1].linear.x, msgs[1].angular.z))
        if not self.stop:
            # publish命令
            self.publisher100.publish(msgs[0])
            self.publisher101.publish(msgs[1])
            self.publisher102.publish(msgs[2])
            self.publisher83.publish(msgs[3])
            self.publisher120.publish(msgs[4])


# 四元数转欧拉角(x,y,z)
def quaternion_to_euler(qx,qy,qz,qw):
    roll = atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = asin(2 * (qw * qy - qx * qz))
    yaw = atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qz * qz + qy * qy))
    return roll, pitch, yaw


def py_data_func(pFrameOfMocapData, pUserData):
    if pFrameOfMocapData == None:  
        print("Not get the data frame.\n")
    else:
        frameData = pFrameOfMocapData.contents
        global preFrmNo, curFrmNo,frame_cnt
        curFrmNo = frameData.iFrame
        if curFrmNo == preFrmNo:
            return
        preFrmNo = curFrmNo
        print( "FrameNo: %d\tTimeStamp:%Ld" % (frameData.iFrame, frameData.iTimeStamp))					
        print( "nMarkerset = %d" % frameData.nMarkerSets)
        '''
        获取相应小车的xyz坐标和欧拉角信息,存储在字典中
        '''
        coordinates = {}
        for iMarkerSet in range(frameData.nMarkerSets):
            markerset = frameData.MocapData[iMarkerSet]
            print( "Markerset%d: %s [nMarkers Count=%d]\n" % (iMarkerSet+1, markerset.szName, markerset.nMarkers))
            print("{\n")

            for iMarker in range(markerset.nMarkers):
                print("\tMarker%d: %3.2f,%3.2f,%3.2f\n" %(	
				iMarker,
				markerset.Markers[iMarker][0],
				markerset.Markers[iMarker][1],
				markerset.Markers[iMarker][2]))
            print( "}\n")

            # MarkerSet对应的刚体信息
            body = frameData.RigidBodies[iMarkerSet] # body的排列顺序和MarkerSet的顺序是一样的
            x, y, z = body.x,body.y,body.z
            roll, pitch, yaw = quaternion_to_euler(body.qx,body.qy,body.qz,body.qw)
            coordinates[markerset.szName] = (x, y, z, roll, pitch, yaw)
        print(coordinates)
        if (frame_cnt%20) == 0:
            minimal_publisher.timer_callback(frameData)
        frame_cnt +=1

def py_msg_func(iLogLevel, szLogMessage):
    szLevel = "None"
    if iLogLevel == 4:
        szLevel = "Debug"
    elif iLogLevel == 3:
        szLevel = "Info"
    elif iLogLevel == 2:
        szLevel = "Warning"
    elif iLogLevel == 1:
        szLevel = "Error"
  
    print("[%s] %s" % (szLevel, cast(szLogMessage, c_char_p).value))

def py_forcePlate_func(pFocePlates, pUserData):
    if pFocePlates == None:  
        print("Not get the forcePlate frame.\n")
        pass
    else:
        ForcePlatesData = pFocePlates.contents
        print("iFrame:%d" % ForcePlatesData.iFrame)
        for iForcePlate in range(ForcePlatesData.nForcePlates):
            print("Fxyz:[%f,%f,%f] xyz:[%f,%f,%f] MFree:[%f]" % (
                ForcePlatesData.ForcePlates[iForcePlate].Fxyz[0],
                ForcePlatesData.ForcePlates[iForcePlate].Fxyz[1],
                ForcePlatesData.ForcePlates[iForcePlate].Fxyz[2],
                ForcePlatesData.ForcePlates[iForcePlate].xyz[0],
                ForcePlatesData.ForcePlates[iForcePlate].xyz[1],
                ForcePlatesData.ForcePlates[iForcePlate].xyz[2],
                ForcePlatesData.ForcePlates[iForcePlate].Mfree
            ))

def main(argv):
    serverIp = '10.1.1.198'
    try:
        opts, args = getopt.getopt(argv,"hs:",["server="])
    except getopt.GetoptError:
        print('SeekerSDKClient.py -s <serverIp>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('SeekerSDKClient.py -s <serverIp>')
            sys.exit()
        elif opt in ("-s", "--server"):
            serverIp = arg

    print ('serverIp is %s' % serverIp)
    print("Started the Seeker_SDK_Client Demo")
    client = PySDKClient()

    ver = client.PySeekerVersion()
    print('SeekerSDK Sample Client 2.4.0.3142(SeekerSDK ver. %d.%d.%d.%d)' % (ver[0], ver[1], ver[2], ver[3]))

    client.PySetVerbosityLevel(0)
    client.PySetMessageCallback(py_msg_func)
    client.PySetDataCallback(py_data_func, None)

    print("Begin to init the SDK Client")
    ret = client.Initialize(bytes(serverIp, encoding = "utf8"))
    
    if ret == 0:
        print("Connect to the Seeker Succeed")
    else:
        print("Connect Failed: [%d]" % ret)
        exit(0)

    #Give 5 seconds to system to init forceplate device
    ret = client.PyWaitForForcePlateInit(5000)
    if (ret != 0):
        print("Init ForcePlate Failed[%d]" % ret)
        exit(0)

    client.PySetForcePlateCallback(py_forcePlate_func, None)

    while(input("Press q to quit\n") != "q"):
        pass
 
if __name__ == "__main__":
    # init ros Node
    global minimal_publisher
    frame_cnt = 0
    rospy.init_node('minimal_publisher', anonymous=True)
    minimal_publisher = MinimalPublisher()
    rate = rospy.Rate(10)  # 10 Hz, adjust as needed
    # main
    main(sys.argv[1:])
    # stop ros
    rospy.signal_shutdown('User interrupted')
    print("Exit now")
    sys.exit(0)

