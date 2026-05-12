__author__ = 'duguguang'

from nokov.nokovsdk import *
import numpy as np
import time
import torch
import rospy
from geometry_msgs.msg import Twist, Vector3
from harl.algorithms.actors import ALGO_REGISTRY
import sys, getopt
from math import *
from harl.utils.trans_tools import _t2n
from limo_control.sim2real_util import *
from limo_control.Utility import *
from harl.utils.envs_tools import make_eval_env
import copy

preFrmNo = 0
curFrmNo = 0
frame_cnt = 0

# 真实场景四个角的坐标，应该为左下角，右下角，右上角，左上角
real_corners = [(-638, -830), (846, -830), (846, 654), (-638, 654)]
robot_list = ["9070","8325","9185"]
global output_vel
output_vel = np.zeros((3, 2))
nav_target_list = ["target1"]

robotName_list = []
FingertripVelocityArray_list = []
for robot_id in robot_list:
    temp_name = "cmd_vel" + robot_id
    robotName_list.append(temp_name)
    FingertripVelocityArray = SlideFrameArray()
    FingertripVelocityArray_list.append(FingertripVelocityArray)
info_list = []
    
args, algo_args, env_args = init_args()
world_size = env_args["world_size"]
num_agents = env_args["nr_agents"]
scenario = env_args["scenario"]
int_points_num = env_args["int_points_num"]
last_location = np.zeros((num_agents, 2))
nav_target_location = np.zeros((int_points_num, 2))
    
class MinimalPublisher():
    def __init__(self, ue_envs):
        self.device = "cuda"
        self.ue_envs = ue_envs
        self.actor = []
        agent = ALGO_REGISTRY[args["algo"]](
            {**algo_args["model"], **algo_args["algo"], **env_args, **algo_args["train"]},
            self.ue_envs.observation_space[0],
            self.ue_envs.action_space[0],
            device=self.device,
        )
        for agent_id in range(num_agents):
            self.actor.append(agent)
        for agent_id in range(num_agents):
            self.actor[agent_id].restore(algo_args["train"]["model_dir"], agent_id)

            
        self.agent_num = num_agents
        self.robot_list = robotName_list
        self.publisher_list = []
        for temp_name in self.robot_list:
            temp_publisher = rospy.Publisher(temp_name,Twist,queue_size=10)
            self.publisher_list.append(temp_publisher)

        self.cnt = 0  # 控制step數量
        self.stop = False  # 用于正常退出程序
    
    def timer_callback(self, actions, back_raw):
        self.cnt += 1
        msgs = []
        for i in range(num_agents):
            msg = Twist()
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = 0.0
            
            msg.linear.x = actions[0, i, 0]
            msg.linear.y = actions[0, i, 1]
            msg.linear.z = 0
            msgs.append(msg)

        if self.cnt > 500:
            for i in range(num_agents):
                msgs[i].angular.z = 0.0
                msgs[i].linear.x = 0.0
                msgs[i].linear.y = 0.0

        if not self.stop:
            for i, robot in enumerate(self.publisher_list):
                robot.publish(msgs[i])

def get_current_observation(coordinates, frame_cnt):
    assert len(coordinates) == num_agents
    states = np.zeros((1, num_agents, 5))
    for index, id in enumerate(robot_list):
        info = coordinates['limo' + id]
        # print('limo' + str(id) + ":", info)
        ue_point = map_to_virtual(info[0:2], real_corners, world_size)
        states[0, index, 0:2] = ue_point
        states[0, index, 2] = np.where(info[5] < 0, info[5] + 2 * np.pi, info[5])
        states[0, index, 3:] = info[6:]
                
    global output_vel
    # print("output_vel:", output_vel)
    
    status_list = []

    if scenario == "rendezvous":
        status_list.append(states)
    elif scenario == "navigation":
        status_list.append(states)
        status_list.append(nav_target_location)
    elif scenario == "pursuit":
        status_list.append(states)
        status_list.append(nav_target_location)
    else:
        raise Exception("no such environment!")
    # print("nav_target_location", nav_target_location)
    obs, s_obs, available_actions = ue_envs.ue_set(status_list)
    # print("obs", obs)

    for index, id in enumerate(robot_list):
        info = coordinates['limo' + id]
        last_location[index] = info[0:2]
    return obs, states[0, :, 2].copy()
    
def action_adjust(actions, back_raw):
    max_lin_velocity = 0.1
    dt = 0.1
    action_repeat = 10
    scaled_actions = np.zeros_like(actions)
    real_actions = np.zeros_like(actions)
    scaled_actions[:, :, 0] = actions[:, :, 0] * max_lin_velocity
    scaled_actions[:, :, 1] = actions[:, :, 1] * max_lin_velocity
    
    current_norm = np.linalg.norm(scaled_actions, axis=2)
    # 计算缩放因子
    scale_factor = max_lin_velocity / (current_norm + 1e-8)
    # 如果范数超过最大值，则对速度进行缩放
    scaled_actions[:, :, 0] = np.where(current_norm <= max_lin_velocity, scaled_actions[:, :, 0],
                                scaled_actions[:, :, 0] * scale_factor)
    scaled_actions[:, :, 1] = np.where(current_norm <= max_lin_velocity, scaled_actions[:, :, 1],
                                scaled_actions[:, :, 1] * scale_factor)                
    scaled_actions = scaled_actions * dt * action_repeat
    global output_vel
    output_vel = scaled_actions[0]
        
    # scaled_actions[:, :, 0] = 0
    # scaled_actions[:, :, 1] = -0.1
    print("scaled_actions", scaled_actions)
    print("*"*10)
    
    safe_dis = 0#15
    move_dis = 0.001
    # 计算更新后的位置
    now_location = last_location + np.reshape(scaled_actions,(scaled_actions.shape[1], scaled_actions.shape[2]))
    #print(now_location)
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            distance = np.linalg.norm(now_location[i] - now_location[j])
            if distance < safe_dis:
                direction = now_location[i] - now_location[j]
                norm_direction = direction / np.linalg.norm(direction)
                scaled_actions[0, i] = norm_direction * move_dis
                scaled_actions[0, j] = norm_direction * move_dis
    real_actions[:, :, 0] = np.multiply(scaled_actions[:, :, 0], np.cos(back_raw).T) + np.multiply(scaled_actions[:, :, 1], np.sin(back_raw).T)
    real_actions[:, :, 1] = -np.multiply(scaled_actions[:, :, 0], np.sin(back_raw).T) + np.multiply(scaled_actions[:, :, 1], np.cos(back_raw).T)
    info = {'state': ue_envs.envs[0].env.world.agent_states, 'actions': actions,
            'pursuer_states': ue_envs.envs[0].env.world.agent_states,
            'evader_states': ue_envs.envs[0].env.world.landmark_states,
            'velocities': np.vstack([agent.state.p_vel for agent in ue_envs.envs[0].env.agents]),
            'pos': np.vstack([agent.state.p_pos for agent in ue_envs.envs[0].env.agents]),
            'ori': np.vstack([agent.state.p_orientation for agent in ue_envs.envs[0].env.agents])}
    info_list.append(copy.deepcopy(info))
    return real_actions


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
        # print( "FrameNo: %d\tTimeStamp:%Ld" % (frameData.iFrame, frameData.iTimeStamp))					
        # print( "nMarkerset = %d" % frameData.nMarkerSets)
        '''
        获取相应小车的xyz坐标和欧拉角信息,存储在字典中
        '''
        coordinates = {}
        nav_index = 0
        for iMarkerSet in range(frameData.nMarkerSets):
            markerset = frameData.MocapData[iMarkerSet]
            # print( "Markerset%d: %s [nMarkers Count=%d]\n" % (iMarkerSet+1, markerset.szName, markerset.nMarkers))
            # print("{\n")
            # for iMarker in range(markerset.nMarkers):
            #     print("\tMarker%d: %3.2f,%3.2f,%3.2f\n" %(	
			# 	iMarker,
			# 	markerset.Markers[iMarker][0],
			# 	markerset.Markers[iMarker][1],
			# 	markerset.Markers[iMarker][2]))
            # print( "}\n")

            # MarkerSet对应的刚体信息
            body = frameData.RigidBodies[iMarkerSet] # body的排列顺序和MarkerSet的顺序是一样的
            # 提取导航任务的信息
            if scenario == "navigation":
                nav_name = str(markerset.szName)[2:-1]
                if nav_name in nav_target_list:
                    nav_target_location[nav_index, 0] = 0 # body.x
                    nav_target_location[nav_index, 1] = 0 # body.y
                    nav_target_location[nav_index] = map_to_virtual(nav_target_location[nav_index], real_corners, world_size)
                    nav_index += 1
            
            # 过滤无关信息
            vel_name = str(markerset.szName)[2:-1]
            if vel_name[4:] in robot_list:
                vel_index = robot_list.index(vel_name[4:])
                x, y, z = body.x,body.y,body.z
                roll, pitch, yaw = quaternion_to_euler(body.qx,body.qy,body.qz,body.qw)
                Fingertrip = Point(markerset.Markers[iMarkerSet][0], markerset.Markers[iMarkerSet][1], markerset.Markers[iMarkerSet][2], vel_name)
                global FingertripVelocityArray_list
                FingertripVelocityArray_list[vel_index].cache(Fingertrip)
                # method = CalculateVelocity(90, 21) # FPS:60 FrameFactor:3 which means the first 2 frame has None
                method = CalculateVelocityByTwoFrame(90)
                real_vel = FingertripVelocityArray_list[vel_index].try_to_calculate(method)
                if real_vel is None:
                    vx = 0
                    vy = 0
                else:
                    vx = real_vel.Vx/1000
                    vy = real_vel.Vy/1000
                coordinates[vel_name] = (x, y, z, roll, pitch, yaw, vx, vy)
                
                     
        if (frame_cnt % 20) == 0:
            print("coordinates:", coordinates)
            print("-"*10)
            obs, back_raw = get_current_observation(coordinates, frame_cnt)
            actions = []
            for agent_id in range(num_agents):
                actions.append(
                    _t2n(minimal_publisher.actor[agent_id].get_actions(obs[:, agent_id][0], False))
                )
            actions = np.expand_dims(np.array(actions), 0)
            actions = action_adjust(actions, back_raw)
            minimal_publisher.timer_callback(actions, back_raw)
            # frame_cnt = 0   #  这里是不是不应该置为0，这一操作会导致每次frame_cnt % 20都为0，即每一帧都会生成动作并发布。
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
    ue_envs = (make_eval_env(args["env"], algo_args["seed"]["seed"], 1, env_args, ))
    # init ros Node
    frame_cnt = 0
    rospy.init_node('minimal_publisher', anonymous=True)
    minimal_publisher = MinimalPublisher(ue_envs)
    rate = rospy.Rate(10)  # 10 Hz, adjust as needed
    # main
    main(sys.argv[1:])
    # stop ros
    rospy.signal_shutdown('User interrupted')
    print("Exit now")
    ue_envs.envs[0].make_ani((info_list, None, None))
    sys.exit(0)
    

    

