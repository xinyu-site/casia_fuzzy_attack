__author__ = 'duguguang'

from nokov.nokovsdk import *
import numpy as np
import torch
import rospy
from geometry_msgs.msg import Twist
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
real_corners = [(-987, -983), (1019, -983), (1019, 1021), (-987, 1021)]
robot_list = ["9070", "8325"]#, "9185"]
move_target_list = ["9185"] # ["0000"]
global agent_output_action
global mover_output_action
global coord_temp


robotName_list = []
moveName_list = []
FingertripVelocityArray_list = []
for robot_id in robot_list:
    temp_name = "cmd_vel" + robot_id
    robotName_list.append(temp_name)
    FingertripVelocityArray = SlideFrameArray()
    FingertripVelocityArray_list.append(FingertripVelocityArray)
    
for robot_id in move_target_list:
    temp_name = "cmd_vel" + robot_id
    moveName_list.append(temp_name)
    FingertripVelocityArray = SlideFrameArray()
    FingertripVelocityArray_list.append(FingertripVelocityArray)
    
info_list = []
args, algo_args, env_args = init_args()
move_num = len(move_target_list)
world_size = env_args["world_size"]
num_agents = env_args["nr_agents"]
scenario = env_args["scenario"]
int_points_num = env_args["int_points_num"]
if scenario == "pursuit":
    nr_evaders = env_args["nr_evaders"]
last_location = np.zeros((num_agents, 2))
agent_output_action = np.zeros((1, num_agents, 2))
mover_output_action = np.zeros((move_num, 2))
agent_max_lin_velocity = 0.1
mover_max_speed = 0.1
    
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
        self.actor.append(agent)
        policy_actor_state_dict = torch.load(
                str(algo_args["train"]["model_dir"])
                + "/actor_agent"
                + str(0)
                + ".pt"
            )
        self.actor[0].actor = policy_actor_state_dict
        self.actor[0].actor.n_threads = 1
        if "Eg" in self.actor[0].actor.__class__.__name__ or "G" in self.actor[0].actor.__class__.__name__:
            self.actor[0].actor.local_tool.update_edges()
            
        self.agent_num = num_agents
        self.robot_list = robotName_list
        self.move_list = moveName_list
        
        self.publisher_list = []
        self.move_publisher_list = []
        for temp_name in self.robot_list:
            temp_publisher = rospy.Publisher(temp_name,Twist,queue_size=10)
            self.publisher_list.append(temp_publisher)
        
        for temp_name in self.move_list:
            temp_publisher = rospy.Publisher(temp_name,Twist,queue_size=10)
            self.move_publisher_list.append(temp_publisher)

        self.cnt = 0  # 控制step數量
        self.stop = False  # 用于正常退出程序
    
    def timer_callback(self, actions, prey_action):
        time_length = 5000
        self.cnt += 1
        if scenario == "pursuit":
            assert len(prey_action) == move_num
            m_msgs = []
            for i in range(move_num):
                msg = Twist()
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                msg.angular.z = 0.0
                msg.linear.x = prey_action[i, 0]
                msg.linear.y = prey_action[i, 1]
                msg.linear.z = 0
                m_msgs.append(msg)
            if self.cnt > time_length:
                for i in range(move_num):
                    m_msgs[i].angular.z = 0.0
                    m_msgs[i].linear.x = 0.0
                    m_msgs[i].linear.y = 0.0
            if not self.stop:
                for i, robot in enumerate(self.move_publisher_list):
                    robot.publish(m_msgs[i])

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

        if self.cnt > time_length:
            for i in range(num_agents):
                msgs[i].angular.z = 0.0
                msgs[i].linear.x = 0.0
                msgs[i].linear.y = 0.0

        if not self.stop:
            for i, robot in enumerate(self.publisher_list):
                robot.publish(msgs[i])

def get_current_raw(coordinates):
    if scenario == "rendezvous":
        states_dim = num_agents
    elif scenario == "pursuit":
        states_dim = num_agents + nr_evaders
    elif scenario == "navigation":
        states_dim = num_agents + int_points_num
    assert len(coordinates) == states_dim
    back_raw = np.zeros((states_dim, 1))
    
    for index, id in enumerate(robot_list):
        info = coordinates['limo' + id]
        back_raw[index, 0] = np.where(info[5] < 0, info[5] + 2 * np.pi, info[5])
    if scenario == "navigation" or scenario == "pursuit":
        for index, id in enumerate(move_target_list):
                i = index + num_agents
                info = coordinates['limo' + id]
                back_raw[i, 0] = np.where(info[5] < 0, info[5] + 2 * np.pi, info[5])
    return back_raw
        
def get_current_observation(coordinates, frame_cnt):
    if scenario == "rendezvous":
        states_dim = num_agents
    elif scenario == "pursuit":
        states_dim = num_agents + nr_evaders
    elif scenario == "navigation":
        states_dim = num_agents + int_points_num
    assert len(coordinates) == states_dim
    states = np.zeros((1, states_dim, 5))
    back_raw = np.zeros((states_dim, 1))
    for index, id in enumerate(robot_list):
        info = coordinates['limo' + id]
        ue_point = map_to_virtual(info[0:2], real_corners, world_size)
        states[0, index, 0:2] = ue_point
        states[0, index, 2] = info[6]
        states[0, index, 3:] = np.array(info[7:]) / agent_max_lin_velocity * ue_envs.envs[0].env.world.policy_agents[0].max_lin_velocity
        back_raw[index, 0] = np.where(info[5] < 0, info[5] + 2 * np.pi, info[5])

    if scenario == "navigation" or scenario == "pursuit":
        for index, id in enumerate(move_target_list):
            i = index + num_agents
            info = coordinates['limo' + id]
            ue_point = map_to_virtual(info[0:2], real_corners, world_size)
            states[0, i, 0:2] = ue_point
            states[0, i, 2] = info[6]
            states[0, i, 3:] = np.array(info[7:]) / mover_max_speed * 8 # 8为逃跑智能体的最大速度
            back_raw[i, 0] = np.where(info[5] < 0, info[5] + 2 * np.pi, info[5])
    
    obs, s_obs, available_actions = ue_envs.ue_set(states)

    for index, id in enumerate(robot_list):
        info = coordinates['limo' + id]
        last_location[index] = info[0:2]
    return obs, back_raw


def move_action_adjust(actions, back_raw):
    # 调整被追逐者的action
    dt = 0.1
    action_repeat = 10
    real_actions = np.zeros_like(actions)
    scaled_actions = np.zeros_like(actions)
    scaled_actions[:, 0] = actions[:, 0] * mover_max_speed
    scaled_actions[:, 1] = actions[:, 1] * mover_max_speed            
    scaled_actions = scaled_actions * dt * action_repeat
    # scaled_actions[:, 0] = 0.1
    # scaled_actions[:, 1] = 0.1
    real_actions[:, 0] = np.multiply(scaled_actions[:, 0], np.cos(back_raw).T) + np.multiply(scaled_actions[:, 1], np.sin(back_raw).T)
    real_actions[:, 1] = -np.multiply(scaled_actions[:, 0], np.sin(back_raw).T) + np.multiply(scaled_actions[:, 1], np.cos(back_raw).T)
    return real_actions
 
def action_adjust(actions, back_raw):
    # 调整智能体的action
    dt = 0.1
    action_repeat = 10
    scaled_actions = np.zeros_like(actions)
    real_actions = np.zeros_like(actions)
    scaled_actions[:, :, 0] = actions[:, :, 0] * agent_max_lin_velocity
    scaled_actions[:, :, 1] = actions[:, :, 1] * agent_max_lin_velocity
    
    current_norm = np.linalg.norm(scaled_actions, axis=2)
    # 计算缩放因子
    scale_factor = agent_max_lin_velocity / (current_norm + 1e-8)
    # 如果范数超过最大值，则对速度进行缩放
    scaled_actions[:, :, 0] = np.where(current_norm <= agent_max_lin_velocity, scaled_actions[:, :, 0],
                                scaled_actions[:, :, 0] * scale_factor)
    scaled_actions[:, :, 1] = np.where(current_norm <= agent_max_lin_velocity, scaled_actions[:, :, 1],
                                scaled_actions[:, :, 1] * scale_factor)                
    scaled_actions = scaled_actions * dt * action_repeat
    
    # scaled_actions[:, :, 0] = 0
    # scaled_actions[:, :, 1] = 0
    # scaled_actions = np.where(np.abs(scaled_actions) < 1e-3, scaled_actions * 20, scaled_actions)
    # scaled_actions = np.where(np.abs(scaled_actions) < 1e-2, scaled_actions * 10, scaled_actions)
    
    safe_dis = 0#15
    move_dis = 0.001
    # 计算更新后的位置
    now_location = last_location + np.reshape(scaled_actions,(scaled_actions.shape[1], scaled_actions.shape[2]))
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
        global preFrmNo, curFrmNo,frame_cnt, coord_temp
        global agent_output_action
        global mover_output_action
        curFrmNo = frameData.iFrame
        if curFrmNo == preFrmNo:
            return
        preFrmNo = curFrmNo
        '''
        获取相应小车的xyz坐标和欧拉角信息,存储在字典中
        '''
        coordinates = {}
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
            
            # 过滤无关信息
            obj_type = 0 # 0表示无关信息，1表示导航点或被追逐点，2表示智能体
            vel_name = str(markerset.szName)[2:-1]
            if vel_name[4:] in robot_list:
                obj_type = 2
            elif scenario == "navigation" or scenario == "pursuit":
                if vel_name[4:] in move_target_list:
                    obj_type = 1
                    
            if obj_type == 1 or obj_type == 2:   
                x, y, z = body.x, body.y, body.z
                roll, pitch, yaw = quaternion_to_euler(body.qx,body.qy,body.qz,body.qw)
                coordinates[vel_name] = [x, y, z, roll, pitch, yaw, yaw, 0, 0]
        if frame_cnt == 0:
            coord_temp = coordinates.copy()
        # print(coordinates)

        if (frame_cnt % 100) == 0:
            for vel_name in coordinates:
                vx = (coordinates[vel_name][0] - coord_temp[vel_name][0]) / 10. * 9. / 1000
                vy = (coordinates[vel_name][1] - coord_temp[vel_name][1]) / 10. * 9. / 1000
                coordinates[vel_name][7] = vx
                coordinates[vel_name][8] = vy
                coordinates[vel_name][6] = np.arctan2(vy, vx)
            coord_temp = coordinates.copy()
            obs, back_raw = get_current_observation(coordinates, frame_cnt)
            rnn_states = np.zeros((1, num_agents, algo_args["model"]["recurrent_n"], algo_args["model"]["hidden_sizes"][0], ), dtype=np.float32,)
            masks = np.ones((1, num_agents, 1), dtype=np.float32, )
            obs_list = []
            rnn_states_list = []
            masks_list = []
            for agent_id in range(num_agents):
                obs_list.append(obs[:, agent_id])
                rnn_states_list.append(rnn_states[:, agent_id])
                masks_list.append(masks[:, agent_id])
            actions, temp_rnn_state = minimal_publisher.actor[0].act(
                np.stack(obs_list, axis=0).transpose(1, 0, 2),
                np.stack(rnn_states_list, axis=0),
                np.stack(masks_list, axis=0),
                None,
                deterministic=True,
            )
            rnn_states = _t2n(temp_rnn_state).transpose(1, 0, 2, 3)
            actions = _t2n(actions)
            clipped_actions = np.clip(actions, ue_envs.envs[0].env.agents[0].action_space.low, ue_envs.envs[0].env.agents[0].action_space.high)
            actions = action_adjust(clipped_actions, back_raw[:num_agents])
            agent_output_action = actions
            
            if scenario == "pursuit":
                prey_action = []
                for i, agent in enumerate(ue_envs.envs[0].env.world.scripted_agents):
                    t_prey_action = agent.action_callback(agent, ue_envs.envs[0].env.world)
                    prey_action.append(t_prey_action)
                prey_actions = np.stack(prey_action)
                prey_actions = move_action_adjust(prey_actions, back_raw[num_agents:])
                mover_output_action = prey_actions
            else:
                mover_output_action = None

            minimal_publisher.timer_callback(agent_output_action, mover_output_action)
        else:
            back_raw = get_current_raw(coordinates)
            minimal_publisher.timer_callback(agent_output_action, mover_output_action)
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
    

    

