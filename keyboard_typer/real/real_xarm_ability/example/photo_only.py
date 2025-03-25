import sys

sys.path.append('/home/jiyue/Documents/chengjing/neural-teleop/real_xarm_ability')

from real_control.xarm7_ability import XArm7Ability
import numpy as np 
from typing import Dict
from fastdev.robo import RobotModel, RobotModelConfig
import pickle




class RealPhotoOnly():
    
    def __init__(self):
        
        self.robot = XArm7Ability(
            use_arm=False,
            use_hand=True,
            arm_ip="192.168.1.242",
            hand_tty_index="usb-Prolific_Technology_Inc._USB-Serial_Controller_BUBZb11A922-if00-port0",
            is_right=True,
            use_servo_control=False,
            enable_finger_tactile=False,
            enable_palm_tactile=False,
            # finger_port='/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_44231313430351F0D1B1-if00',
            finger_port="/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_4423131343035140A011-if00",
            palm_port="/dev/ttyACM1",
        )
        
        config = RobotModelConfig(urdf_or_mjcf_path="/home/jiyue/Documents/chengjing/neural-teleop/assets/robot_description/ability_hand/ability_hand_right_glb.urdf")
        hand_model = RobotModel(config)    
        
        self.joint_limits = hand_model.joint_limits[[2, 3, 4, 5, 1, 0] , :] # index q1, middle q1, ring q1, pinky q1, thumb q2, thumb q1
        
        self.magic_number = 2.0943952
    
    def initialiation(self):
        
        self.robot.reset()
        self.robot.start()
        self.robot.hand.start_thread()
        
    
    
    def set_hand(self, qpos: np.ndarray) -> None:
        self.robot.control_hand_qpos(qpos)

import time


def main():
    photobot = RealPhotoOnly()
    photobot.initialiation()
    mn = photobot.magic_number
    
    """  with open("/home/jiyue/Documents/chengjing/neural-teleop/ability_hand_succ_6d.pkl", 'rb') as f:
        data = pickle.load(f)['full_qpos'] """
    
    """    
    data_keys = list(data.keys())
    
    grasp_idx = 230
    
    for key in data_keys:
        grasp_pose = data[key][grasp_idx, 6:]
        grasp_pose = grasp_pose[[2, 3, 4, 5, 1, 0]]
        photobot.set_hand(grasp_pose)
        
        import IPython; IPython.embed() # noqa """
        
    PRE_DEFINED_GRASP_QPOS = {
    "pre_grasp_big" : np.array([ 0.        ,  0.31144047,  0.43592626,  0.5883939 ,  0.37488815, -0.83802634]), 
    "pre_grasp_medium" : np.array([ 0.47739908,  0.60591507,  0.7078066 ,  0.89841664,  0.        ,-1.0762186 ]),
    "pre_grasp_ok": np.array([ 0.50204027,  0.6876209 ,  0.8075026 ,  0.87726295,  0.        ,-1.1309468 ]),
    "pre_grasp_box": np.array([ 0.4704736 ,  0.6071083 ,  0.68537676,  0.6301631 ,  0.5587983 ,-1.4845282 ]),
    "pre_grasp_articulation_flip": np.array([ 0.4704736 ,  0.6071083 ,  0.68537676,  0.6301631 ,  0.5587983 ,-1.4845282 ]),
    "pre_grasp_mug_flipped": np.array([ 0.        ,  0.12430208,  0.5866085 ,  0.8027905 ,  0.83199966, -0.5606921 ]),
    "pre_grasp_articulation_flip2" : np.array([ 0.43704072,  0.5438471 ,  0.53281456,  0.4988353 ,  1.0463606 , -1.1565962 ]),
    "good_grasp_start": np.array([ 0.6539901 ,  0.7697663 ,  0.8852924 ,  1.0771929 ,  0.96033925, -1.0593338 ]),
    "touch?": np.array([ 0.        ,  0.531208  ,  0.707204  ,  0.90617687,  0.94008327, -0.53727716]),
    "good_articulation_grasp_start": np.array([ 0.6508637 ,  0.63752687,  0.58964396,  0.489967  ,  0.8555166 , -1.3491131 ]),
    "good_articulation_grasp_start2": np.array([ 0.7688316,  0.8394436,  1.0699748,  1.080073 ,  0.837922 , -1.382531 ]),
    "index_open_grasp": np.array([ 0.04776989,  0.44679022,  0.6055593 ,  0.8294693 ,  1.5979425 , -0.18667705]),
    "articulaion_grasp_3": np.array([ 0.28623012,  0.53993404,  0.62230074,  0.80142856,  0.6922632 , -1.5928285 ]),
    "grasping_start_pose": np.array([ 0.49098393,  0.63924515,  0.734021  ,  0.8453297 ,  1.1317776 ,-0.74169457])
}
        
    grasp_pose = PRE_DEFINED_GRASP_QPOS['pre_grasp_big']
    # grasp_pose[0] = 1
    # grasp_pose[1] *= 3
    # grasp_pose[2] *= 1.5
    # grasp_pose[3] *= 1.4
    # grasp_pose[-2] *= 1.8
    # grasp_pose[-1] *= 1.8
    print(grasp_pose)
    photobot.set_hand(grasp_pose)
    import IPython; IPython.embed() # noqa
    


if __name__ == "__main__":
    
    main()