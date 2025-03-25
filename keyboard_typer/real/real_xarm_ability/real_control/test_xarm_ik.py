import time
import threading

import numpy as np
from xarm import XArmAPI

def main():
    arm_ip = "192.168.1.209"
    arm = XArmAPI(arm_ip, is_radian=True)
    data_buffer = []
    time_buffer = []

    def get_xarm_state():
        for _ in range(5000):
            code, xarm_state = arm.get_joint_states(is_radian=True)
            data_buffer.append(xarm_state)
            time_buffer.append(time.perf_counter())
            time.sleep(0.001)

    thread = threading.Thread(target=get_xarm_state)

    thread.start()
    
    # arm.set_position(473, 0.4, 171.4, 98.6, 16.5, 90, is_radian=False, motion_type=2)
    # arm.set_position(478.2, -65.7, 150.5, 98.4, 27.2, 74.6, is_radian=False, motion_type=2)
    # arm.set_position(502.1, 51.9, 241.8, 109.6, 33.3, 66.3, is_radian=False, motion_type=2)
    arm.set_position(465.8, 52.5, 229.5, 132.4, 36, 90.1, is_radian=False, motion_type=2)
    
    time.sleep(5)
    thread.join()

    # pos = np.array([data[0] for data in data_buffer])

    # plt.plot(time_buffer, pos[:, 3])
    # plt.show()


if __name__ == "__main__":
    main()