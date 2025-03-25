import serial
from serial.tools import list_ports
import time
import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading

from .arduino_comm import ArduinoMotorController
from .python.PPP_stuffing import *
from .python.abh_api_core import *


#
#   Ascii Art Depction of the 4bar linkage mechanism:
#
#                              L3
#                              |
#                       OOOOOOOO
#                       \    q2/
#                        \    /
#                         \  /
#                          \/
#                          /\
#                   L1<---/  \
#                        /    \
#                       /      \--->L2
#                      /q1      \
#                     ############
#                            |
#                           L0
"""
  Main function used to determine the relationship of the Ability Hand 4bar linkage driven angle q2.
  This is an exact calculation of the mechanism. The output q2 angle is a function of the q1 angle
  in the URDF model of the Ability Hand, and is valid for the geometry defined in those files.
  
  INPUT: q1 (URDF q1 of any of the finger digits)
  OUTPUT: q2 (URDF q2 for that finger digit)
"""


def get_abh_4bar_driven_angle(q1):
    q1 = (
        q1 + 0.084474
    )  # factor in offset imposed by our choice of link frame attachments

    # L0 = 9.5
    L1 = 38.6104
    L2 = 36.875
    L3 = 9.1241
    # if X of the base frame was coincident with L3, p3 = [9.5, 0 0]. However, our frame choices are different to make the 0 references for the fingers nice, so this location is a little less convenient.
    p3 = np.array([9.47966, -0.62133, 0])

    cq1 = np.cos(q1)
    sq1 = np.sin(q1)
    p1 = np.array([L1 * cq1, L1 * sq1, 0])

    sol0, sol1 = get_intersection_circles(p3, L2, p1, L3)

    # copy_vect3(&p2, &sols[1])
    p2 = sol1

    # calculate the linkage intermediate angle!
    q2pq1 = np.arctan2(p2[1] - L1 * sq1, p2[0] - L1 * cq1)
    q2 = q2pq1 - q1
    q2 = np.mod(q2 + np.pi, 2 * np.pi) - np.pi
    return q2


"""
  Helper function for the above function. 
  Solves for the intersection of two circles.
  
  Behavior of this function for circles that intersect at only one point, or 
  circles that do not intersect, is not defined.
  
  INPUTS: 
    o0: origin of circle 0
    r0: radius of circle 0
    o1: origin of circle 1
    r1: origin of circle 1
  OUTPUS:
    sol0: 2d position of the first intersection point
    sol1: 2d position of the second intersection point
"""


def get_intersection_circles(o0, r0, o1, r1):
    d = np.sqrt(np.sum((o0 - o1) ** 2))

    sol0 = np.zeros(2)
    sol1 = np.zeros(2)

    r0_sq = r0 * r0
    r1_sq = r1 * r1
    d_sq = d * d

    # solve for a
    a = (r0_sq - r1_sq + d_sq) / (2 * d)

    # solve for h
    h_sq = r0_sq - a * a
    h = np.sqrt(h_sq)

    # find p2
    p2 = o0 + a * (o1 - o0) / d

    t1 = h * (o1[1] - o0[1]) / d
    t2 = h * (o1[0] - o0[0]) / d

    sol0[0] = p2[0] + t1
    sol0[1] = p2[1] - t2

    sol1[0] = p2[0] - t1
    sol1[1] = p2[1] + t2

    return sol0, sol1


class DummyAbilityHand:
    def __init__(
        self, baud=460800, usb_port="/dev/ttyUSB0", reply_mode=0x10, hand_address=0x50
    ) -> None:
        self.usb_port = usb_port

        self.current_joint_pos_read = [-1] * 6
        self.current_joint_vel_read = [-1] * 6
        # Reused Arrays for touch data
        # self.current_touchRead = [-1] * 30
        self.current_touchRead = np.array([0] * 30)

    # Search for Serial Port to use
    # Communicate with the hand
    def set_joint_angle(self, joint_angle):
        if isinstance(joint_angle, list):
            joint_angle = np.array(joint_angle)
        if len(joint_angle) == 10:
            joint_angle = joint_reindex(joint_angle)
        # self.current_joint_pos_read = list(joint_angle.copy())

    def get_hand_state(
        self,
    ):
        augmented_joint_pos_read = []
        temp_current_joint_read = np.array(self.current_joint_pos_read.copy())
        temp_current_joint_read = temp_current_joint_read / 360 * (2 * np.pi)
        # print(self.current_joint_pos_read)
        for i in range(0, 4):
            # print(
            #   self.current_joint_pos_read[i],
            #   get_abh_4bar_driven_angle(self.current_joint_pos_read[i])
            # )
            # print(type(augmented_joint_pos_read), type(self.current_joint_pos_read[i]))
            current_temp_read = temp_current_joint_read[i]
            augmented_joint_pos_read += [
                current_temp_read,
                # temp_current_joint_read[i],
                get_abh_4bar_driven_angle(current_temp_read),
            ]
        # Swap Order
        augmented_joint_pos_read += temp_current_joint_read[-2:]
        # augmented_joint_pos_read = joint_reindex(augmented_joint_pos_read)
        # print(augmented_joint_pos_read)
        augmented_joint_pos_read = np.array(augmented_joint_pos_read)
        augmented_joint_pos_read = joint_reindex(augmented_joint_pos_read)
        # print(augmented_joint_pos_read.shape)
        # for i in range(4):
        return {
            "pos": augmented_joint_pos_read,
            "vel": self.current_joint_vel_read,
            "touch": self.current_touchRead,
        }


# 0x10 1. Finger position, current, touch sensors
# 0x11 2. Finger position, rotor velocity, touch sensor
# 0x12 3. Finger position, current, rotor velocity


# linear function
LINEAR_M = 0.966
LINEAR_B = 0.068
LINEAR_B_ARRAY = np.array([LINEAR_B] * 6)
LINEAR_B_ARRAY[-1] = -LINEAR_B


class RealAbilityHand:
    def __init__(
        self,
        baud=460800,
        usb_port="/dev/ttyUSB0",
        reply_mode=0x10,
        hand_address=0x50,
        plot_touch=False,
        enable_arduino_comm=False,
        verbose=False,
    ) -> None:
        self.usb_port = usb_port
        self.reply_mode = reply_mode
        self.hand_address = hand_address
        self.baud = baud
        self.setup_serial(self.baud)

        self.reset_count = 0
        self.stuff_data = False
        self.isRS485 = False

        self.reset_count = 0
        self.stuff_data = False
        self.isRS485 = False

        self.num_writes = 0
        self.num_reads = 0

        self.current_joint_pos_read = [-1] * 6
        self.current_joint_vel_read = [-1] * 6
        # Reused Arrays for touch data
        self.current_touchRead = [-1] * 30
        self.current_target_joint_pos = None

        self.control_frequency = 30

        self.bytebuffer = bytes([])
        self.stuff_buffer = np.array([])

        self.setup_reply_mode(reply_mode)

        self.plot_touch = plot_touch

        self.prev_error = 0
        self.prev_integral = 0
        self.idx = 1
        
        self.enable_arduino_comm = enable_arduino_comm
        self.arduino_motor_controller = None
        if self.enable_arduino_comm:
            self.arduino_motor_controller = ArduinoMotorController()

        self.verbose = verbose

    def setup_reply_mode(self, reply_mode):
        self.reply_mode = reply_mode
        if reply_mode & 0x10 == 0x10:
            self.farr_to_barr = farr_to_dposition
        elif reply_mode & 0x20 == 0x20:
            self.farr_to_barr = farr_to_dvelocity
        elif reply_mode & 0x30 == 0x30:
            self.farr_to_barr = farr_to_dcurrent
        else:
            raise NotImplementedError

    def __del__(self):
        self.stop_thread()

    def start_thread(
        self,
    ):
        self.running = True
        if self.current_target_joint_pos is None:
            self.current_target_joint_pos = self.get_hand_state()["raw_pos"]
        self.control_thread = threading.Thread(target=self.control_thread_function)
        self.control_thread.start()
        self.tstart = time.time()
        if self.plot_touch:
            self.plot_thread = threading.Thread(target=self.live_plot)
            self.plot_thread.start()

        if self.verbose:
            print(f"[AbilityHand {self.usb_port}] start thread.")

    def stop_thread(
        self,
    ):
        self.running = False
        if hasattr(self, "control_thread"):
            self.control_thread.join()
            print("control_thread joined")
            if self.plot_touch:
                self.plot_thread.join()
                print("plot_thread joined")
        if self.enable_arduino_comm:
            self.arduino_motor_controller.stop_motors()
        if self.verbose:
            print(f"[AbilityHand {self.usb_port}] stop thread.")

    # Search for Serial Port to use
    def setup_serial(self, baud):
        # print(self.baud)
        self.ser = serial.Serial(self.usb_port, baud, timeout=0, write_timeout=0)
        assert self.ser is not None
        print(f"[AbilityHand {self.usb_port}] connected!")
        self.ser.reset_input_buffer()
        # print("set up

    def create_misc_msg(self, cmd):
        barr = []
        barr.append((struct.pack("<B", 0x50))[0])  # device ID
        barr.append((struct.pack("<B", cmd))[0])  # command!
        sum = 0
        for b in barr:
            sum = sum + b
        chksum = (-sum) & 0xFF
        barr.append(chksum)
        return barr

    def pd_vel_control(self, current_pos, dest_pos, prev_error, prev_integral, dt=0.01):
        Kp = np.array([1, 1, 1, 1, 0.3, 0.3]) * 1.0  # Proportional gain
        Kd = Kp / 50
        # Kd = np.array([0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002])  # Derivative gain
        Ki = np.array([0.003, 0.003, 0.003, 0.003, 0.001, 0.001]) * 0.0  # Integral gain

        # import ipdb
        # ipdb.set_trace(context=10)
        error = dest_pos - current_pos
        derivative = (error - prev_error) / dt
        integral = np.clip(prev_integral + error * dt, -100, 100)
        # print(error.shape, derivative.shape, integral.shape)
        output = Kp * error + Kd * derivative + Ki * integral

        output = np.clip(output, -0.2, 0.2)

        # print(f"Thumb error: {np.mean(np.abs(output[4:6])):.4f}, other finger error: {np.mean(np.abs(output[:4])):.4f}",
        #       flush=False)

        return output, error, integral

    # Generate Message to send to hand from array of farr (floating point)
    def joint_to_cmd_msg(self, farr):
        # global reply_mode, hand_address
        # farr could be: position / velocity / current
        msg = self.farr_to_barr(self.hand_address, farr, self.reply_mode - 0x10)
        return msg

    @staticmethod
    def joint_remap_10_qpos_to_6(joint):
        # Ability Hand Real Order:
        # index, middle, ring, pinky, thumb l2, thumb l1

        # Sapien Order:
        # thumb l1, thumb l2, index * 2, middle * 2, ring * 2, pinky * 2

        return joint[[2, 4, 6, 8, 1, 0]]
        # return joint[[0, 2, 4, 6, 8, 9]]

    @staticmethod
    def joint_remap_6_qpos_to_10(joint):

        augmented_joint_pos_read = []
        augmented_joint_pos_read += list(joint[-2:][::-1])
        for i in range(0, 4):
            current_temp_read = joint[i]
            augmented_joint_pos_read += [
                current_temp_read,
                # temp_current_joint_read[i],
                get_abh_4bar_driven_angle(current_temp_read),
            ]
        return np.array(augmented_joint_pos_read)

    def set_joint_angle(self, joint_angle, reply_mode=None, dt=0.01):
        if len(joint_angle) == 10:
            joint_angle = self.joint_remap_10_qpos_to_6(joint_angle)
        if reply_mode is not None:
            self.setup_reply_mode(reply_mode)
        if self.reply_mode & 0x10 == 0x10:
            self.current_target_joint_pos = joint_angle.copy()
        elif self.reply_mode & 0x20 == 0x20:
            self.target = joint_angle.copy()
            # print(self.current_joint_pos_read)
            vel, self.prev_error, self.prev_integral = self.pd_vel_control(
                np.array(self.current_joint_pos_read) / 180 * np.pi,
                self.target,
                self.prev_error,
                self.prev_integral,
                dt=dt
            )
            # print(vel, dt)
            self.current_target_joint_pos = vel.copy()
        # if self.verbose:
        #     print(f"[AbilityHand {self.usb_port}] Target joint angle: {joint_angle}.")
        #     print(f"[AbilityHand {self.usb_port}] Current joint angle: {self.get_hand_state()['raw_pos']}.")

    # Communicate with the hand
    def _inner_set_joint_angle(self, joint_angle):
        # Reindex and Convert Unit
        if isinstance(joint_angle, list):
            joint_angle = np.array(joint_angle)
        if len(joint_angle) == 10:
            joint_angle = self.joint_remap_10_qpos_to_6(joint_angle)
        joint_angle = joint_angle / (2 * np.pi) * 360

        # joint_angle = (joint_angle - LINEAR_B_ARRAY) / LINEAR_M
        # print(joint_angle)

        # self.current_joint_pos_read = list(joint_angle.copy())
        joint_angle = list(joint_angle)

        msg = self.joint_to_cmd_msg(joint_angle)
        # We use Stuff Data = True, isRS485 = False for now, for other variant, check ability API
        # print("before write")
        self.ser.write(PPP_stuff(bytearray(msg)))
        # print("after write")

        # Read first response byte - format header
        # data = self.ser.read(1)
        nb = bytes([])
        # while (len(nb) == 0):
        #     nb = self.ser.read(512)  # gigantic read size with nonblocking

        # Jiyue: read only once
        nb = self.ser.read(512)  # gigantic read size with nonblocking
        # print("length nb: ", len(nb))

        self.bytebuffer = self.bytebuffer + nb
        # print("buffer len", len(self.bytebuffer))
        if len(self.bytebuffer) != 0:  # redundant, but fine to keep
            npbytes = np.frombuffer(self.bytebuffer, np.uint8)
            for b in npbytes:
                payload, self.stuff_buffer = unstuff_PPP_stream(b, self.stuff_buffer)
                if len(payload) != 0:
                    rPos, rI, rV, rFSR = parse_hand_data(payload)
                    if (rPos.size + rI.size + rV.size + rFSR.size) != 0:
                        """If the parser got something, print it out. This is blocking, time consuming, and execution time is not guaranteed, but it is guaranteed to reduce average bandwidth"""
                        self.current_joint_pos_read = rPos.copy()
                        self.current_joint_vel_read = rV.copy()
                        self.current_touchRead = rFSR.copy()
                        self.bytebuffer = bytes([])
                        self.stuff_buffer = np.array([])
                        # print(self.current_joint_pos_read)

    def get_hand_state(
        self,
    ):
        # Filling Missing Values // Ability Read Angle -> Convert to Radian
        augmented_joint_pos_read = []
        temp_current_joint_read = np.array(self.current_joint_pos_read.copy())
        temp_current_joint_read = temp_current_joint_read / 180 * np.pi

        augmented_joint_pos_read = self.joint_remap_6_qpos_to_10(
            temp_current_joint_read
        )

        return {
            "pos": augmented_joint_pos_read,
            "raw_pos": temp_current_joint_read,
            "vel": self.current_joint_vel_read,
            "touch": self.current_touchRead.copy(),
        }

    def control_thread_function(
        self,
    ):
        start = time.monotonic()
        while self.running:
            self._inner_set_joint_angle(self.current_target_joint_pos)
            if self.arduino_motor_controller:
                self.arduino_motor_controller.control_motors_with_pwm(
                    self.current_touchRead.copy().reshape(-1, 6).max(axis=1),
                    [500] * 5, [2000] * 5
                )
            if time.monotonic() - start < 1 / self.control_frequency:
                time.sleep(1 / self.control_frequency - (time.monotonic() - start))

    def get_hand_state_wrapper(
        self,
    ):
        while 1:
            # print("get_hand_state_wrapper")
            # print(self.get_hand_state()["touch"])
            # print(type(self.get_hand_state()["touch"]))
            t = time.time() - self.tstart
            touch_list = self.get_hand_state()["touch"]
            if isinstance(touch_list, np.ndarray):
                touch_list = touch_list.tolist()
            touch_list.insert(0, t)
            # print(touch_list)
            yield touch_list

    def live_plot(
        self,
    ):

        global fig, ax, lines, xbuf, ybuf, num_lines, bufwidth, tstart, x_data, y_data, expname

        expname = "default"
        fig, ax = plt.subplots()
        plt.setp(ax, ylim=(0, 4500))  # manually set axis y limits
        plt.setp(ax, xlim=(0, 30))
        plt.title("Touch Sensor Data")
        plt.xlabel("Time(s)")
        plt.ylabel("Raw Touch Data")

        num_lines = 30
        bufwidth = 500

        lines = []
        xbuf = []
        ybuf = []
        x_data = []
        y_data = []
        for i in range(num_lines):
            lines.append(ax.plot([], [])[0])
            xbuf.append([])
            ybuf.append([])
            y_data.append([])
        # initalize all xy buffers to 0
        for i in range(0, num_lines):
            y_data[i].append(0)
            for j in range(0, bufwidth):
                xbuf[i].append(0)
                ybuf[i].append(0)
        x_data.append(0)

        tstart = self.tstart

        print("animation start")

        anim = animation.FuncAnimation(
            fig,
            self.animate,
            init_func=self.init,
            frames=self.get_hand_state_wrapper(),
            interval=0,
            blit=True,
            save_count=50,
        )

        print("animation end")
        plt.show()

    # lines = []
    # xbuf = []
    # ybuf = []
    # tstart = 0
    # num_lines = 0
    # bufwidth = 0
    # idx = 1

    # x_data = []
    # y_data = []
    # expname = None

    # initialization function. needed for the 'blitting' option,
    # which is the lowest latency plotting option

    def init(
        self,
    ):  # required for blitting to give a clean slate.
        global lines

        for line in lines:
            line.set_data([], [])
        return lines

    def animate(self, args):
        # print("--------args--------")
        # print(args)
        # print(len(args))
        global ax, lines, xbuf, ybuf, num_lines, bufwidth, x_data, y_data, expname
        t_interval = 10
        for i in range(0, num_lines):
            del xbuf[i][0]
            del ybuf[i][0]
            xbuf[i].append(args[0])
            # print(f"----{i}----")
            # print(args[0])
        x_data.append(args[0])
        i = 0
        for arg in args:
            if i > 0:
                # print(i-1)
                try:
                    ybuf[i - 1].append(arg)
                    y_data[i - 1].append(arg)
                except:
                    import ipdb

                    ipdb.set_trace(context=10)
                # print(len(ybuf[i-1]))
                # print(arg)
            i = i + 1
        for i, line in enumerate(lines):
            line.set_data(xbuf[i], ybuf[i])

        xmin = min(xbuf[0])
        xmax = max(xbuf[0])

        # print(type(xbuf))
        if time.time() - tstart >= self.idx * t_interval:
            print(len(x_data))
            print(len(y_data[0]))
            # with open(f"./experiment_data/{expname}/time/x_time_{idx}.json", "w") as f:
            # 	json.dump(x_data, f)
            # with open(f"./experiment_data/{expname}/touch/y_touch_{idx}.json", "w") as f:
            # 	json.dump(y_data, f)
            print(len(x_data))
            print(len(y_data[0]))
            x_data = []
            y_data = []
            for i in range(num_lines):
                y_data.append([])
            self.idx += 1

        # print("--------xmax xmin--------")
        # print(len(xbuf[0]))
        # print(xmax)
        # print(xmin)
        # print(len(ybuf[0]))
        # print("--------xbuf--------")
        # print(len(xbuf[0]))
        # print("--------ybuf--------")
        # print(len(ybuf[0]))
        # print(ybuf[0])
        plt.setp(ax, xlim=(xmin, xmax))
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=False)
        # print("--------end--------")
        # plt.savefig("img.png")
        # time.sleep(10)
        return lines

    def get_data(self):
        data = self.get_hand_state()
        data = np.concatenate([data["raw_pos"], data["vel"], data["touch"]])
        return data

    def save_data(self, path, time):
        data = self.get_hand_state()
        data = np.concatenate([data["raw_pos"], data["vel"], data["touch"]])
        path.mkdir(parents=True, exist_ok=True)
        path = path / f"{time}.npy"
        np.save(path, data)
