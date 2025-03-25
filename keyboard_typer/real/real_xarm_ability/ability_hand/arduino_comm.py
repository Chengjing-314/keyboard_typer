import numpy as np
import serial
import time


class ArduinoMotorController():
    def __init__(self, serial_port='/dev/ttyACM0') -> None:
        self.serial_port = serial_port
        self.ser = serial.Serial(self.serial_port, 9600, timeout=2)
        self.adaptive_thresholds = [0, 0, 0, 0, 0, 0]
        # print(self.ser)
        # time.sleep(0.1)

    def control_motors_with_pwm(self, signal_values, thresholds, signal_maxes):
        pwm_values = []
        # self.adaptive_thresholds = self.adaptive_thresholds * 0.9 + signal_values * 0.1
        for value, threshold, s_max in zip(signal_values, thresholds, signal_maxes):
            pwm_value = max(0, min(255, int((value - threshold) * 255 / (s_max - threshold))))
            pwm_values.append(pwm_value)
            self.ser.write(f"{pwm_value} ".encode())
        print("pwm values: ", pwm_values)

    def stop_motors(self):
        for _ in range(20):
            self.ser.write("0 ".encode())
        print('motors stopped')


if __name__ == '__main__':
    # controller = ArduinoMotorController('/dev/serial/by-id/usb-Prolific_Technology_Inc._USB-Serial_Controller_BUBZb11A922-if00-port0')
    controller = ArduinoMotorController()
    thresholds = [0, 0, 0, 0, 0, 0]
    signal_maxes = [100, 100, 100, 100, 100, 100]
    try:
        for i in range(10000):
            controller.control_motors_with_pwm([100] * 5, thresholds, signal_maxes)
            time.sleep(0.005)
    finally:
        controller.stop_motors()
        controller.ser.close()
