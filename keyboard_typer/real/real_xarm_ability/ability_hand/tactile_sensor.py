import time
import serial
import numpy as np
import threading
import json

class TactileSensor():
    def __init__(self, 
                finger_serial_port='/dev/ttyACM0', 
                palm_serial_port='/dev/ttyACM1') -> None:

        self.finger_serial_port = finger_serial_port
        self.finger_ser = serial.Serial(self.finger_serial_port, 9600, timeout=2)
        self.finger_contact_thresholds = np.zeros(14)
        self.finger_force_thresholds = np.ones(14) * 300
        self.init_finger_sensor()
        self._finger_lock = threading.Lock()
        self._finger_thread = threading.Thread(target=self.read_finger_data)

        self.palm_serial_port = palm_serial_port
        if palm_serial_port:
            self.prev_frame = np.zeros((16, 16))
            self.palm_ser = serial.Serial(self.palm_serial_port, 200000, timeout=2)
            self.palm_threshold = 12
            self.palm_noise_scale = 60
            self.palm_contact_thresholds = np.ones(4) * 150
            self.palm_force_thresholds = np.zeros(4)
            self.init_palm_sensor()
            self._palm_lock = threading.Lock()
            self._palm_thread = threading.Thread(target=self.read_palm_data)
    
    def start(self):
        self._finger_thread.start()
        if self.palm_serial_port:
            self._palm_thread.start()
    
    def stop(self):
        self._finger_thread.join()
        self.finger_ser.close()
        if self.palm_serial_port:
            self._palm_thread.join()
            self.palm_ser.close()
    
    def get_tactile(self):
        
        tactile_data = np.zeros(18) # 14 from finger, 4 from palm
        
        with self._finger_lock:
            tactile_data[:14] = self.finger_tactile

        if self.palm_serial_port:
            with self._palm_lock:
                tactile_data[14:] = self.palm_tactile
        
        return tactile_data
                
    def init_finger_sensor(self):
        data_tac = []
        self.finger_tactile = np.zeros(14)
        num = 0
        start_time = time.time()
        while True:
            if self.finger_ser.in_waiting > 0:
                line = self.finger_ser.readline().decode('utf-8').strip()
                # print(line)
                if '[' not in line or ']' not in line:
                    continue
                if len(line) < 33:
                    continue
                else:
                    line = line[1:-1]
                    # print(line)
                    # print((line.split(',')))
                if '[' in line or ']' in line:
                    continue
                if line.split(',') == '':
                    continue
                # print(line.split(','))
                # print(len(np.array(line.split(',')).astype(np.float32)))
                finger_tactile = np.delete(np.array(line.split(',')).astype(np.float32), [4, 15])
                if len(finger_tactile) != 14:
                    continue
                else:
                    self.finger_tactile = finger_tactile
                
                # print(len(line.split(',')))
                print(self.finger_tactile)
                assert len(self.finger_tactile) == 14
                if self.finger_tactile is not None:
                    data_tac.append(self.finger_tactile)
                    num += 1
                    if num > 30:
                        break
        data_tac = np.array(data_tac)
        # self.finger_contact_thresholds = np.mean(data_tac, axis=0) + np.array([20, 20, 100, 20, 
        #                                                                     20, 20, 50 ,20,
        #                                                                     20, 20, 20, 20,
        #                                                                     100, 20])
        self.finger_contact_thresholds = np.array([200, 200, 950, 300, 200, 950, 200, 200, 200, 200, 200, 200, 900, 200])
        # self.finger_contact_thresholds = np.array([300, 200, 1000, 400, 400, 1000, 300, 600, 500, 200, 500, 200, 950, 500])
        # self.finger_contact_thresholds = np.array([100, 100, 1000, 200, 100, 1000, 100, 100, 100, 100, 100, 100, 950, 100])
        self.finger_force_thresholds = np.min((self.finger_contact_thresholds * 1.5, np.ones(14) * 1000), axis=0)
        print(np.mean(data_tac, axis=0))
        print(self.finger_contact_thresholds)
        print("Finger Tactile Sensor Finish Initialization!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    def read_finger_data(self):
        while True:
            if self.finger_ser.in_waiting > 0:
                line = self.finger_ser.readline().decode('utf-8').strip()
                if len(line) < 33:
                    continue
                else:
                    line = line[1:-1]
                finger_tactile = np.delete(np.array(line.split(',')).astype(np.float32), [4, 15])
                if len(finger_tactile) < 14:
                    continue
                # print(finger_tactile)
                with self._finger_lock:
                    self.finger_tactile[:] = 0
                    self.raw_tactile = finger_tactile
                    self.finger_tactile[finger_tactile < self.finger_contact_thresholds] = 0
                    self.finger_tactile[finger_tactile > self.finger_contact_thresholds] = 1
                    # import pdb; pdb.set_trace()
                    # self.finger_tactile[finger_tactile > self.finger_force_thresholds] = 2
    
    def init_palm_sensor(self):
        self.palm_tactile = np.zeros(14)
        data_tac = []
        self.palm_current_tactile = None
        backup = None
        # self.is_palm_init=False
        self.palm_tactile_norm = np.zeros((16, 16))
        num = 0
        start_time = time.time()
        while True:
            if self.palm_ser.in_waiting > 0:
                try:
                    line = self.palm_ser.readline().decode('utf-8').strip()
                except:
                    line = ""
                if len(line) < 10:
                    if self.palm_current_tactile is not None and len(self.palm_current_tactile) == 16:
                        backup = np.array(self.palm_current_tactile)
                        print("fps",1 / (time.time() - start_time))
                        start_time = time.time()
                        data_tac.append(backup)
                        num += 1
                        if num > 30:
                            break
                    self.palm_current_tactile = []
                    continue
                if self.palm_current_tactile is not None:
                    str_values = line.split()
                    int_values = [int(val) for val in str_values]
                    matrix_row = int_values
                    self.palm_current_tactile.append(matrix_row) 

        data_tac = np.array(data_tac)
        self.palm_tactile_median = np.median(data_tac, axis=0)
        # self.is_palm_init=True
        print("Palm Tactile Sensor Finish Initialization!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    def read_palm_data(self):
        while True:
            if self.palm_ser.in_waiting > 0:
                try:
                    line = self.palm_ser.readline().decode('utf-8').strip()
                    # print("fps",1/(time.time()-t1))
                    # t1 =time.time()
                except:
                    line = ""
                if len(line) < 10:
                    if self.palm_current_tactile is not None and len(self.palm_current_tactile) == 16:
                        backup = np.array(self.palm_current_tactile)
                        # print(backup)
                    self.palm_current_tactile = []
                    if backup is not None:
                        contact_data= backup - self.palm_tactile_median - self.palm_threshold
                        contact_data = np.clip(contact_data, 0, 100)
                        
                        if np.max(contact_data) < self.palm_threshold:
                            self.palm_tactile_norm = contact_data / self.palm_noise_scale
                        else:
                            # contact_data_norm = np.log(contact_data + 1) / np.log(2.0)
                            self.palm_tactile_norm = contact_data / np.max(contact_data)
                        
                        self.palm_tactile_norm = self.temporal_filter(self.palm_tactile_norm, self.prev_frame)
                        self.prev_frame = self.palm_tactile_norm
                        
                        self.palm_tactile_norm = (self.palm_tactile_norm * 255).astype(np.uint8)
                        
                        self.palm_tactile_norm = self.palm_tactile_norm.reshape(2, 8, 2, 8).transpose(0, 2, 1, 3)
                        
                        with self._palm_lock:
                            palm_tactile = self.palm_tactile_norm.max(axis=(2, 3)).reshape(-1)
                            self.palm_tactile[palm_tactile < self.palm_contact_thresholds] = 0
                            self.palm_tactile[palm_tactile > self.palm_contact_thresholds] = 1
                            # self.palm_tactile[palm_tactile > self.palm_force_thresholds] = 2
                        
                    continue

                if self.palm_current_tactile is not None:
                    str_values = line.split()
                    int_values = [int(val) for val in str_values]
                    matrix_row = int_values
                    self.palm_current_tactile.append(matrix_row) 
                        
                    continue
    
    def temporal_filter(self, new_frame, prev_frame, alpha=0.2):
        """
        Apply temporal smoothing filter.
        'alpha' determines the blending factor.
        A higher alpha gives more weight to the current frame, while a lower alpha gives more weight to the previous frame.
        """
        return alpha * new_frame + (1 - alpha) * prev_frame


if __name__ == '__main__':
    tactile_sensor = TactileSensor(palm_serial_port=None)
    tactile_sensor.start()
    while True:
        tactile_data = tactile_sensor.get_tactile()
        print(tactile_data)
        time.sleep(0.005)

    print("reaching end")
    tactile_sensor.stop()
    print("stopped")
        
            
    
    