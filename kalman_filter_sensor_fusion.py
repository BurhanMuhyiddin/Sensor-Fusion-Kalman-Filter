from http.server import HTTPServer, BaseHTTPRequestHandler

import sys
import threading
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# Some global variable initializations

all_acc_ang_data = np.zeros((1,2)) # all accelerometer data
crt_acc_data = np.zeros((1,3)) # current accelerometer data

all_gyr_ang_data = np.zeros((1,2)) # all acceleroemeter data
crt_gyr_data = np.zeros((1,3)) # current accelerometer data

prd_ang_data = np.zeros((1,2)) # predicted roll and pitch angles

g_roll = 0
g_pitch = 0
a_roll = 0
a_pitch = 0

dt = 0
old_seconds = time.time()

flag = False
######################################

class KalmanFilter():
    
    def __init__(self):
        global dt
        self.A = np.array([
                            [1, 0,       -dt,          0],
                            [0, 1,         0,        -dt],
                            [0, 0,         1,          0],
                            [0, 0,         0,          1]
                          ])

        self.B = np.array([
                            [dt,              0],
                            [       0,       dt],
                            [       0,        0],
                            [       0,        0]
                          ])

        self.C = np.array([
                            [1, 0, 0, 0],
                            [0, 1, 0, 0]
                          ])

        self.P_t = np.identity(4)*0.2
        self.Q_t = np.identity(4)        # process error covariance matrix
        self.R_t = np.identity(2)*0.2    # measurement error covariance matrix
        self.x_hat_t = np.zeros((4,1))

    def apply_kalman_filter(self):
        global crt_acc_data, prd_ang_data, a_roll, a_pitch

        observed_data = np.array([[a_roll],[a_pitch]])

        self.predict_stage()
        self.update_stage(observed_data)

        prd_ang_data = np.append(prd_ang_data, self.x_hat_t[0:2,:].transpose(), axis=0)
        prd_ang_data = prd_ang_data[-50:,:]

    def predict_stage(self):
        global crt_gyr_data, dt
        t_gyr_data = np.array([
                                [crt_gyr_data[0,0]],
                                [crt_gyr_data[0,1]]
                              ])
        self.x_hat_t = self.A.dot(self.x_hat_t) + self.B.dot(t_gyr_data)
        self.P_t = self.A.dot(self.P_t).dot(self.A.transpose()) + self.Q_t.dot(dt)

    def update_stage(self, y_t):
        K_t = self.P_t.dot(self.C.transpose()).dot(np.linalg.inv(self.C.dot(self.P_t).dot(self.C.transpose())+self.R_t))
        self.x_hat_t = self.x_hat_t + K_t.dot(y_t - self.C.dot(self.x_hat_t))
        self.P_t = self.P_t - K_t.dot(self.C).dot(self.P_t)

kf = KalmanFilter()


class Serv(BaseHTTPRequestHandler):

    def do_GET(self):
        global all_acc_ang_data, crt_acc_data, all_gyr_ang_data, crt_gyr_data,\
             a_pitch, a_roll, g_pitch, g_roll
        
        self.send_response(200)
        self.send_header('content_type', 'text/html')
        self.end_headers()
        
        # get data and assign them correctly
        _request = self.path[1:].split(",")
        r_acc_datas = _request[0:3]
        r_gyr_datas = _request[3:6]
        ind=0
        for acc_data, gyr_data in zip(r_acc_datas, r_gyr_datas):
            crt_acc_data[0,ind] = float(acc_data)
            crt_gyr_data[0,ind] = float(gyr_data) * (math.pi / 180.0)
            ind+=1

        # convert to angles in radian
        get_angles_from_gyro()
        get_angles_from_acc()

        # collect data for futher usage
        all_acc_ang_data = np.append(all_acc_ang_data, np.array([[a_roll],[a_pitch]]).transpose(), axis=0)
        all_acc_ang_data = all_acc_ang_data[-50:,:]
        all_gyr_ang_data = np.append(all_gyr_ang_data, np.array([[g_roll],[g_pitch]]).transpose(), axis=0)
        all_gyr_ang_data = all_gyr_ang_data[-50:,:]

        kf.apply_kalman_filter()

def get_angles_from_gyro():
    global old_seconds, g_roll, g_pitch, dt
    x = crt_gyr_data[0,1]
    y = crt_gyr_data[0,0]
    dt = time.time() - old_seconds
    old_seconds = time.time()
    g_roll = g_roll + x * dt
    g_pitch = g_pitch + y * dt

def get_angles_from_acc():
    global a_roll, a_pitch
    x = crt_acc_data[0,0]
    y = crt_acc_data[0,1]
    z = crt_acc_data[0,2]
    a_roll = math.atan2(x, z)
    a_pitch = math.atan2(y, z)

def init_web_server():
    httpd = HTTPServer(('192.168.1.75', 8080), Serv)
    httpd.serve_forever()

def press(event):
    global flag
    sys.stdout.flush()
    if event.key == 'q':
        flag = True

def plot_signal():
    global all_acc_ang_data, all_gyr_ang_data, flag, prd_ang_data

    # plot related initializations
    plt.close('all')
    fig, _ = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', press)
    plt.ion()
    plt.show()
    ###############################
    
    while not flag:   
        plt.cla()
        plt.xlim(left=5, right=50)
        plt.ylim(bottom=-math.pi, top=math.pi)
        plt.autoscale(False)

        # plot accelerometer x data
        # plt.plot(all_acc_ang_data[:,0])
        # plt.plot(all_gyr_ang_data[:,0])
        # plt.plot(prd_ang_data[:,0])

        # plot accelerometer y data
        plt.plot(all_acc_ang_data[:,1])
        plt.plot(all_gyr_ang_data[:,1])
        plt.plot(prd_ang_data[:,1])

        plt.legend(["a_x", "g_x", "p_x", "a_y", "g_y", "p_y"])

        plt.pause(0.01)

def main():
    
    data_get_th = threading.Thread(target=init_web_server)
    data_get_th.start()

    plot_signal_th = threading.Thread(target=plot_signal)
    plot_signal_th.start()

    data_get_th.join()
    plot_signal_th.join()

if __name__ == '__main__':
    main()