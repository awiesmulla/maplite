import numpy as np
import numba
import random
import matplotlib.pyplot as plt
import pandas as pd
import time
import toml
import cv2

from utilities import *
from lidar_prep import gen_voxel, z_grad_obstacle
from process_model import fuse

@numba.njit(fastmath=True)
def score_fn(ground_truth, sampled_pose, dtheta, lidar, road, fnfd, start, rw, b):

    """
    Calculate the weights of each sampled pose
    from results of segmentation and pointcloud
    (Refer report for function)
    """
    
    score = 0
    phi_req = 0
    odom = 1

    for j in range(3):
        correct = 0
        phi = (j - 1) * dtheta
        x, y, theta = sampled_pose
        theta += phi
        rotation_matrix = np.array([[np.cos(theta) , np.sin(theta)],
                                    [-np.sin(theta) , np.cos(theta)]])
        for i in range(lidar.shape[0]):
            if i % 1 == 0:
                scan = np.array([lidar[i,0] , lidar[i,1]])
                scan = np.dot(scan, rotation_matrix)
                scan = scan + sampled_pose[:2]
                x_id = int(scan[0] + start[0])
                y_id = int(start[1] - scan[1])
                if x_id >= (fnfd.shape[1] - 1):
                    x_id = fnfd.shape[1] - 2
                if y_id >= (fnfd.shape[0] - 1):
                    y_id = fnfd.shape[0] - 2
                if x_id < 0:
                    x_id = 0
                if y_id < 0:
                    y_id = 0
                fd = fnfd[y_id + 1][x_id + 1]
                fd = (2.71828 ** ((fd - rw)))
                fd = fd + 1
                fd = 1 / (fd)
                if not road[i]:
                    fd = 1 - fd
                correct += fd
        
        delta = sampled_pose - ground_truth
        delta[2] += phi
        temp_odom = (2.71828 ** ((-np.linalg.norm(delta)) / b))
        correct = correct * temp_odom
        if correct > score:
            score = correct
            phi_req = phi
            odom = temp_odom
        
    return score, odom, phi_req

def lidar_preprocess(on_road, off_road, phi):

    """
    Assigning label
    road[0] = off-road
    road[1] = on-road
    """
    
    lidar = []
    road  = []
    rot = np.array([[np.cos(phi),np.sin(phi)],
                    [-np.sin(phi),np.cos(phi)]])

    data = list(on_road) + list(off_road)
    data = np.array(data)
    
    for i in range(data.shape[0]):
        temp = np.array([data[i][0],data[i][1]])
        dist = np.linalg.norm(temp)
        if dist < 3000:
            if i < on_road.shape[0]:
                road.append(1)
            else:
                road.append(0)
            lidar.append(np.dot(temp,rot))
    
    lidar = np.array(lidar)
    road  = np.array(road)
    
    return lidar, road
            

if __name__ == '__main__':

    #Import all parameters

    params = (toml.load("config.toml"))
    
    img = cv2.imread(params.get("map"))                            #Map for visualising

    fnfd = pd.read_csv(params.get("fd_table"))                     #fd table (refer report)
    fnfd = np.array(fnfd)

    parent_dir = params.get("parent_dir")
    data_dir = params.get("dataset_dir")
    req_dir = parent_dir + data_dir
    label_dir = params.get("lidar_label")
    
    rest = params.get("rest")                                      #Remove readings for when robot at rest
    synchro = np.array(params.get("synchronize"))

    imu, enc = get_data(req_dir, synchro, rest)                    #import IMU, wheel encoder data

    ax = lp_filter(imu[:,2])                                       #Pass acceleration through low-pass filter
    ay = lp_filter(imu[:,3])
    g = np.mean(imu[:,4])
    wz = hp_filter(imu[:,7])                                       #Pass angular velocity through high-pass filter

    rw = params.get("road_width")
    dt = params.get("dt")

    #EKF parameters

    sigma_u = np.array(params.get("sigma_u"))
    P = np.array(params.get("P"))
    R = np.array(params.get("R"))
    G = np.array(params.get("G"))
    F = np.array(params.get("F"))
    H = np.array(params.get("H"))

    #Initialize the variables (states, control etc.)

    states = []
    vel = []
    velocity = np.array([0, 0, 0])
    pose = np.zeros(3)
    temp_pose = pose
    states.append(pose)
    vel.append(velocity)
    start_pose = np.array(params.get("start"))
    dtheta = params.get("d_theta")
    yaw = 1.57

    b = params.get("b")
    var = params.get("variance")
    scan_id = 0

    for n in range(min(imu.shape[0], enc.shape[0])):

        if n < rest:
            if n % 10 == 0:
                scan_id += 1
            continue

        start_time = time.time()

        #EKF implemented

        u = np.array([ax[n], ay[n], wz[n]])
        z = np.array([enc[n][2] * np.cos(yaw),
                      enc[n][2] * np.sin(yaw),
                      enc[n][3]])

        velocity, P, yaw = fuse(velocity, yaw, u, z, P, F, G, R, sigma_u, H)         #EKF fusion
        vel.append(velocity)
        temp_pose[:2] = pose[:2] + (dt * velocity[:2] * 10)
        temp_pose[2] = yaw

        if (n % 10 == 0):

            #List for storing the sampled poses

            x = []
            y = []
            alpha = temp_pose[2]
            
            var += 15
            
            #For LiDAR segmentation

            data = np.load(req_dir + "lidar/frame-%05d.npy"%(scan_id))

            data = data.astype('int16')
            data = data / 100
            voxel_size = 0.2
            
            voxel_grid, _, lidar = gen_voxel(data, voxel_size)
            road, off_road, _ = z_grad_obstacle(voxel_grid)

            lidar, road = lidar_preprocess(road, off_road, -1.57)

            k = 0
            x.append(temp_pose[0])
            y.append(temp_pose[1])

            while not k >= params.get("sample_number"):
                tempx = random.uniform(pose[0] - var, pose[0] + var)
                tempy = random.uniform(pose[1] - var, pose[1] + var)
                if ((tempx - pose[0]) ** 2 + (tempy - pose[1]) ** 2) <= (var ** 2):
                    x.append(tempx)
                    y.append(tempy)
                    k = k + 1

            #Score (weights) of the mean position

            threshold, odom, phi_gt = score_fn(temp_pose, temp_pose, dtheta, lidar, road, fnfd, start_pose, rw, b)
            score = threshold
            phi = phi_gt
            
            #Score (weights) of all the sampled points

            for i in range(len(x)):
                temp = np.array([x[i], y[i], (alpha)])

                temp_score, temp_odom, temp_phi = score_fn(temp_pose, temp, dtheta, lidar, road, fnfd, start_pose, rw, b)

                if temp_score > score:
                    odom = temp_odom
                    score = temp_score
                    phi = temp_phi
                    pose = temp
                    print("------------------")
                    #pose[2] = alpha + phi
            
            #Calculate the final pose

            w1 = score #* odom
            w2 = threshold
            pose = temp_pose#((w1 * pose) + (w2 * temp_pose)) / (w1 + w2)
            pose[2] = alpha + phi
            yaw = pose[2]

            var = var * (np.log(w1 / w2))
            scan_id += 1

        else:
            pose = temp_pose
        states.append(pose)
        
        #print(pose)
        print(n)
        #print(time.time() - start_time)

    vel = np.array(vel)
    poses = []
    poses.append(np.array([0, 0]))

    for i in range(vel.shape[0] - 1):
        temp = np.array([poses[-1][0]+(0.1*vel[i,0]),poses[-1][1]+(0.1*vel[i,1])])
        poses.append(temp)
        #print(temp)
    states = np.array(states)
    poses = np.array(poses)
    plt.scatter(states[:,0] + start_pose[0], -states[:,1] + start_pose[1], s=0.1, color="red")
    plt.scatter(poses[:,0] + start_pose[0], -poses[:,1] + start_pose[1], s=0.1, color="blue")
    plt.imshow(img)
    plt.show()