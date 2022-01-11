# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 12:12:02 2021

@author: Awies
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import pykitti
import time

basedir = './../Datasets/KITTI-Dataset-master/KITTI/'
date = '2011_09_26'  
drive = '0059'

rw = 19.76
off_road = 8000
factor = 2.5
img = cv2.imread('maps/%s.png'%(drive))

fnfd = pd.read_csv("fd_tables/fd_%s.csv"%(drive))
fnfd = np.array(fnfd)

dataset = pykitti.raw(basedir, date, drive)
time_stamps = dataset.timestamps
calib = dataset.calib.T_velo_imu
oxts = dataset.oxts

state_x = 875#610#640
state_y = 990#770#1050

pose = []

var = 7
b = 10

pose.append(np.array([state_x, state_y]))
S = []

for n in range(len(oxts)):
    start = time.time()
    
    x = []
    y = []
    
    lid_x = []
    lid_y = []
    road  = []
    
    lidar = []
    alpha = oxts[n][0][5]
    vel_x = np.cos(alpha) * oxts[n][0][8]
    vel_y = np.sin(alpha) * oxts[n][0][8] 
    
    if n == 0:
        state_x = state_x + (vel_x / factor)
        state_y = state_y + (vel_y / factor)
        continue
    
    rot = np.array([[np.cos(alpha) , np.sin(alpha)],
                    [-np.sin(alpha) , np.cos(alpha)]])
    
    data = dataset.get_velo(n)
    
    for i in range(len(data)):
        temp = np.array([data[i][0] , data[i][1]])
        temp = np.dot(temp, rot)
        temp[0] = temp[0] + state_x
        temp[1] = temp[1] + state_y
        lidar.append(temp)        
    
    var += 4
    k = 0
    x.append(state_x)
    y.append(state_y)
    while not k >= 150:
        tempx = random.uniform(state_x - var, state_x + var)
        tempy = random.uniform(state_y - var, state_y + var)
        if ((tempx - state_x) ** 2 + (tempy - state_y) ** 2) <= (var ** 2):
            x.append(tempx)
            y.append(tempy)
            k = k + 1
    
    for i in range(len(lidar)):
        if i % 9000 == 0:
            for j in range(1500):
                if (i + j) < len(lidar):
                    lid_x.append(lidar[i + j][0])
                    lid_y.append(lidar[i + j][1])
                    if (i + j) <= off_road:
                        road.append(0)
                    else:
                        road.append(1)
    
    lid_x = np.array(lid_x)
    lid_y = np.array(lid_y)
    
    threshold = 0 
    correct = 0
    counter = 0
    for j in range(len(lid_x)):
        #j = random.randint(0 , (len(lid_x) - 1))
        if j % 20 == 0:
            counter += 1
            temp1 = np.int(lid_x[j])
            temp2 = np.int(img.shape[0] - lid_y[j])
            if temp1 >= (fnfd.shape[1] - 1):
                temp1 = fnfd.shape[1] - 2
            if temp2 >= (fnfd.shape[0] - 1):
                temp2 = fnfd.shape[0] - 2
            if temp1 < 0:
                temp1 = 0
            if temp2 < 0:
                temp2 = 0
            fd = fnfd[temp2 + 1][temp1 + 1]
            fd = 1 - min((fd / rw) , 1)
            """
            fd = (2.71828 ** (fd - rw))
            fd = fd + 1
            fd = 1 / fd
            """
            if not road[j]:
                fd = 1 - fd
            if fd >= 0.5:
                correct += 1
    threshold = correct
    
    da = []
    for i in range(len(x)):
        temp_correct = 0
        temp_counter = 0
        for j in range(len(lid_x)):
            #j = random.randint(0 , (len(lid_x) - 1))
            if j % 20 == 0:
                temp_counter += 1
                temp1 = np.int(lid_x[j] - x[0] + x[i])
                temp2 = np.int(img.shape[0] - (lid_y[j] - y[0] + y[i]))
                if temp1 >= (fnfd.shape[1] - 1):
                    temp1 = fnfd.shape[1] - 2
                if temp2 >= (fnfd.shape[0] - 1):
                    temp2 = fnfd.shape[0] - 2
                if temp1 < 0:
                    temp1 = 0
                if temp2 < 0:
                    temp2 = 0
                fd = fnfd[temp2 + 1][temp1 + 1]
                fd = 1 - min((fd / rw) , 1)
                """
                fd = (2.71828 ** (fd - rw))
                fd = fd + 1
                fd = 1 / fd
                """
                if not road[j]:
                    fd = 1 - fd
                if fd >= 0.5:
                    temp_correct += 1
        delta_x   = abs(x[i]-x[0])
        delta_y   = abs(y[i]-y[0])
        temp = np.array([delta_x, delta_y])
        odom = (2.71828 ** ((-np.linalg.norm(temp)) / b))
        score = temp_correct
        #score = score * odom
        da.append(score)
        if score > threshold:
            w1 = score * odom
            w2 = threshold
            state_x = ((w1 * x[i]) + (w2 * x[0])) / (w1 + w2)
            state_y = ((w1 * y[i]) + (w2 * y[0])) / (w1 + w2)
            correct = temp_correct
            counter = temp_counter
            threshold = score
        
    var = var * (np.log(counter / correct)) 
    
    state_x = state_x + (vel_x / factor)
    state_y = state_y + (vel_y / factor)
    
    S.append(da)
    pose.append(np.array([state_x, state_y]))
    
    print(n)
    print(time.time() - start)
    
pose = np.array(pose)
plt.plot(pose[:,0], pose[:,1])
plt.imshow(img, zorder=0, extent=([0,img.shape[1],0,img.shape[0]]))
plt.show()