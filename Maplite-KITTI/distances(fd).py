# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:40:07 2021

@author: Awies Mohammad Mulla
"""
import numpy as np
import pandas as pd
import cv2

"""
Dimensions of image imported:
   width = 789
   height= 491
   RGBA color channel
Make appropriate changes to the code if using custom image.
"""

drive = '0059'
rw = 22
off_road = 30
#Read the topological map
img = cv2.imread('maps/%s.png'%(drive))
#Graph map as stored for navigation
gph = cv2.imread('maps/%s_graph.png'%(drive))
#Calculate distances at given angualr intervals
#(similar to LiDAR scanning)
alpha = 20
#Minimum distance where the edge can be present
threshold = np.array([3 , 0])
#Rotation matrix to rotate the ray calculating distance
rot = np.array([[np.cos(np.radians(alpha)) , np.sin(np.radians(alpha))],
              [-np.sin(np.radians(alpha)) , np.cos(np.radians(alpha))]])
#Points for which distances need to calculated to find the likelihood in the algorithm
pts = []
#Look-up table of the above points to reduce runtime
fd = np.zeros((img.shape[0],img.shape[1]))
#Compiling all distances of all pts in a list for debugging and analysing
distances = []

#Loop to find the pts from topo map whose signed distances needs to be calculated
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i][j][0] != 0:
            pts.append(np.array((i,j)))

#Find minimum distances of pts from the road edges
for pt in pts:
    #Initial vector to calculate distance
    ray = threshold
    #List of all distances from pt to road edge
    temp = []
    print(pt)
    #Loop to calculate distances at a pt
    for i in range(18):
        ray = ray / ((sum(ray ** 2)) ** 0.5) #unit vector of ray
        r = 3 #magnitude of the ray to determine the road edge
        ray = ray * r
        ray = np.array([(np.int(i)) for i in ray])
        x = ray[0] #x_component of the ray
        y = ray[1] #y_component of the ray
        #If the ray is out of bounds of the image
        #Set a distance larger than max possible signed distance
        #And move onto next ray at a pt
        if pt[0] + y >= img.shape[0] or pt[1] + x >= img.shape[1]:
            ray = np.dot(ray , rot)
            temp.append(off_road)
            continue
        if pt[0] + y < 0 or pt[1] + x < 0:
            ray = np.dot(ray , rot)
            temp.append(off_road)
            continue
        #Iterate along a ray to find the road edge
        while gph[pt[0] + y][pt[1] + x][0]:
            if r >= rw:
                break
            a = ray / ((sum(ray ** 2)) ** 0.5)
            a = a * r
            a = np.array([(np.int(i)) for i in a])
            r = r + 1
            x = a[0]
            y = a[1]
            if pt[0] + y >= img.shape[0] or pt[1] + x >= img.shape[1]:
                break
        temp.append((x ** 2 + y ** 2) ** 0.5)
        ray = ray / ((sum(ray ** 2)) ** 0.5)
        ray = ray * r
        #move onto next ray at a pt
        ray = np.dot(ray , rot)
    distances.append(np.array(temp))
    #Enter the minimum distance in the look up table
    fd[pt[0]][pt[1]] = np.amin(temp)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if fd[i][j] == 0:
            fd[i][j] = off_road

#Generate the look up table in csv format
data = pd.DataFrame(fd)
#save with appropriate name correspondingto map used
data.to_csv("fd_tables/fd_%s.csv"%(drive))