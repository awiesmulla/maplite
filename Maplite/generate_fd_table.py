# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 18:14:19 2021

@author: Awies
"""
import numpy as np
import numba
import pandas as pd
import cv2
import time

def signed_distance(pt, alpha, ray, rw, gph, off_road = 100):
    temp = []
    vect = []
    rot = np.array([[np.cos(np.radians(alpha)) , np.sin(np.radians(alpha))],
                    [-np.sin(np.radians(alpha)) , np.cos(np.radians(alpha))]])
    
    for i in range(int(360 / alpha)):
        ray = ray / ((sum(ray ** 2)) ** 0.5)
        r = 3
        ray = ray * r
        ray = np.array([(np.int(i)) for i in ray])
        x = ray[0]
        y = ray[1]
        if pt[0] + y >= gph.shape[0] or pt[1] + x >= gph.shape[1]:
            ray = np.dot(ray , rot)
            temp.append(off_road)
            continue
        if pt[0] + y < 0 or pt[1] + x < 0:
            ray = np.dot(ray , rot)
            temp.append(off_road)
            continue
        while gph[pt[0] + y][pt[1] + x][0]:
            if r >= rw:
                break
            a = ray / ((sum(ray ** 2)) ** 0.5)
            a = a * r
            a = np.array([(np.int(i)) for i in a])
            r = r + 1
            x = a[0]
            y = a[1]
            if pt[0] + y >= gph.shape[0] or pt[1] + x >= gph.shape[1]:
                break
        temp.append((x ** 2 + y ** 2) ** 0.5)
        vect.append(np.array([x , y]))
        ray = ray / ((sum(ray ** 2)) ** 0.5)
        ray = ray * r
        ray = np.dot(ray , rot)
        
    return temp, vect



if __name__ == '__main__':
    
    map_name = "farm3"
    img = cv2.imread('maps/%s.png'%(map_name))
    gph = cv2.imread('maps/%s_graph.png'%(map_name))
    
    rw = 12
    off_road = 25
    alpha = 20
    threshold = np.array([1 , 0])
    
    pts = []
    fd = np.zeros((img.shape[0],img.shape[1]))
    distances = []
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][0] != 0:
                pts.append(np.array((i , j)))
    
    start = time.time()
    for pt in pts:
        ray = threshold
        print(pt)
        temp, vect = signed_distance(pt, alpha, ray, rw, gph, off_road)
        print(np.amin(temp))
        distances.append(np.array(temp))
        fd[pt[0]][pt[1]] = np.amin(temp)
    
    print(time.time()-start)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if fd[i][j] == 0:
                fd[i][j] = off_road
       
    data = pd.DataFrame(fd)
    data.to_csv("fd_tables/fd_%s.csv"%(map_name))  
