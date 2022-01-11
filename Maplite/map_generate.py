# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 17:09:26 2021

@author: Awies
"""
from skimage.morphology import skeletonize
import cv2

def map_to_graph(map_name):
    img = cv2.imread("maps/%s.png"%(map_name))
    gph = skeletonize(img)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if not gph[i][j][1]:
                gph[i][j][:] = 255
            else:
                gph[i][j][:] = 0
    
    cv2.imwrite("maps/%s_graph.png"%(map_name), gph)

def graph_to_map(map_name, rw):
    gph = cv2.imread("maps/%s_graph.png"%(map_name))
    
    for i in range(gph.shape[0]):
        for j in range(gph.shape[1]):
            if gph[i][j][1]==255:
                gph[i][j][:] = 0
            else:
                gph[i][j][:] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (rw,rw))
    dilate = cv2.dilate(gph, kernel, iterations=1)
    cv2.imwrite("maps/%s.png"%(map_name), dilate)
    
if __name__ == '__main__':
    map_name = "farm3"
    rw = 20
    #map_to_graph(map_name)
    graph_to_map(map_name, rw)

