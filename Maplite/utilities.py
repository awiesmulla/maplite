import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import time

def get_data(req_dir, synchro, rest):

	imu = pd.read_csv(req_dir + "imu.csv")
	enc = pd.read_csv(req_dir + "wheel_control.csv")
	imu = np.array(imu)
	enc = np.array(enc)
	imu = imu[synchro[1]:]
	enc = enc[synchro[0]:]
	imu[:,2] = imu[:,2] - np.mean(imu[:rest,2])
	imu[:,3] = imu[:,3] - np.mean(imu[:rest,3])
	"""
	temp = []
	i=0
	while i < imu.shape[0]-1:
		accx=imu[i,2]#(imu[i,29]+imu[i+1,29])/2
		accy=imu[i,3]#(imu[i,30]+imu[i+1,30])/2
		wz=imu[i,7]#(imu[i,19]+imu[i+1,19])/2
		temp.append(np.array([accx, accy, wz]))
		i=i+1#2
	temp=np.array(temp)
	imu=temp
	"""

	return imu, enc

def lp_filter(arr, alpha = 0.02):
	fil_arr = [arr[0]]
	for a in arr[1:]:
	    fil_arr.append(alpha*a+(1-alpha)*fil_arr[-1])
	return fil_arr

def hp_filter(arr, alpha = 0.02):
	fil_arr = [arr[0]]
	for i in range(arr.shape[0]):
	    if i==0:
	        continue
	    fil_arr.append(alpha*fil_arr[-1]+(1-alpha)*(arr[i]-arr[i-1]))
	return fil_arr