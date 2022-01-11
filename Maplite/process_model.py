import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import time
import toml
from utilities import *

def fuse(state, yaw, u, z, P, F, G, R, sigma_u, H):

	x = np.array([state[0] + (0.01 * u[0]),
				  state[1] + (0.01 * u[1]),
				  u[2]])
	Q = np.dot(G, sigma_u)
	Q = np.dot(Q, np.transpose(G))

	temp_p = np.dot(F, P)
	temp_p = np.dot(temp_p, np.transpose(F))
	temp_p = temp_p + Q

	temp_z = np.dot(H, np.transpose(x))

	K = np.dot(temp_p, np.transpose(H))
	temp = np.dot(H, temp_p)
	temp = np.dot(temp, np.transpose(H))
	temp = np.linalg.pinv(temp + R)
	K = np.dot(K, temp)

	P=np.dot((np.eye(3) - np.dot(K, H)), temp_p)
	state = x + np.dot(K, (z - temp_z))
	yaw = yaw + (0.01 * state[2])

	return state, P, yaw

if __name__ == '__main__':

	params = (toml.load("config.toml"))

	img = cv2.imread(params.get("map"))

	parent_dir = params.get("parent_dir")
	data_dir = params.get("dataset_dir")
	req_dir = parent_dir + data_dir
	
	rest = params.get("rest")
	synchro = np.array(params.get("synchronize"))

	imu, enc = get_data(req_dir, synchro, rest)
	
	ax = lp_filter(imu[:,2])
	ay = lp_filter(imu[:,3])
	#g = np.mean(imu[:,4])
	wz = hp_filter(imu[:,7])

	vel = []
	velocity = np.array([0, 0, 0])
	start_pose = params.get("start")
	yaw = 1.57
	vel.append(velocity)

	sigma_u = np.array(params.get("sigma_u"))
	P = np.array(params.get("P"))
	R = np.array(params.get("R"))
	G = np.array(params.get("G"))
	F = np.array(params.get("F"))
	H = np.array(params.get("H"))

	for i in range(6000):#(min(enc.shape[0], imu.shape[0])):

		u = np.array([ax[i], ay[i], wz[i]])
		z = np.array([enc[i][2] * np.cos(yaw),
					  enc[i][2] * np.sin(yaw),
					  enc[i][3]])

		velocity, P, yaw = fuse(velocity, yaw, u, z, P, F, G, R, sigma_u, H)

		vel.append(velocity)

		#print(velocity)

	vel = np.array(vel)

	pose = []
	pose.append(np.array([0, 0]))

	print("==============================================================================")

	for i in range(vel.shape[0] - 1):
		temp = np.array([pose[-1][0]+(0.1*vel[i,0]),pose[-1][1]+(0.1*vel[i,1])])
		pose.append(temp)
		#print(temp)

	pose = np.array(pose)
	#plt.ylim(-150,950)
	#plt.xlim(-702,398)
	plt.scatter(pose[:,0] + start_pose[0], -pose[:,1] + start_pose[1], s=0.1)
	plt.imshow(img)
	plt.show()