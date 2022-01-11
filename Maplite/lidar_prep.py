import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from mpl_toolkits import mplot3d

def plane_fit(X, Z):

	X_transpose = X.transpose()
	XTX = X_transpose.dot(X)
	XTZ = X_transpose.dot(Z)
	model = (np.linalg.pinv(XTX)).dot(XTZ) 
	return model

def ransac(scan):

	temp_x = scan[: , 0]
	temp_y = scan[: , 1]

	X = np.stack((temp_x, temp_y, np.ones((len(temp_x)), dtype = int)), axis = 1)
	Z = scan[: , 2]

	num_iterations = math.inf
	iterations_done = 0
	num_sample = 3
	threshold = np.std(Z) / 5

	max_inlier_count = 0
	best_model = None

	prob_outlier = 0.5
	desired_prob = 0.95

	total_data = np.column_stack((X, Z))
	data_size = len(total_data)

	while num_iterations > iterations_done:

		np.random.shuffle(total_data)
		sample_data = total_data[:num_sample, :]

		estimated_model = plane_fit(sample_data[:,:-1], sample_data[:, -1:])

		z_cap = X.dot(estimated_model)
		err = np.abs(Z - z_cap.T)
		inlier_count = np.count_nonzero(err < threshold)

		if inlier_count > max_inlier_count:
		    max_inlier_count = inlier_count
		    best_model = estimated_model


		prob_outlier = 1 - inlier_count/data_size
		num_iterations = math.log(1 - desired_prob)/math.log(1 - (1 - prob_outlier)**num_sample)
		iterations_done = iterations_done + 1

	temp = X.dot(best_model)
	error = np.abs(Z - temp.T)
	error = np.reshape(error, (-1,))
	ground_pts = []
	
	for i in range(len(temp_x)):
		if error[i] < threshold:
			ground_pts.append(np.array(scan[i]))

	ground_pts = np.array(ground_pts)

	return ground_pts

##############################Reduce complexity################################################
def z_grad_obstacle(voxel_grid):

	keys = list(voxel_grid.keys())
	keys = np.array(keys)
	grid_size = np.amax(keys, axis = 0)
	grid_map = np.zeros((grid_size[1], grid_size[0]))
	temp = {}
	test_keys = {}
	road = []
	off_road = []
	test = []

	for i in range(keys.shape[0]):
		if (keys[i][0], keys[i][1]) in temp:
			for j in range(voxel_grid[tuple(keys[i])].shape[0]):
				temp[(keys[i][0],keys[i][1])].append(voxel_grid[tuple(keys[i])][j])
		else:
			temp[(keys[i][0],keys[i][1])] = []
			for j in range(voxel_grid[tuple(keys[i])].shape[0]):
				temp[(keys[i][0],keys[i][1])].append(voxel_grid[tuple(keys[i])][j])

	for key in temp.keys():
		temp[tuple(key)] = np.array(temp[tuple(key)])
		test_keys[tuple(key)] = 1

	for i in range(grid_map.shape[0]):
		for j in range(grid_map.shape[1]):
			if not ((j,i) in temp):
				temp[(j,i)] = np.array([[0, 0, -1.85]])

	for i in range(grid_map.shape[0]):
		if i == 0 or i == (grid_map.shape[0]-1):
			continue
		for j in range(grid_map.shape[1]):
			if j == 0 or j == (grid_map.shape[1]-1):
				continue
			if (j,i) in test_keys:
				g_ref = np.amin(np.array([np.amin(temp[(j-1,i+1)], axis=0), np.amin(temp[(j+1,i)], axis=0), np.amin(temp[(j-1,i)], axis=0), np.amin(temp[(j,i+1)], axis=0), np.amin(temp[(j,i)], axis=0),
								np.amin(temp[(j+1,i-1)], axis=0), np.amin(temp[(j-1,i-1)], axis=0), np.amin(temp[(j+1,i+1)], axis=0), np.amin(temp[(j,i-1)], axis=0)]), axis=0)
				delta = np.amax((temp[(j,i)] - g_ref), axis=0)[2]
				point = temp[(j,i)][np.argmax((temp[(j,i)] - g_ref), axis=0)[2]]
				test.append(delta)

				#(minimum delta should be [2 - 0.25, 2 + 0.25])
				if delta > 1.75 and delta < 8:
					off_road.append(point)
				else:
					road.append(point)

	road = np.array(road)
	off_road = np.array(off_road)
	test = np.array(test)


	return road, off_road, test
##############################Reduce complexity################################################

def gen_voxel(points, voxel_size = 0.1):

	nb_vox=np.ceil((np.max(points, axis=0) - np.min(points, axis=0))/voxel_size)
	
	non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((points - np.min(points, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
	idx_pts_vox_sorted=np.argsort(inverse)

	voxel_grid={}
	grid_barycenter,grid_candidate_center=[],[]
	last_seen=0

	for idx,vox in enumerate(non_empty_voxel_keys):
		voxel_grid[tuple(vox)]= points[idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]]
		grid_barycenter.append(np.mean(voxel_grid[tuple(vox)],axis=0))
		grid_candidate_center.append(voxel_grid[tuple(vox)][np.linalg.norm(voxel_grid[tuple(vox)] - np.mean(voxel_grid[tuple(vox)],axis=0),axis=1).argmin()])
		last_seen+=nb_pts_per_voxel[idx]

	grid_barycenter = np.array(grid_barycenter)
	grid_candidate_center = np.array(grid_candidate_center)

	return voxel_grid, grid_barycenter, grid_candidate_center

if __name__ == "__main__":

	scan_id = 421
	data = np.load("./../Datasets/factory/factory/lidar/frame-%05d.npy"%(scan_id))
	#./../Datasets/farm/farm/lidar/frame-%05d.npy
	#/media/amm/Backup/acads/ARTPARK/test/transvahan/Data/2021-12-09/run1_lidar/%05d.npy
	data = data[:,:3]
	temp = []
	"""
	for i in range(data.shape[0]):
		if data[i,2] < -0.3:
			temp.append(data[i])
	data = np.array(temp)
	"""
	data = data.astype('int16')
	data = data / 100
	voxel_grid, mean_representative, closest_to_grid = gen_voxel(data)
	road, off_road, test = z_grad_obstacle(voxel_grid)
	#ground = ransac(mean_representative)
	#plt.scatter(mean_representative[:,0], mean_representative[:,1], s=0.1)
	plt.scatter(road[:,0],road[:,1],s=0.1, color="blue")
	#plt.scatter(data[:,0], data[:,1],s=0.1)
	"""
	for i in range(off_road.shape[0]):
		if off_road[i,2] < -0.5:
			temp.append(off_road[i])
	off_road = np.array(temp)
	"""
	plt.scatter(off_road[:,0],off_road[:,1],s=0.1, color="red")
	#ax = plt.axes(projection='3d')
	#ax.scatter(data[:,0]+30, data[:,1], data[:,2], s=0.01)
	#ax.scatter(road[:,0], road[:,1],road[:,2],s=0.01)
	#ax.scatter(mean_representative[: , 0], mean_representative[: , 1], mean_representative[: , 2], s=0.01, color="green")
	plt.show()