import numpy as np
import matplotlib.pyplot as plt
import cv2

def a_star(src, dest, area_map, weight):
	path = []
	found = False
	open_list = []
	visit = np.zeros((area_map.shape[0], area_map.shape[1]))
	hist_cost = 1000 * np.ones((area_map.shape[0], area_map.shape[1]))
	temp = np.empty((), dtype=object)
	temp[()] = (0, 0)
	prev = np.full((area_map.shape[0],area_map.shape[1]), temp, dtype=object)
	
	hist_cost[src[0]][src[1]] = 0
	prev[src[0]][src[1]] = src
	open_list.append(src)
	visit[src[0]][src[1]] = 1

	while len(open_list):
		curr_dist = []
		for i in range(len(open_list)):
			pt = open_list[i]
			curr_dist.append(hist_cost[pt[0]][pt[1]])

		curr_dist = np.array(curr_dist)
		i = min_coord(curr_dist, open_list, visit, dest, weight)
		pt = open_list[i]

		prev_pt = prev[pt[0]][pt[1]]

		hist_cost[pt[0]][pt[1]] = hist_cost[prev_pt[0]][prev_pt[1]] + 1

		if (pt[0] == dest[0]) and (pt[1] == dest[1]):
			found = True
			break

		#8 possible routes
		
		if (not visit[pt[0]+1][pt[1]]) and (not area_map[pt[0]+1][pt[1]][0]):
			open_list.append(np.array([pt[0] + 1, pt[1]]))
			visit[pt[0]+1][pt[1]] = 1
			prev[pt[0]+1][pt[1]] = pt
		if (not visit[pt[0]+1][pt[1]-1]) and (not area_map[pt[0]+1][pt[1]-1][0]):
			open_list.append(np.array([pt[0] + 1, pt[1] - 1]))
			visit[pt[0]+1][pt[1]-1] = 1
			prev[pt[0]+1][pt[1]-1] = pt
		if (not visit[pt[0]+1][pt[1]+1]) and (not area_map[pt[0]+1][pt[1]+1][0]):
			open_list.append(np.array([pt[0] + 1, pt[1] + 1]))
			visit[pt[0]+1][pt[1]+1] = 1
			prev[pt[0]+1][pt[1]+1] = pt
		if (not visit[pt[0]-1][pt[1]]) and (not area_map[pt[0]-1][pt[1]][0]):
			open_list.append(np.array([pt[0] - 1, pt[1]]))
			visit[pt[0]-1][pt[1]] = 1
			prev[pt[0]-1][pt[1]] = pt
		if (not visit[pt[0]-1][pt[1]+1]) and (not area_map[pt[0]-1][pt[1]+1][0]):
			open_list.append(np.array([pt[0] - 1, pt[1] + 1]))
			visit[pt[0]-1][pt[1]+1] = 1
			prev[pt[0]-1][pt[1]+1] = pt
		if (not visit[pt[0]-1][pt[1]-1]) and (not area_map[pt[0]-1][pt[1]-1][0]):
			open_list.append(np.array([pt[0] - 1, pt[1] - 1]))
			visit[pt[0]-1][pt[1]-1] = 1
			prev[pt[0]-1][pt[1]-1] = pt
		if (not visit[pt[0]][pt[1]+1]) and (not area_map[pt[0]][pt[1]+1][0]):
			open_list.append(np.array([pt[0], pt[1] + 1]))
			visit[pt[0]][pt[1]+1] = 1
			prev[pt[0]][pt[1]+1] = pt
		if (not visit[pt[0]][pt[1]-1]) and (not area_map[pt[0]][pt[1]-1][0]):
			open_list.append(np.array([pt[0], pt[1] - 1]))
			visit[pt[0]][pt[1]-1] = 1
			prev[pt[0]][pt[1]-1] = pt

		#4 possible routes
		
		"""
		if (not visit[pt[0]+1][pt[1]]) and (area_map[pt[0]+1][pt[1]][0]):
			open_list.append(np.array([pt[0] + 1, pt[1]]))
			visit[pt[0]+1][pt[1]] = 1
			prev[pt[0]+1][pt[1]] = pt
		if (not visit[pt[0]-1][pt[1]]) and (area_map[pt[0]-1][pt[1]][0]):
			open_list.append(np.array([pt[0] - 1, pt[1]]))
			visit[pt[0]-1][pt[1]] = 1
			prev[pt[0]-1][pt[1]] = pt
		if (not visit[pt[0]][pt[1]+1]) and (area_map[pt[0]][pt[1]+1][0]):
			open_list.append(np.array([pt[0], pt[1] + 1]))
			visit[pt[0]][pt[1]+1] = 1
			prev[pt[0]][pt[1]+1] = pt
		if (not visit[pt[0]][pt[1]-1]) and (area_map[pt[0]][pt[1]-1][0]):
			open_list.append(np.array([pt[0], pt[1] - 1]))
			visit[pt[0]][pt[1]-1] = 1
			prev[pt[0]][pt[1]-1] = pt
		"""

		open_list = open_list[:i] + open_list[(i+1):]

	if found:
		current = dest
		while not ((current[0] == src[0]) and (current[1] == src[1])):
			path.append(current)
			current = prev[current[0]][current[1]]

	return path, found

def min_coord(curr_dist, open_list, visit, dest, weight):
	cost = np.ones(curr_dist.shape) * 1000
	for i in range(len(open_list)):
		cost[i] = curr_dist[i] + (weight * heuristic(open_list[i], dest)) + (1000 * visit[open_list[i][0]][open_list[i][1]])
	return np.argmin(cost)

def heuristic(pt1, pt2):
	return np.linalg.norm(pt1 - pt2)

if __name__ == "__main__":

	area_map = cv2.imread("maps/factory_graph.png")
	img = cv2.imread("maps/factory.png")
	src = np.array([1000, 700])
	dest = np.array([110, 454])
	weight = 1
	path, found = a_star(src, dest, area_map, weight)
	if found:
		path = np.array(path)
		plt.scatter(path[:,1], path[:,0], s=0.1)
		plt.imshow(img)
		plt.show()
	else:
		print("Path not found")