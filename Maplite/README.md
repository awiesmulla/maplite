# Maplite

Here we use the wheel encoder, IMU, LiDAR and topometric map to localize the pose of the vehicle.  
You can find the report to walkthrough the mentioned algorithm [here](https://drive.google.com/file/d/17iGhsjx-3EBaIm70u2Qb7s1VyUAvLm-O/view?usp=sharing).  

## Preprocessing

To run maplite you would need a topometrric map (sparse graph-like map consisting of roads as edges and intersections as nodes). You can find the examples in the maps folder. From the map you would need to calculate the signed distances and these can be stored as csv files in fd_tables (these can be generated from `generate_fd_table.py`).  
You can adjust the required parameter in `config.toml`.  
`lidar_prep.py` is used to preprocess the LiDAR pointcloud such that we can segment the roads in the pointcloud.  
Use `map_generate.py ` to convert the topometric map to road map and vice-versa.  
`utilities.py` consists of various supplementary blocks like High Pass Filter, Low Pass Filter etc.  

## Usage

Run the following to plan a path between the coordinates on a map mentioned in config file  

    python3 astar.py
    
To combine the IMU and wheel encoder data for pose of vehicle using Extended Kalman Filter run 

    python3 process_model.py

To localize using Maplite run the following  

    python3 maplite.py
