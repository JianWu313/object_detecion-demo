# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 17:25:46 2020

@author: wujian
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""
from nuscenes.utils.data_classes import RadarPointCloud, Box
from nuscenes.utils.geometry_utils import box_in_image, points_in_box, BoxVisibility
import nuscenes
import numpy as np
import os.path as osp
import cv2 
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, transform_matrix
#plz change the belowing parameters to search different situations 
required_sweep_count=13
scene_index=5
sample_index=5
def points_in_box(box: 'Box', points: np.ndarray, wlh_factor: float = 1.0):
    """
    note that here I have reduce the dimension of z only to find x,y plane
    the original function see below:https://github.com/Fellfalla/nuscenes-devkit/
    blob/1e1b3d6320d7d9b0eca05969a316b0bd747d7e95
    /python-sdk/nuscenes/utils/geometry_utils.py
    """
    corners = box.corners(wlh_factor=wlh_factor)

    p1 = corners[:, 0]
    p_x = corners[:, 4]
    p_y = corners[:, 1]

    i = p_x - p1
    j = p_y - p1

    v = points - p1.reshape((-1, 1))

    iv = np.dot(i, v)
    jv = np.dot(j, v)

    mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
    mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
    mask = np.logical_and(mask_x, mask_y)

    return mask

def get_sensor_sample_data(nusc, sample, sensor_channel, dtype=np.float32, size=None):
    """
    This function takes the token of a sample and a sensor sensor_channel and returns the according data
    :param sample: the nuscenes sample dict
    :param sensor_channel: the target sensor channel of the given sample to load the data from
    :param dtype: the target numpy type
    :param size: for resizing the image
    Radar Format:
        - Shape: 19 x n
        - Semantics: 
            [0]: x (1)
            [1]: y (2)
            [2]: z (3)
            [3]: dyn_prop (4)
            [4]: id (5)
            [5]: rcs (6)
            [6]: vx (7)
            [7]: vy (8)
            [8]: vx_comp (9)
            [9]: vy_comp (10)
            [10]: is_quality_valid (11)
            [11]: ambig_state (12)
            [12]: x_rms (13)
            [13]: y_rms (14)
            [14]: invalid_state (15)
            [15]: pdh0 (16)
            [16]: vx_rms (17)
            [17]: vy_rms (18)
            [18]: distance (19)

    """
    # Get filepath
    sd_rec = nusc.get('sample_data', sample['data'][sensor_channel])
    file_name = osp.join(nusc.dataroot, sd_rec['filename'])

    # Check conditions
    if not osp.exists(file_name):
        raise FileNotFoundError(
            "nuscenes data must be located in %s" % file_name)

    # Read the data
    if "RADAR" in sensor_channel:
        pcs, times = RadarPointCloud.from_file_multisweep(nusc, sample, sensor_channel, \
                sensor_channel, nsweeps=required_sweep_count, min_distance=0.0)  # Load radar points
        data = pcs.points.astype(dtype)
       
    else:
        raise Exception("\"%s\" is not supported" % sensor_channel)

    return data

nusc = NuScenes(version='v1.0-mini', dataroot='D:\\nuscenes', verbose=True)
my_scene = nusc.scene[0]
sample_tokens = {}
nbr_samples = nusc.scene[scene_index]['nbr_samples']
first_sample_token = my_scene['first_sample_token']
curr_sample = nusc.get('sample', first_sample_token)
prog=0
for _ in range(nbr_samples):
    sample_tokens[prog] = curr_sample['token']
    if curr_sample['next']:
        next_token = curr_sample['next']
        curr_sample = nusc.get('sample', next_token)
    prog += 1
my_sample_token = sample_tokens[sample_index];
my_sample=nusc.get('sample', my_sample_token)
sensor_channel="RADAR_FRONT";
my_sample_data=nusc.get('sample_data', my_sample['data'][sensor_channel])
radar_data=get_sensor_sample_data(nusc, my_sample, sensor_channel)
my_annotation_token = my_sample['anns'][18]
#note that the annotation here is in globar coordinate symstem
my_annotation_metadata =  nusc.get('sample_annotation', my_annotation_token)
#pay attention to the class Box in nuscenes-devkit
#in get_sample_data will transform the coordinate from global to sensor frame
_, boxes, _ = nusc.get_sample_data(my_sample_data['token'])
box_list=[]

for box in boxes:
    mask = points_in_box(box, radar_data[0:3,:])
    #caculate the number of points contains in a bounding box
    num_of_pts=np.count_nonzero(mask)
    if num_of_pts !=0:
       box_list.append((box.name,num_of_pts))
print(box_list)    








