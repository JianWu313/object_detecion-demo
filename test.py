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
from nuscenes.utils.geometry_utils import box_in_image, view_points, BoxVisibility
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
def reclass(original_name):
    #because nusenes dataset has totally 21 classes ,which is too much
    #I project it to 7 classes:car/bus/motorbycle/truck/trailer/bicyle/human
    #it will also return the color of different class
    if "human" in original_name:
        return (255,0,0),"human" #the color of bbox is blue
    if "bus"  in original_name:
        return (0,255,255),"bus" #bus is yellow
    if "truck" in original_name:
        return (0,165,255),"truck" #track is orange   
    if "trailer" in original_name:
        return (0,255,0),"trailer" #trailer is green
    if "motorcycle" in original_name:
        return (0,100,0),"motorcycle"   #motorcycle is deep green
    if "bicyle" in original_name:
        return (71,99,255),"bicyle"#bike is tomato
    if "car" in original_name:#car is red
        return (0,0,255),"car"
    return (255,255,255),"others" #other is white

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
#then visualize all the points
#note that here in order to show the data better ,i have scale the coordinate
#because all the x >0,but y can smaller as 0,so i move the center to (0,500)in numpy array
expand_factor=8;
img=np.zeros((1000,1000,3),dtype=np.uint8)
rows=np.around(expand_factor*radar_data[1]+500).astype(np.int32)
cols=np.around(expand_factor*radar_data[0]).astype(np.int32)
img[rows,cols]=(255,0,255)
#at last we visualize the bounding box in sensor frame
for box in boxes:
    color,label=reclass(box.name)
    bottom_corners=box.bottom_corners()
    #filter out the boxes outside the canvas,first check whether x<0
    is_outside=False
    for x in bottom_corners[0]:
        if x<=0 or x*expand_factor>=1000:
            is_outside=True
            break
        
    for y in bottom_corners[1]:
        if abs(y*expand_factor)>=500:
            is_outside=True
            break
    if(is_outside):
         continue
    bottom_corners[0]=bottom_corners[0]*expand_factor
    bottom_corners[1]=(bottom_corners[1])*expand_factor+500
    pts=(bottom_corners[0:2,].T).astype(np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img, [pts],True,color, thickness=1)
cv2.imshow("point cloud",img)
cv2.waitKey(0)









