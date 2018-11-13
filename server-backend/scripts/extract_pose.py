#coding=utf-8

from poseEstimate import openpose
import os
import cv2
import pickle
import uniout

# try:
#     from openpose import *
# except:
#     raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
print(openpose)
file_path = '/media/yaosy/办公/bdmeet/mowei/mowei_squareMov' #'/media/yaosy/办公/bdmeet/m/m_movs'
save_Kpath = '/media/yaosy/办公/bdmeet/mowei/mowei_keypoints' #'/media/yaosy/办公/bdmeet/m/m_keypoints'
# save_Ipath =
if not os.path.exists(save_Kpath):
    os.mkdir(save_Kpath)
file_names = sorted(os.listdir(file_path))
for i, file in enumerate(file_names):
#     if i > 99:
#         break
    img = cv2.imread(os.path.join(file_path, file))
    keypoints, label = openpose.forward(img, True)
    with open(os.path.join(save_Kpath, file[:-3]+'pkl'), 'wb') as f:
        pickle.dump(keypoints, f)
    print('dump keypoints to pickle file {}'.format(file))
