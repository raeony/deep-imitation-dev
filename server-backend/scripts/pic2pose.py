#coding=utf-8

from poseEstimateFileSave import poseEstimate
import os
import cv2
import pickle
import uniout

file_path = '/media/yaosy/办公/bdmeet/maskGirl/m_mov480x480'
save_grayPath = '/media/yaosy/办公/bdmeet/maskGirl/m_label480x480_gray'
save_rgbPath = '/media/yaosy/办公/bdmeet/maskGirl/m_label480x480_rgb'

filename_list = sorted(os.listdir(file_path))

for i, filename in enumerate(filename_list):
    input_img = cv2.imread(os.path.join(file_path, filename))
    print(filename)
    gray_img, rgb_img = poseEstimate(input_img, is_square=False)
    cv2.imwrite(os.path.join(save_grayPath, filename), gray_img)
    cv2.imwrite(os.path.join(save_rgbPath, filename), rgb_img)

