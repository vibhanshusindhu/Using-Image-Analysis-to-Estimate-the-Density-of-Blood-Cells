import skimage
from skimage.transform import resize
import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (8,8)
import glob
import os
import json
import random
import pickle
from skimage.transform import resize

good_files = []
good_file_names = []
gf = open("good_videos.txt", "r")
for ln in gf:
  good_files.append(int(ln.split()[0]))
  good_file_names.append(ln.split()[1])
gf.close()

input_size = 36

for gf_idx, stabilized_file_name in enumerate(good_file_names):
    pickeled_file_path = 'xy_data/' + stabilized_file_name.split("/")[-3] + "_" + stabilized_file_name.split("/")[-2] + "_xy.pkl"
    with open(pickeled_file_path, 'rb') as f:
        video_file, video_points, y_points, sorted_end_frames, points_on_line, points_on_line_old, total_diff, bgr, mean_angles, (date2, hgb, wbc), first_frame, capil_mask, capil_contours = pickle.load(f)
        #video_file, video_points, y, points_on_line, total_diff, mean_angles, (date2, hgb, wbc) = pickle.load(f)
    video_points_copy = np.array(video_points).transpose().copy()
    video_points_copy = resize(video_points_copy, (input_size, video_points_copy.shape[1]))
    break_line = np.array([1] * video_points_copy.shape[0])
    for wbckey in np.where(np.array(y) == 1)[0]:
        video_points_copy[:, wbckey] = break_line

    #Plot data
   
    fig, axs = plt.subplots(int(video_points_copy.shape[1] / 350 + 1),1,figsize=(9,29))
    for idx, st in enumerate(range(0, video_points_copy.shape[1], 350)):
        axs[idx].axis("off")
        axs[idx].imshow(video_points_copy[:, st:st+350], cmap = 'gray')
    pic_name = stabilized_file_name.split("/")[-3] + "_" + stabilized_file_name.split("/")[-2] + "_" + str(good_files[gf_idx]) + "_rnn.png"
    plt.savefig("rnn_png/" + pic_name)
    plt.clf()