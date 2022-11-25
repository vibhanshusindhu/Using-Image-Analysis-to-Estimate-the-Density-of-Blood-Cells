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
import shutil

def get_num(x):
    return int(x[0:-4])

good_files = []
good_file_names = []
gf = open("good_videos.txt", "r")
for ln in gf:
  good_files.append(int(ln.split()[0]))
  good_file_names.append(ln.split()[1])
gf.close()

print(good_file_names[good_files.index(132)])

stabilized_file_name = good_file_names[good_files.index(132)]
for gf_idx, stabilized_file_name in enumerate(good_file_names):
        pickeled_file_path = 'xy_data/' + stabilized_file_name.split("/")[-3] + "_" + stabilized_file_name.split("/")[-2] + "_xy.pkl"
        anno_file_name = stabilized_file_name.replace('stabilzedAvi.avi', 'annotation.json')
        with open(pickeled_file_path, 'rb') as f:
            video_file, video_points, y_points, sorted_end_frames, points_on_line, points_on_line_old, total_diff, bgr, mean_angles, (date2, hgb, wbc), first_frame, capil_mask, capil_contours = pickle.load(f)
            #video_file, video_points, y_points, points_on_line, total_diff, mean_angles, (date2, hgb, wbc) = pickle.load(f)
            if video_file != stabilized_file_name:
                print("STRANGE")
        try:
            f = open(anno_file_name)
            anno = json.load(f)
            f.close()
            annotation_keys = list(anno["annotations"]["annotations"].keys())
            wbc_end_frames = {}
            all_frames_with_bboxes = {}
            for annot_key in annotation_keys:
                N = int(annot_key)
                frames_with_bboxes = {}
                anno_location_keys = list(anno["annotations"]["annotations"][annotation_keys[N]]['locations'].keys())
                for akey in anno_location_keys:
                        a_dict = anno["annotations"]["annotations"][annotation_keys[N]]['locations'][akey]
                        frame_num = a_dict['frame']
                        if frame_num not in frames_with_bboxes.keys():
                            frames_with_bboxes[frame_num] = []
                        p1 = (int(a_dict['data'][0]['x']), int(a_dict['data'][0]['y']))
                        p2 = (int(a_dict['data'][1]['x']), int(a_dict['data'][1]['y']))
                        frames_with_bboxes[frame_num].append((p1, p2))
                        if frame_num not in all_frames_with_bboxes.keys():
                          all_frames_with_bboxes[frame_num] = []
                        all_frames_with_bboxes[frame_num].append((p1, p2))
                cross_frames = []
                for frame_num in frames_with_bboxes.keys():
                    for bbox in frames_with_bboxes[frame_num]:
                        for x, y in points_on_line:
                            if y < np.max([bbox[0][1], bbox[1][1]]) and y > np.min([bbox[0][1], bbox[1][1]]) and x > np.min([bbox[0][0], bbox[1][0]]) and x < np.max([bbox[0][0], bbox[1][0]]):
                                cross_frames.append(frame_num)
                                break
                if len(cross_frames) > 0:
                    wbc_end_frames[np.max(cross_frames)] = frames_with_bboxes[np.max(cross_frames)]
            sorted_end_frames = np.sort(list(wbc_end_frames.keys()))
        except:
            print("ERROR")

        image_folder = 'im_' + str(good_files[gf_idx])
        if os.path.isdir(image_folder):
            shutil.rmtree(image_folder)
        os.makedirs(image_folder)
        video_name = 'images_with_lines/test' + str(good_files[gf_idx]) + '.avi'

        vidcap = cv2.VideoCapture(stabilized_file_name)
        success,image_rgb = vidcap.read()
        frame_count = 0
        while success:
            # calculate difference and update previous frame
            if frame_count in all_frames_with_bboxes.keys():
                for p1, p2 in all_frames_with_bboxes[frame_count]:
                    image_rgb = cv2.rectangle(image_rgb, p1, p2, (255, 0, 255), 1)
            for p in points_on_line:
                image_rgb = cv2.circle(image_rgb, p, 2, (0, 255, 0), 1)
            cv2.imwrite(image_folder + "/" + str(frame_count) + ".png", image_rgb)
            success,image_rgb = vidcap.read()
            frame_count += 1

        vidcap.release()


        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        images = sorted(images, key = get_num)
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, channels = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(video_name, fourcc, 5.0, (width,height))

        for image in images:
            frame = cv2.imread(os.path.join(image_folder, image))
            frame = cv2.resize(frame, (width, height)) 
            video.write(frame)

        video.release()
        shutil.rmtree(image_folder)