import cv2
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,15)
import numpy as np
import pandas as pd
import os
import json

import glob

af = open("opflow_file.csv", "a")
line = ",".join(["Video","Frame Number", "Bbox Direction", "Flow Direction"])
af.write(line + "\n")

files = glob.glob('/mnt/litebc/patient-data' + '/**/**/annotation.json')
files2 = []
for i in range(len(files)):
    k = files[i].replace('annotation.json', 'stabilzedAvi.avi')   
    if os.path.exists(k):
        files2.append((files[i], k))

for anno_file, video_file in files2:
    try:
        f = open(anno_file)
        anno = json.load(f)
        f.close()
        annotation_keys = list(anno["annotations"]["annotations"].keys())
        if len(annotation_keys) == 0:
            continue
    except:
        print("ERROR", anno_file)
        continue

    frames_with_bboxes = {}
    for annot_key in annotation_keys:
        N = int(annot_key)
        anno_location_keys = list(anno["annotations"]["annotations"][annotation_keys[N]]['locations'].keys())
        for akey in anno_location_keys:
            a_dict = anno["annotations"]["annotations"][annotation_keys[N]]['locations'][akey]
            frame_num = a_dict['frame']
            if frame_num not in frames_with_bboxes.keys():
                frames_with_bboxes[frame_num] = []
            p1 = (int(a_dict['data'][0]['x']), int(a_dict['data'][0]['y']))
            p2 = (int(a_dict['data'][1]['x']), int(a_dict['data'][1]['y']))
            frames_with_bboxes[frame_num].append((p1, p2))

    frames_with_bboxes_key_list = list(frames_with_bboxes.keys())

    try:
        cap = cv2.VideoCapture(video_file)

        n_parts = video_file.split("/")
        new_name = n_parts[-3] + "_" + n_parts[-2] + "_frame_"

        prev_frame = frames_with_bboxes_key_list[0]
        bbox1 = frames_with_bboxes[prev_frame]
        cap.set(cv2.CAP_PROP_POS_FRAMES, prev_frame)
        ret, frame1 = cap.read()
        prvs = frame1[:, :, 0]
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        for frame_num in frames_with_bboxes_key_list:
            cur_frame = frame_num
            if cur_frame != prev_frame + 1:
                prev_frame = cur_frame
                bbox1 = frames_with_bboxes[prev_frame]
                cap.set(cv2.CAP_PROP_POS_FRAMES, prev_frame)
                ret, frame1 = cap.read()
                prvs = frame1[:, :, 0]
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
            bbox2 = frames_with_bboxes[cur_frame]
            ret, frame2 = cap.read()
            if not ret:
                print('No frames grabbed!')
                break

            next = frame2[:, :, 0]
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

            prvs = next

            frame2 = cv2.rectangle(frame2, bbox2[0][0], bbox2[0][1], (0, 1, 0), 1)
            cv2.imwrite("opt_flow/" + new_name + str(frame_num) + ".png", frame2)

            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            bbox_hsv = np.zeros_like(frame1)
            bbox_hsv[..., 1] = 255
            bbox_img1 = np.zeros_like(frame1,dtype=np.uint8)
            bbox_img1.fill(1)
            bbox_img1 = cv2.rectangle(bbox_img1, bbox1[0][0], bbox1[0][1], (0, 0, 0), -1)
            bbox_img2 = np.zeros_like(frame2,dtype=np.uint8)
            bbox_img2.fill(1)
            bbox_img2 = cv2.rectangle(bbox_img2, bbox2[0][0], bbox2[0][1], (0, 0, 0), -1)
            flow = cv2.calcOpticalFlowFarneback(bbox_img1[:, :, 0], bbox_img2[:, :, 0], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            bbox_mag, bbox_ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            bbox_hsv[..., 0] = bbox_ang*180/np.pi/2
            bbox_hsv[..., 2] = cv2.normalize(bbox_mag, None, 0, 255, cv2.NORM_MINMAX)
            bbox_bgr = cv2.cvtColor(bbox_hsv, cv2.COLOR_HSV2BGR)

            x1 = bbox2[0][0][0]
            y1 = bbox2[0][0][1]
            x2 = bbox2[0][1][0]
            y2 = bbox2[0][1][1]
            ang_bbox = np.mean(bbox_hsv[..., 0][y1:y2, x1:x2])
            ang_frame = np.mean(hsv[..., 0][y1:y2, x1:x2])
            line = ",".join([new_name + str(frame_num) + ".png",str(frame_num), str(ang_bbox), str(ang_frame)])
            af.write(line + "\n")

        cap.release()
    except:
        print("ERROR", video_file)
af.close()

