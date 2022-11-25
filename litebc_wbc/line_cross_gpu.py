import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (15,15)
import glob
import os
import json
import random
import pickle

import torch, detectron2

from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset_train", {}, "/home/user_gpu_60/elenas_dir/capil_coco/annotations/instances_default.json", "/home/user_gpu_60/elenas_dir/capil_coco/images")
register_coco_instances("my_dataset_val", {}, "/home/user_gpu_60/elenas_dir/capil_coco_test/annotations/instances_default.json", "/home/user_gpu_60/elenas_dir/capil_coco_test/images")


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "/mnt/litebc/elena-files/model_final.pth"  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 8  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 2400    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (capillary). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.SOLVER.CHECKPOINT_PERIOD = 400
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set a custom testing threshold
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
predictor = DefaultPredictor(cfg)

def get_capil_contour(predictor, total_diff):
  im = np.zeros((total_diff.shape[0],total_diff.shape[1],3))
  im[:, :, 0] = total_diff
  im[:, :, 1] = total_diff
  im[:, :, 2] = total_diff

 
  outputs = predictor(im) 

  mask = None
  if len(outputs['instances']) == 0:
    mask = None
  elif len(outputs['instances']) > 1:
    max_index = torch.argmax(outputs['instances'].scores).item()
    mask = outputs['instances'][max_index].pred_masks.squeeze(0)
    mask = mask.detach().cpu().numpy()
  else:
    mask = outputs['instances'][0].pred_masks.squeeze(0)
    mask = mask.detach().cpu().numpy()

  contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  return mask, contours

def get_line(img, contours, mask):
  mask = 255 * mask.astype(np.uint8)
  thresh = mask.copy()
  contour = None
  if len(contours) > 0:
      max_area = 2048
      for c in contours:
        ar = cv2.contourArea(c)
        if ar > max_area:
          max_area = ar
          contour = c
  if contour is not None:
      #print("CONT", contour)
      # Step 1: Create an empty skeleton
      #ret,thresh = cv2.threshold(img.astype(np.uint8), 127, 255, 0)
      skel = np.zeros(thresh.shape, np.uint8)

      # Get a Cross Shaped Kernel
      element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

      # Repeat steps 2-4
      while True:
          #Step 2: Open the image
          open_morth = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, element)
          #Step 3: Substract open from the original image
          temp = cv2.subtract(thresh, open_morth)
          #Step 4: Erode the original image and refine the skeleton
          eroded = cv2.erode(thresh, element)
          skel = cv2.bitwise_or(skel,temp)
          thresh = eroded.copy()
          # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
          if cv2.countNonZero(thresh)==0:
              break
      skel = 255 - skel
      kkk = np.where(skel == 0)
      skeleton_dots = list(zip(kkk[1], kkk[0]))

  # print("SKEL", skeleton_dots)
  for th_point in range(255, 0, -1):
    kk = np.where(img.astype(np.uint8) == th_point)
    brightest_dots = list(zip(kk[1], kk[0]))
    bdots = []
    for bdot in brightest_dots:
      result = cv2.pointPolygonTest(contour, (int(bdot[0]), int(bdot[1])), False) 
      if result > 0:
          bdots.append(bdot)
    if len(bdots) > 0:
      break
  #print("BDOTS", bdots)
  second_dot = None
  for bdot in bdots:
    min_dist = np.iinfo('int').max
    dists = []
    for sdot in skeleton_dots:
        a = np.array([bdot[0], bdot[1]])
        b = np.array([sdot[0], sdot[1]]) # img.shape[0] - 
        dst = cv2.norm(a - b, cv2.NORM_L2)
        dists.append(dst)
    sorted_idx = np.argsort(dists)
    #print(dists)
    #print(sorted_idx)
    #print(dists[sorted_idx[0]], dists[sorted_idx[1]])
    second_dot1 = skeleton_dots[sorted_idx[0]]
    second_dot2 = skeleton_dots[sorted_idx[1]]
    #sdot = second_dot1
    second_dot = [np.around(np.mean([second_dot1[0], second_dot2[0]])), np.around(np.mean([second_dot1[1], second_dot2[1]]))]
    if bdot is not None and second_dot is not None:
      break
  #print("SDOT", sdot)
  slope = (second_dot[1] - bdot[1]) / (second_dot[0] - bdot[0])
  bias = second_dot[1] - slope * second_dot[0]
  #intersection with contour
  for dst in range(1, 10):
    xs = []
    ys = []
    for pr in contour:
      x = pr[0][0]
      y = pr[0][1]
      if np.abs(y - (slope * x + bias)) < dst:
        ln = np.linalg.norm(np.array(bdot) - np.array([x,y]))
        pln = np.linspace(bdot, [x,y], int(np.round(ln)))
        pln = np.round(pln).astype(int)
        is_black = 0
        for pl in pln:
          if mask[pl[1]][pl[0]] == 0:
            is_black = is_black + 1
        if is_black == 0:
          xs.append(x)
          ys.append(y)
    if len(xs) >= 4:
      break
  if len(xs) >= 4:
    p1 = xs[np.argmin(xs)], ys[np.argmin(xs)]
    p2 = xs[np.argmax(xs)], ys[np.argmax(xs)]
    ln = np.linalg.norm(np.array(p1) - np.array(p2))
    points_on_line = np.linspace(p1, p2, int(np.round(ln)))
    points_on_line = np.round(points_on_line).astype(int)
  else:
    return skeleton_dots, [0, 0], [0, 0], []
  return skeleton_dots, bdot, second_dot, points_on_line

def get_line_old(img):
  for th_point in range(255, 0, -1):
    contour = None
    ret,thresh = cv2.threshold(img.astype(np.uint8),th_point,255,cv2.THRESH_BINARY)
    thresh_im = thresh.copy()
    contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
      max_area = 2048
      for c in contours:
        ar = cv2.contourArea(c)
        if ar > max_area:
          max_area = ar
          contour = c
    if contour is not None:
      #print("CONT", contour)
      # Step 1: Create an empty skeleton
      #ret,thresh = cv2.threshold(img.astype(np.uint8), 127, 255, 0)
      skel = np.zeros(thresh.shape, np.uint8)

      # Get a Cross Shaped Kernel
      element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

      # Repeat steps 2-4
      while True:
          #Step 2: Open the image
          open_morth = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, element)
          #Step 3: Substract open from the original image
          temp = cv2.subtract(thresh, open_morth)
          #Step 4: Erode the original image and refine the skeleton
          eroded = cv2.erode(thresh, element)
          skel = cv2.bitwise_or(skel,temp)
          thresh = eroded.copy()
          # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
          if cv2.countNonZero(thresh)==0:
              break
      skel = 255 - skel
      kkk = np.where(skel == 0)
      skeleton_dots = list(zip(kkk[1], kkk[0]))
      break
  #print("SKEL", skeleton_dots)
  for th_point in range(255, 0, -1):
    kk = np.where(img.astype(np.uint8) == th_point)
    brightest_dots = list(zip(kk[1], kk[0]))
    bdots = []
    for bdot in brightest_dots:
      result = cv2.pointPolygonTest(contour, (int(bdot[0]), int(bdot[1])), False) 
      if result > 0:
          bdots.append(bdot)
    if len(bdots) > 0:
      break
  #print("BDOTS", bdots)
  second_dot = None
  for bdot in bdots:
    min_dist = np.iinfo('int').max
    dists = []
    for sdot in skeleton_dots:
        a = np.array([bdot[0], bdot[1]])
        b = np.array([sdot[0], sdot[1]]) # img.shape[0] - 
        dst = cv2.norm(a - b, cv2.NORM_L2)
        dists.append(dst)
    sorted_idx = np.argsort(dists)
    #print(dists)
    #print(sorted_idx)
    #print(dists[sorted_idx[0]], dists[sorted_idx[1]])
    second_dot1 = skeleton_dots[sorted_idx[0]]
    second_dot2 = skeleton_dots[sorted_idx[1]]
    #sdot = second_dot1
    second_dot = [np.around(np.mean([second_dot1[0], second_dot2[0]])), np.around(np.mean([second_dot1[1], second_dot2[1]]))]
    if bdot is not None and second_dot is not None:
      break
  #print("SDOT", sdot)
  slope = (second_dot[1] - bdot[1]) / (second_dot[0] - bdot[0])
  bias = second_dot[1] - slope * second_dot[0]
  #intersection with contour
  for dst in range(1, 10):
    xs = []
    ys = []
    for pr in contour:
      x = pr[0][0]
      y = pr[0][1]
      if np.abs(y - (slope * x + bias)) < dst:
        ln = np.linalg.norm(np.array(bdot) - np.array([x,y]))
        pln = np.linspace(bdot, [x,y], int(np.round(ln)))
        pln = np.round(pln).astype(int)
        is_black = 0
        for pl in pln:
          if thresh_im[pl[1]][pl[0]] == 0:
            is_black = is_black + 1
        if is_black == 0:
          xs.append(x)
          ys.append(y)
    if len(xs) >= 4:
      break
  if len(xs) >= 4:
    p1 = xs[np.argmin(xs)], ys[np.argmin(xs)]
    p2 = xs[np.argmax(xs)], ys[np.argmax(xs)]
    ln = np.linalg.norm(np.array(p1) - np.array(p2))
    points_on_line = np.linspace(p1, p2, int(np.round(ln)))
    points_on_line = np.round(points_on_line).astype(int)
  else:
    return skeleton_dots, [0, 0], [0, 0], []
  return skeleton_dots, bdot, second_dot, points_on_line

def motion_detector(file_path):
  vidcap = cv2.VideoCapture(file_path)
  success,image_rgb = vidcap.read()
  if not success:
    return None, None
  image = image_rgb[:, :, 0]
  frame_count = 0
  previous_frame = None
  total_diff = None
  total_sum = None

  cy = image.shape[0]/2
  cx = image.shape[1]/2

  max_length = np.sqrt(cy * cy + cx * cx)

  c = np.array([cy, cx])
  radial_tensor = np.zeros_like(image).astype(np.float)
  for y in range(radial_tensor.shape[0]):
      for x in range(radial_tensor.shape[1]):
          b = np.array([y, x])
          dist = np.sqrt((cy - y) * (cy - y) + (cx - x) * (cx - x))
          radial_tensor[y][x] = np.around(1 - 1 * (dist/max_length), 2)
          
  while success:     
    
    if True: #((frame_count % 2) == 0):

        # 2. Prepare image; grayscale and blur
      prepared_frame = image
      #prepared_frame = cv2.GaussianBlur(src=image, ksize=(5,5), sigmaX=0)
      
      # 3. Set previous frame and continue if there is None
      if previous_frame is None:
          # First frame; there is no previous one yet
          previous_frame = prepared_frame
          total_diff = np.zeros_like(prepared_frame)
          continue
      
      # calculate difference and update previous frame
      diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
      previous_frame = prepared_frame

      # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
      #kernel = np.ones((5, 5))
      #diff_frame = cv2.dilate(diff_frame, kernel, 1)
      total_diff = total_diff + np.multiply(diff_frame / 255, radial_tensor)

      if total_sum is None:
        total_sum = np.multiply(prepared_frame / 255, radial_tensor)
      else:
        total_sum = total_sum + np.multiply(prepared_frame / 255, radial_tensor)

    success,image = vidcap.read()
    if success:
      image = image[:, :, 0]
    frame_count += 1
  
  vidcap.release()
  total_sum = 1 - total_sum / frame_count
  #total_diff = total_diff / np.max(total_diff)
  fres = total_diff * total_sum
  fres = fres / np.max(fres)
  fres = 255 * fres
  return image_rgb, fres

def get_direction(file_path):
  cap = cv2.VideoCapture(file_path)
  ret, frame1 = cap.read()
  prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
  hsv = np.zeros_like(frame1)
  hsv[..., 1] = 255
  angles = []
  mags = []
  while(1):
      ret, frame2 = cap.read()
      if not ret:
          break
      next = frame2[:, :, 0]
      flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
      mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
      hsv[..., 0] = np.rad2deg(ang)
      hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
      angles.append(hsv[..., 0])
      mags.append(hsv[..., 2])
      prvs = next
  cap.release()
  mean_angles = np.mean(angles, axis = 0)
  mean_mags = np.mean(mags, axis = 0)
  res = np.zeros_like(frame1)
  res[:, :, 0] = mean_angles
  res[:, :, 1] = 255
  res[:, :, 2] = mean_mags
  bgr = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
  return bgr, mean_angles

def get_rnn_input(vfile_path, sorted_end_frames, points_on_line):
  video_points = []
  y_points = []
  frame_count = 0
  vidcap = cv2.VideoCapture(vfile_path)
  success,image_rgb = vidcap.read()
  if success:
    img = image_rgb[:,:,0]/255

  while success:
    frame_points = []
    for px, py in points_on_line:
      frame_points.append(img[py, px])
    success, image_rgb = vidcap.read()
    if success:
      img = image_rgb[:,:,0]/255
    video_points.append(frame_points)
    if frame_count in sorted_end_frames:
      y_points.append(1)
    else:
      y_points.append(0)
    frame_count += 1
  vidcap.release()
  return video_points, y_points

def read_json_file(json_file):
    with open(json_file) as f:
        data = json.load(f)
    date = data['cbc_performed_at']
    cbc_results = data['cbc_results']
    hgb = cbc_results['hgb']
    wbc = cbc_results['wbc']
    hgb = float(hgb)
    wbc = float(wbc)
    date = date.split('-')
    date2 = date[1] + "/" + date[2] + "/" + date[0]
    return date2, hgb, wbc

def get_xy_data(inp):
    anno_file, video_file, ground_truth_file = inp
    try:
        f = open(anno_file)
        anno = json.load(f)
        f.close()
        annotation_keys = list(anno["annotations"]["annotations"].keys())
        if len(annotation_keys) == 0:
            return anno_file
    except:
        print("ERROR", anno_file)
        return anno_file

    try:
      first_frame, total_diff = motion_detector(video_file)
      capil_mask, capil_contours = get_capil_contour(predictor, total_diff)
      skeleton_dots, bdot, sdot, points_on_line_old = get_line_old(total_diff)
      skeleton_dots, bdot, second_dot, points_on_line = get_line(total_diff, capil_contours, capil_mask)
      bgr, mean_angles = get_direction(video_file)
      #direction_at_line = get_direction_at_a_point(mean_angles, points_on_line)

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
      video_points, y_points = get_rnn_input(video_file, sorted_end_frames, points_on_line)
      date2, hgb, wbc = read_json_file(ground_truth_file)
      n_parts = video_file.split("/")
      new_name = n_parts[-3] + "_" + n_parts[-2] + "_xy.pkl"
      f = open("/home/user_gpu_60/elenas_dir/xy_data/" + new_name, 'wb')
      pickle.dump((video_file, video_points, y_points, sorted_end_frames, points_on_line, points_on_line_old, total_diff, bgr, mean_angles, (date2, hgb, wbc), first_frame, capil_mask, capil_contours), f)
      f.close()
    except:
        print("ERROR", video_file)
        return anno_file
    return anno_file

files = glob.glob('/mnt/litebc/patient-data' + '/**/**/annotation.json')
files2 = []
for i in range(len(files)):
    k = files[i].replace('annotation.json', 'stabilzedAvi.avi')  
    k1 = "/".join(files[i].split("/")[:-2]) + "/ground_truth.json"
    if os.path.exists(k) and os.path.exists(k1):
        files2.append((files[i], k, k1))

import torch.multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('forkserver', force=True)
    num_processes = 4
    predictor.model.share_memory()

    with mp.Pool(processes=num_processes) as pool:
      result = pool.map(get_xy_data, files2, chunksize=None)
      print("Number Of Processed Files:", len(result))

