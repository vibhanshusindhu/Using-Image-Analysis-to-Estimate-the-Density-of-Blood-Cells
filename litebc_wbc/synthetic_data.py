height = 36
width = 350

import pickle
from random import randint
import cv2
import numpy as np
x = [(randint(0, 350), randint(0, 350), randint(0, 350)) for p in range(0, 120)]
for idx, (a, b, c) in enumerate(x):
    img = np.zeros(shape=[36, 350], dtype=np.uint8)
    y = [0] * 350

    if a + 19 < 350:
        y[a + 19] = 1
    if b + 19 < 350:
        y[b + 19] = 1
    if c + 19 < 350:
        y[c + 19] = 1
    cv2.circle(img, (a, 18), 18, (255, 255, 255), -1)
    cv2.circle(img, (b, 18), 18, (255, 255, 255), -1)
    cv2.circle(img, (c, 18), 18, (255, 255, 255), -1)
    cv2.imwrite("synth_data_ims_test/" + str(idx) + ".png", img)
    with open('synth_data_pkl_test/' + str(idx) + ".pkl", 'wb') as handle:
        pickle.dump((img, y), handle)

x = [randint(0, 350) for p in range(0, 350)]
for idx, a in enumerate(x):
    img = np.zeros(shape=[36, 350], dtype=np.uint8)
    y = [-1] * 350

    if a + 19 < 350:
        y[:a + 19] = [0] * (a+19)
        y[a + 19] = 1
    else:
        y = [0] * 350

    cv2.circle(img, (a, 18), 18, (255, 255, 255), -1)
    cv2.imwrite("synth_data_ims/" + str(idx) + ".png", img)
    with open('synth_data_pkl/' + str(idx) + ".pkl", 'wb') as handle:
        pickle.dump((img, y, np.min([350, a + 19 + 1])), handle)
