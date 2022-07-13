import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import warnings
from PIL import Image

warnings.filterwarnings("ignore", category=DeprecationWarning)
path = os.listdir('C:/Users/admin\PycharmProjects\AI\densityDetection\Data_modified\Data_gt/train_gt')
p2 = 'C:/Users/admin\PycharmProjects\AI\densityDetection\Data_modified\Data_gt/train_gt/'

def scale_array(x, new_size):
    min_el = np.min(x)
    max_el = np.max(x)
    y = scipy.misc.imresize(x, new_size, mode='L', interp='lanczos')
    y = y / 255 * (max_el - min_el) + min_el
    return y


for p in path:
    im = np.load('C:/Users/admin\PycharmProjects\AI\densityDetection\Data_modified\Data_gt/train_gt/train_data_gt_B_400.npy')
    cv2.imshow('im',im)
    cv2.waitKey(0)
    print(im)
    # im = np.load(p2 + p)
    # new_size = (128,128)
    # im2 = scale_array(im, new_size)
    # p3 = p2 + p
    # np.save(p3, im2)
    #
    # if "_h" in p:
    # width = (512,512)
    # im = cv2.imread(p2 + p)
    # im = cv2.resize(im, width, interpolation=cv2.INTER_LANCZOS4)
    # cv2.imwrite(p2 + p, im)



