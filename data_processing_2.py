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
path = os.listdir('C:/Users/admin\PycharmProjects\AI\densityDetection\Data_modified_cropped\Data_im/train_im')
p2 = 'C:/Users/admin\PycharmProjects\AI\densityDetection\Data_modified_cropped\Data_im/train_im/'
p3 = 'C:/Users/admin\PycharmProjects\AI\densityDetection\Data_modified_cropped\Data_im/1/'

p4 = 'C:/Users/admin\PycharmProjects\AI\densityDetection\Data_modified_cropped\Data_gt/train_gt/'
p5 = 'C:/Users/admin\PycharmProjects\AI\densityDetection\Data_modified_cropped\Data_gt/1/'
x = 1
total = 0
for p in path:

    c = 1
    print(x)
    x = x + 1
    im = Image.open(p2 + p)
    imshape = np.shape(im)
    front, back = p.split('.jpg')
    num = front.split('_')
    gt = np.load(p4 + 'train_data_gt_' + num[2] + '_' + str(num[1]) + '.npy')
    s = 256
    steprows, stepcols = 0, 0
    if imshape[0]%s != 0:
        steprows = (s - imshape[0]%s)//(imshape[0]//s)
    if imshape[1]%s != 0:
        stepcols = (s - imshape[1]%s)//(imshape[1]//s)

    if steprows > 224 or steprows == 0:
        ni = imshape[0]//s
        steprows = 0
    else:
        ni = imshape[0]//s + 1
    if stepcols > 224 or stepcols == 0:
        nj = imshape[1]//s
        stepcols = 0
    else:
        nj = imshape[1]//s + 1

    for i in range(ni):
        for j in range(nj):
            a = j*(s-stepcols)
            b = i*(s-steprows)
            new_name_im = front + '_' + str(c) + '.jpg'
            new_name_gt = 'train_data_gt_' + num[2] + '_' + str(num[1]) + '_' + str(c) + '.npy'
            im2 = im.crop((a, b, a+s, b+s))
            im2.save(p3 + new_name_im)
            im_gt = gt[b//4:(b+s)//4, a//4:(a+s)//4]
            im_gt = np.pad(im_gt, ((0,64-np.shape(im_gt)[0]),(0,64-np.shape(im_gt)[1])),'constant',constant_values=0)
            np.save(p4 + new_name_gt, im_gt)

            f = open('C:/Users/admin\PycharmProjects\AI\densityDetection\Data_modified_cropped\dir_name.txt', 'a')
            f.write(new_name_im + ' ' + new_name_gt + '\n')
            f.close()
            c = c + 1

    total = total + c - 1
    print(total)

    # c = 1
    # print(x)
    # x = x + 1
    # im = np.load(p2 + p)
    # imshape = im.shape
    # front, back = p.split('.npy')
    # s = 64
    # steprows, stepcols= 0, 0
    # if imshape[0]%s != 0:
    #     steprows = (s - imshape[0]%s)//(imshape[0]//s)
    # if imshape[1]%s != 0:
    #     stepcols = (s - imshape[1]%s)//(imshape[1]//s)

    # if steprows > s-8 or steprows == 0:
    #     ni = imshape[0]//s
    # else:
    #     ni = imshape[0]//s + 1
    # if stepcols > s-8 or stepcols == 0:
    #     nj = imshape[1]//s
    # else:
    #     nj = imshape[1]//s + 1
    # print(ni, nj)
    # print(steprows, stepcols)
    # for i in range(ni):
    #     for j in range(nj):
    #         a = j*(s-stepcols)
    #         b = i*(s-steprows)
    #         new_name_gt = front + '_' + str(c) + '.npy'
    #         im2 = im[b:b+s,a:a+s]
    #         im2 = np.pad(im2, ((0,s-np.shape(im2)[0]),(0,s-np.shape(im2)[1])),'constant',constant_values=0)
    #         np.save(p3 + new_name_gt, im2)
    #         c = c + 1

    # total = total + c - 1
