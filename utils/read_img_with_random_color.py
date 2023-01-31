import numpy as np
import skimage.io as io
from skimage.color import rgb2gray
import cv2
import random
import os
import time
import matplotlib.pyplot as plt
def random_rgb():
    r = np.random.rand()*255
    g = np.random.rand()*255
    b = np.random.rand()*255
    return np.array([r, g, b]).astype(np.uint8)


def showConnectedComponents(binary_img, path='lastview2.png'):
    if len(binary_img.shape)>2:
        binary_img=binary_img[:,:,0]
    w, h = binary_img.shape
    color = []
    color.append((0, 0, 0))
    img_color = np.zeros((w, h, 3), dtype=np.uint8)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img)
    for num in range(1, retval):
        color_b = random.randint(0, 255)
        color_g = random.randint(0, 255)
        color_r = random.randint(0, 255)
        color.append((color_b, color_g, color_r))
    for x in range(w):
        for y in range(h):
            lable = labels[x, y]
            img_color[x, y, :] = color[int(lable)]
    io.imsave(path, img_color)
    return img_color

#impath=r'D:\BaiduNetdiskDownload\BrestCancer\softmaxgradient-lrp-master\MASK250\TCGA-18-5592-01Z-00-DX1_15.png'
p1=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\submit_20201021T074705\TCGA-18-5592-01Z-00-DX1_crop_15.png'
p1_1=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\submit_20201021T074705\TCGA-21-5784-01Z-00-DX1_crop_0.png'
p2=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\submit_20201021T074705\TCGA-18-5592-01Z-00-DX1_crop_0.png'
n1='TCGA-18-5592-01Z-00-DX1_crop_15.png'
n1_1='TCGA-21-5784-01Z-00-DX1_crop_0.png'
q1=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\submit_20201023T175715NNN/'
Maskrcnn=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results/nucleus/submit_20210112T104856'
MaskrcnnSAV=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results/nucleus/MaskRCNN_SAV'
best_Loop4=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\submit_20210112T103916'
best_Loop4_sav=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\best_loop4_sav'
best_Loop1=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\submit_20201028T143029'
best_Loop1_sav=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\best_loop1_sav'
best_5=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\submit_20210111T123411'
best_5_sav=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\best_5_sav'
LRP_gt=r'D:\BaiduNetdiskDownload\BrestCancer\softmaxgradient-lrp-master\vgg\Work\gt'
LRP_gt_sav=r'D:\BaiduNetdiskDownload\BrestCancer\softmaxgradient-lrp-master\vgg\Work\gt_sav'
res50=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\submit_20210109T154020'
res50_sav=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\res50_sav'
# for fname in os.listdir(r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\samples\nucleus\simple-IAM\datasets\TestDemo\JPEGImages'):
H1=r'C:\Users\Administrator\Desktop\Loop1gt.png'
H2=r'C:\Users\Administrator\Desktop\Loop2gt.png'
H2=r'D:\BaiduNetdiskDownload\BrestCancer\MoNuSAC-master\GT224\TCGA-21-5786-01Z-00-DX1_mask_0.png'
#     print(fname)
im=io.imread(H1)
# bc_w_result=showConnectedComponents(io.imread(best_Loop1+n1),'15_1.png')
# bc_w_result=showConnectedComponents(io.imread(best_Loop1+n1_1),'15_2.png')

MAIN=res50
MAINSAV=res50_sav

os.makedirs(MAINSAV,exist_ok=True)
for fname in os.listdir(MAIN):
    savpath=MAINSAV
    showConnectedComponents(io.imread(os.path.join(MAIN,fname)), os.path.join(savpath,fname))