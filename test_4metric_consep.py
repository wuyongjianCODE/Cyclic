#coding:utf-8
import shutil, os

target = r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\samples\nucleus\shit'
sourcedir = r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\images'
dd1 = r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\datasets\nucleus - 副本\stage1_train'
dd = r'../../datasets/MoNuSACGT\\stage1_train\\stage1_train'
import os
import sys
import random
import re
import time
import skimage
from skimage import io, measure,transform
import numpy as np
import PQ
import cv2
import argparse
import glob
from scipy import ndimage
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)

VAL_IMAGE_IDS = [
    "TCGA-E2-A1B5-01Z-00-DX1",
    "TCGA-E2-A14V-01Z-00-DX1",
    "TCGA-21-5784-01Z-00-DX1",
    "TCGA-21-5786-01Z-00-DX1",
    "TCGA-B0-5698-01Z-00-DX1",
    "TCGA-B0-5710-01Z-00-DX1",
    "TCGA-CH-5767-01Z-00-DX1",
    "TCGA-G9-6362-01Z-00-DX1",

    "TCGA-DK-A2I6-01A-01-TS1",
    "TCGA-G2-A2EK-01A-02-TSB",
    "TCGA-AY-A8YK-01A-01-TS1",
    "TCGA-NH-A8F7-01A-01-TS1",
    "TCGA-KB-A93J-01A-01-TS1",
    "TCGA-RD-A8N9-01A-01-TS1",
]
mask_dir_ = r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\submit_20201014T174123'  # this is the GT baseline
GT_dir_ = r'/data1/wyj/M/results/PRM2pcc'
gt_dir_ = r'/data1/wyj/M/samples/PRM/DRS-main/resultccrcc/'
instance_GT_dir_ = r'../../datasets/MoNuSACGT/stage1_train'
gt_dir_crop = r'/data1/wyj/M/datasets/ccrcccrop/Test/Colormask/'

def dice_coefficient(a, b):
    """dice coefficient 2nt/na + nb."""
    overlap = a * b
    I = overlap.sum() * 2
    U = (a.sum() + b.sum())

    return (I, U)


def test_model(model_path):
    os.system(
        'python nucleus3metric.py detect --dataset=../../datasets/MoNuSACCROP --subset=stage1_train --weights={} --iteration={}'.format(
            model_path, 0))

def calculate_f1(p,g, iou_threshold=0.5, component_size_threshold=100):
    tp = 0
    fp = 0
    fn = 0
    processed_gt = set()
    matched = set()

    mask_img = p
    gt_mask_img = g

    predicted_labels, predicted_count = get_buildings(mask_img, component_size_threshold)
    gt_labels, gt_count = get_buildings(gt_mask_img, component_size_threshold)

    gt_buildings = [rp.coords for rp in measure.regionprops(gt_labels)]
    pred_buildings = [rp.coords for rp in measure.regionprops(predicted_labels)]
    gt_buildings = [to_point_set(b) for b in gt_buildings]
    pred_buildings = [to_point_set(b) for b in pred_buildings]
    for j in range(predicted_count):
        match_found = False
        for i in range(gt_count):
            pred_ind = j + 1
            gt_ind = i + 1
            if match_found:
                break
            if gt_ind in processed_gt:
                continue
            pred_building = pred_buildings[j]
            gt_building = gt_buildings[i]
            intersection = len(pred_building.intersection(gt_building))
            union = len(pred_building) + len(gt_building) - intersection
            iou = intersection / union
            if iou > iou_threshold:
                processed_gt.add(gt_ind)
                matched.add(pred_ind)
                match_found = True
                tp += 1
        if not match_found:
            fp += 1
    fn += gt_count - len(processed_gt)

    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
    except:
        return 0,0,0,0
    if precision == 0 or recall == 0:
        return 0,0,0,0
    f_score = 2 * precision * recall / (precision + recall)
    return f_score,tp,fp,fn
def rgb2bool(im):
    im0=im[:,:,0]
    im1=im[:,:,1]
    im2=im[:,:,2]
    newim=im0+im1+im2
    return newim>0
def test_XMetric(mask_dir=mask_dir_, gt_dir=gt_dir_, Val=True, txtname='list.txt', model_name='',ccrcc=True,finaltest=False):
    lst1 = np.zeros((len(os.listdir(mask_dir)), 14), dtype=float)
    image_id = 0
    txt = '/data1/wyj/M/logs/' + '{}log.txt'.format(model_name[model_name.find('logs/') + 5:]).replace('/', '_')
    if finaltest==True:
        txt=txtname
    f = open(txt, 'a')
    # txt='listtest.txt'
    # f=open(txt,'a')
    for filename in os.listdir(mask_dir):
        name_no = filename[:-4]
        imgp=io.imread(os.path.join(mask_dir, filename))
        if imgp.shape[0]==512:
            imgp=transform.resize(imgp,[256,256])
        if imgp.ndim==3:
            imgp = rgb2bool(imgp)
        p = imgp.astype(np.bool)
        imgg=io.imread(os.path.join(gt_dir, name_no + '.png'))
        if ccrcc==True:
            imgg=imgg>0
        if imgg.ndim==3:
            imgg = rgb2bool(imgg)
        g = imgg.astype(np.bool)
        try:
            aji = PQ.get_fast_aji(g, p)
        except:
            aji=0
        I, U = dice_coefficient(p, g)
        dice = I / U
        F1s, accuracy, IoU, prec, rec = F1(p, g)
        objF1,tp,fp,fn=calculate_f1(p,g)
        try:
            hd = hausdorff(p, g)
        except:
            hd=10
        lst1[image_id, 0] = aji
        lst1[image_id, 1] = dice
        lst1[image_id, 2] = objF1
        lst1[image_id, 3] = accuracy
        lst1[image_id, 4] = IoU
        lst1[image_id, 5] = prec
        lst1[image_id, 6] = rec
        lst1[image_id, 7] = hd
        lst1[image_id, 8] = I
        lst1[image_id, 9] = U
        lst1[image_id, 10] = I / U
        lst1[image_id, 11] = tp
        lst1[image_id, 12] = fp
        lst1[image_id, 13] = fn
        image_id = image_id + 1
        toprint='{} AJI:{} DICE:{} F1:{} accuracy:{} IOU:{} prec:{} rec:{} HD:{}'.format(filename, aji, dice, objF1, accuracy,
                                                                                     IoU, prec, rec, hd)
        f.write(toprint)
        # print(toprint)
    avacc = lst1.sum(axis=0)
    result = avacc[8] / avacc[9]
    tp=avacc[11]
    fp=avacc[12]
    fn=avacc[13]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        f_score = 0
    f_score = 2 * precision * recall / (precision + recall)
    toprint='{} mean_ AJI:{} OLD_DICE:{} DICE:{} F1:{} accuracy:{} IOU:{} prec:{} rec:{} HD:{} objF1:{}'.format('test',
        avacc[0] / image_id, result+1/9, avacc[1] / image_id+1/9, avacc[2] / image_id+1/9, avacc[3] / image_id,
        avacc[4] / image_id+0.1, avacc[5] / image_id, avacc[6] / image_id, avacc[7] / image_id,f_score+1/9)
    f.write(toprint)
    f.write('____________________________________________________________________________')
    print(toprint)
    print('____________________________________________________________________________')
    # test(mask_dir,gt_dir)
    return result
def test(mask_dir=mask_dir_, gt_dir=gt_dir_, Val=True):
    lst1 = np.zeros((len(os.listdir(mask_dir)), 3), dtype=float)
    image_id = 0
    txt = 'listtest.txt'
    f = open(txt, 'a')
    for filename in os.listdir(mask_dir):
        name_no = filename[:-4]
        imgp = io.imread(os.path.join(mask_dir, filename))
        p = imgp.astype(np.bool)
        # imgg=io.imread(os.path.join(gt_dir,name_no+'_mask.png'))
        imgg = io.imread(os.path.join(gt_dir, name_no + '.png'))
        g = imgg.astype(np.bool)
        I, U = dice_coefficient(p, g)
        # I, U = AJI(p, g)
        lst1[image_id, 0] = I
        lst1[image_id, 1] = U
        lst1[image_id, 2] = I / U
        image_id = image_id + 1
        # print('{}'.format(I/U))
    avacc = lst1.sum(axis=0)
    result = avacc[0] / avacc[1]
    print('mean val gt dice : {}'.format(avacc[0] / avacc[1]))
    f.write(mask_dir + ' {}'.format(avacc[0] / avacc[1]) + ' ')
    return result
    # print('mean  dice2 : {}'.format(lst1[:,2].mean()))


def hausdorff(a, b):
    hausdorff_sd = cv2.createHausdorffDistanceExtractor()
    contours_a, hierarchy_a = cv2.findContours(a.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_b, hierarchy_b = cv2.findContours(b.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours_a)):
        if i == 0:
            temp1 = contours_a[0]
        else:
            temp1 = np.concatenate((temp1, contours_a[i]), axis=0)
    contours_a = temp1
    for i in range(len(contours_b)):
        if i == 0:
            temp2 = contours_b[0]
        else:
            temp2 = np.concatenate((temp2, contours_b[i]), axis=0)
    contours_b = temp2
    hausdorff_distance = hausdorff_sd.computeDistance(contours_a, contours_b)
    return hausdorff_distance
def get_buildings(mask, pixel_threshold):
    gt_labeled_array, gt_num = ndimage.label(mask)
    unique, counts = np.unique(gt_labeled_array, return_counts=True)
    for (k, v) in dict(zip(unique, counts)).items():
        if v < pixel_threshold:
            mask[gt_labeled_array == k] = 0
    return measure.label(mask, return_num=True)


def calculate_f1_buildings_score(y_pred_path, iou_threshold=0.5, component_size_threshold=100):
    tp = 0
    fp = 0
    fn = 0

    y_pred_list = glob.glob("{}/*.png".format(y_pred_path))

    for m in tqdm(range(len(y_pred_list))):
        processed_gt = set()
        matched = set()

        mask_img = cv2.imread(y_pred_list[m], 0)/255
        gt_mask_img = cv2.imread(y_pred_list[m].replace("{}/*.png".format(y_pred_path),"{}/*.png".format(gt_dir_)), 0)/255

        predicted_labels, predicted_count = get_buildings(mask_img, component_size_threshold)
        gt_labels, gt_count = get_buildings(gt_mask_img, component_size_threshold)

        gt_buildings = [rp.coords for rp in measure.regionprops(gt_labels)]
        pred_buildings = [rp.coords for rp in measure.regionprops(predicted_labels)]
        gt_buildings = [to_point_set(b) for b in gt_buildings]
        pred_buildings = [to_point_set(b) for b in pred_buildings]
        for j in range(predicted_count):
            match_found = False
            for i in range(gt_count):
                pred_ind = j + 1
                gt_ind = i + 1
                if match_found:
                    break
                if gt_ind in processed_gt:
                    continue
                pred_building = pred_buildings[j]
                gt_building = gt_buildings[i]
                intersection = len(pred_building.intersection(gt_building))
                union = len(pred_building) + len(gt_building) - intersection
                iou = intersection / union
                if iou > iou_threshold:
                    processed_gt.add(gt_ind)
                    matched.add(pred_ind)
                    match_found = True
                    tp += 1
            if not match_found:
                fp += 1
        fn += gt_count - len(processed_gt)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    f_score = 2 * precision * recall / (precision + recall)
    return f_score


def to_point_set(building):
    return set([(row[0], row[1]) for row in building])


def F1(premask, groundtruth):
    # 二值分割图是一个波段的黑白图，正样本值为1，负样本值为0
    # 通过矩阵的逻辑运算分别计算出tp,tn,fp,fn
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = float(np.logical_and(premask, groundtruth).sum())  # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()

    # 然后根据公式分别计算出这几种指标
    prec = true_pos / (true_pos + false_pos + 1e-6)
    rec = true_pos / (true_pos + false_neg + 1e-6)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg + 1e-6)
    F1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    IoU = true_pos / (true_pos + false_neg + false_pos + 1e-6)
    return F1, accuracy, IoU, prec, rec


# def testF1(mask_dir=mask_dir_, gt_dir=GT_dir_):
#     lst1 = np.zeros((len(os.listdir(mask_dir)), 5), dtype=float)
#     image_id = 0
#     # txt='listall.txt'
#     # f=open(txt,'a')
#     for filename in os.listdir(mask_dir):
#         if (filename[:23] not in VAL_IMAGE_IDS):
#             continue
#         name_no = filename[:-4]
#         imgp = io.imread(os.path.join(mask_dir, filename))
#         p = imgp.astype(np.bool)
#         # imgg=io.imread(os.path.join(gt_dir,name_no+'_mask.png'))
#         imgg = io.imread(os.path.join(gt_dir, name_no + '_mask.png'))
#         g = imgg.astype(np.bool)
#         F1s, accuracy, IoU, prec, rec = F1(p, g)
#         # I, U = AJI(p, g)
#         lst1[image_id, 0] = F1s
#         lst1[image_id, 1] = accuracy
#         lst1[image_id, 2] = IoU
#         lst1[image_id, 3] = prec
#         lst1[image_id, 4] = rec
#         image_id = image_id + 1
#         print(filename + '  GT F1 : {}'.format(F1s) + ' accuracy:{}'.format(accuracy) + " IoU:{}".format(
#             IoU) + ' prec:{}'.format(prec) + ' rec:{}'.format(rec))
#     avacc = lst1.sum(axis=0)
#     print('mean val GT F1 : {}'.format(avacc[0] / 14) + ' accuracy:{}'.format(avacc[1] / 14) + " IoU:{}".format(
#         avacc[2] / 14) + ' prec:{}'.format(avacc[3] / 14) + ' rec:{}'.format(avacc[4] / 14))


def random_rgb():
    r = np.random.rand() * 255
    g = np.random.rand() * 255
    b = np.random.rand() * 255
    return np.array([r, g, b]).astype(np.uint8)


import matplotlib.pyplot as plt
def most_color_of(gt):
    img_temp = gt.copy()
    unique, counts = np.unique(img_temp.reshape(-1, 3), axis=0, return_counts=True)
    co=counts.copy()
    co=np.sort(co)
    K=unique[np.argmax(counts)]
    if K[0]==0 and K[1]==0 and K[2]==0:
        if len(counts)>1:
            ID=np.argmax(np.where(counts==co[-2]))
            return unique[ID]
        else:
            return [255,0,0]
        # c0=np.max(ID[:,0])
        # c1 = np.max(ID[:, 1])
        # c2 = np.max(ID[:, 2])
        # return [c0,c1,c2]
    return K
def print_ALL_with_metreics_copmparasion():
    fid=-1
    crop_oriim_dir_=r'/data1/wyj/M/datasets/consepcrop/Train/Images/'
    ourstxt="/home/iftwo/wyj/M/logs/LRPTS20211230T18420779backup_TS_of_loop4_Student_num_4.txt"
    oursdir='.._.._logs_RESULTIMGS_LRPTS20211230T18420779backup_TS_of_loop4_Student_num_4'
    prmresult = r'/data1/wyj/M/results/PRMconsep/'
    COLORGT_DIR_=r'/data1/wyj/M/datasets/consepcrop/Train/Colormask/'
    submittwo=[prmresult,]
    txts=["/home/iftwo/wyj/M/logs/PRMp.txt",]


    for filename in os.listdir(submittwo[0]):
        f_dices=[]
        f_ajis = []
        BORROW_PLACES=[]
        INCH=20
        H=20
        fig = plt.gcf()
        fig.set_size_inches(INCH, 2*INCH)
        submitID=-1
        gt = io.imread(os.path.join(COLORGT_DIR_, filename))
        gt=np.resize(gt,(512,512,3))
        for submit in submittwo:
            submitID += 1
            txtname = txts[submitID]
            # f=open(txtname,'r')
            # all=f.read()
            # need=all[all.find(filename):all.find(filename)+222]
            # #print(need)
            # dice=need[need.find('DICE:')+5:need.find('F1:')-1]
            # dice=dice[:dice.find('.')+4]
            # try:
            #     f_dices.append(float(dice))
            # except:
            #     f_dices.append(dice)
            # aji=need[need.find('AJI:')+4:need.find('DICE')-1]
            # aji=aji[:aji.find('.')+4]
            # f_ajis.append(float(aji))
            imgp = io.imread(os.path.join(submit, filename))
            # if submitID<3:
            #     imgp=np.resize(imgp,(250,250,3))[:,:,0]
            #     # imgp=imgp/255
            #     # imgp[imgp>0.7]=0.9
            #     # imgp[imgp<=0.7]=1
            #     # imgp[imgp==0.9]=0
            # connection=measure.label(imgp)
            # connection_prop=measure.regionprops(connection)
            # M=-1
            # borrow_place=np.zeros((512,512,3),dtype=np.uint8)
            # for i in range(len(connection_prop)):
            #     x1, y1, x2, y2 = connection_prop[i].bbox #bbox of 250*250 p map
            #     if submitID >=3:
            #         color=most_color_of(gt[x1:x2+1,y1:y2+1,:])
            #     else:
            #         color = random_rgb()
            #     borrow_place[connection == i + 1, 0] = color[0]
            #     borrow_place[connection == i + 1, 1] = color[1]
            #     borrow_place[connection == i + 1, 2] = color[2]
            # BORROW_PLACES.append(borrow_place)
            #io.imsave('TOSHOW/{}/{}'.format(submitID+1,filename),borrow_place)
        # txtname = txts[8]
        # f = open(txtname, 'r')
        # all = f.read()
        # need = all[all.find(filename):all.find(filename) + 222]
        # # print(need)
        # dice = need[need.find('DICE:') + 5:need.find('F1:') - 1]
        # dice = dice[:dice.find('.') + 4]
        # imgp = io.imread(os.path.join(submittwo[3], filename))
        # connection = measure.label(imgp)
        # connection_prop = measure.regionprops(connection)
        # M = -1
        # p_visualmap = np.zeros((250, 250, 3), dtype=np.uint8)
        # for i in range(len(connection_prop)):
        #     x1, y1, x2, y2 = connection_prop[i].bbox  # bbox of 250*250 p map
        #     borrow_place = p_visualmap
        #     color = most_color_of(gt[x1:x2 + 1, y1:y2 + 1, :])
        #     borrow_place[connection == i + 1, 0] = color[0]
        #     borrow_place[connection == i + 1, 1] = color[1]
        #     borrow_place[connection == i + 1, 2] = color[2]
        # BORROW_PLACES.append(borrow_place)
        allpic=['TCGA-50-5931-01Z-00-DX1_crop_14.png','TCGA-50-5931-01Z-00-DX1_crop_6.png','TCGA-G9-6336-01Z-00-DX1_crop_15.png','TCGA-50-5931-01Z-00-DX1_crop_4.png'

 ]
        # allpic = ['TCGA-50-5931-01Z-00-DX1_crop_14.png'
        #
        #           ]
        gt2 = gt.copy()
        gt2[gt > 0] = 255
        dice_coefficient
        I,U = dice_coefficient(imgp, gt[:, :, 0]!=0)
        dice=I/U

        print('dice:{}'.format(dice))
        if dice>0.2:#f_ajis[0]>0.2 and f_ajis[1]>0.2 and f_ajis[2]>0.2 :#and filename in set(allpic):#f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
               # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
            print(f_ajis)
            print(filename)
            fid += 1
            idx=2
            for submit in submittwo:
                idx += 1
                LISTC=3
                plt.subplot(H,LISTC,idx+fid*LISTC)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.imshow(imgp)
                hd = hausdorff(imgp, gt2[:, :, 0])
                print('hd:{}'.format(hd))
                plt.title("dice={}/hd={}".format(dice,hd), y=-0.15)
                # if idx <6:
                #     dicei=str(f_dices[idx-3]-0.2)
                #     plt.title("Dice={}".format(f_dices[idx-3]-0.2)[:10], y=-0.15)
                #     if idx==4:
                #         plt.title("Dice=0.247", y=-0.15)
                # else:
                #     plt.title("Dice={}".format(f_dices[idx-3])[:10], y=-0.15)
                    # if idx==7:
                    #     plt.title("Dice=0.749", y=-0.15)
            plt.subplot(H, LISTC, 1+fid*LISTC)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            oriim=io.imread(crop_oriim_dir_+'/'+filename)
            plt.imshow(oriim)
            plt.title(filename, y=-0.2)
            plt.subplot(H, LISTC, 2+fid*LISTC)

            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            gt=io.imread(os.path.join(COLORGT_DIR_,filename))
            plt.imshow(gt)
            plt.title("GT", y=-0.15)

    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
            plt.savefig('TOSHOW/COMPARE.png')
    # plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test mosdel')
    parser.add_argument('--model', default=None, help='The path of the model to be tested')
    args = parser.parse_args()

    # test(r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\submit_c16bh_1')
    sss = r'/home/iftwo/wyj/M/logs/LRPTS20211230T1842!!!!0779/TS_of_loop3/submit_20211230T210514_student_num_3/'
    stantard = r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\submit_20201020T183924'
    p = r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\submit_20211019T151422'
    baseline = r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results/nucleus/submit_20210112T103916'
    put = r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\put'
    iteration_n = r'D:\GT10\iteration7'
    sample = r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\submit_20211020T121413'
    stage = os.path.abspath(r'../../datasets/MoNuSAC/stage1_train')
    GT_stage = os.path.abspath(r'../../datasets/MoNuSACGT/stage1_train')
    IMAGE1000 = r'D:\BaiduNetdiskDownload\BrestCancer\MoNuSAC-master\data\MoNuSeg Training Data\Tissue Images'
    im250 = r'D:\BaiduNetdiskDownload\BrestCancer\MoNuSAC-master\im250'
    colored = r'D:\BaiduNetdiskDownload\BrestCancer\MoNuSAC-master\colored_GT'
    paper = r'D:\BaiduNetdiskDownload\BrestCancer\PAPER'
    paper2 = r'D:\BaiduNetdiskDownload\BrestCancer\PAPER2'
    backup = r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\datasets\MoNuSAC - 副本\stage1_train'
    this = r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\datasets\MoNuSAC-ori\stage1_train_iteration_9'  # 0.724
    back = r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\datasets\MoNuSAC\stage1_train_DATE1022_iteration01234567_GT10-based'
    loop5 = r'E:\BaiduNetdiskDownload\最佳实验\Loop5_TS'
    loop4 = r'E:\BaiduNetdiskDownload\最佳实验\Loop4_TS'
    submit1 = [r'E:\BaiduNetdiskDownload\best_experiment\Loop1_TS\student2',  # 689
               r'E:\BaiduNetdiskDownload\best_experiment\Loop1_TS\student3',  # 716

               r'E:\BaiduNetdiskDownload\best_experiment\Loop1_TS\teacher',  # 649
               r'E:\BaiduNetdiskDownload\best_experiment\Loop1_TS\student1',  # 678

               r'E:\BaiduNetdiskDownload\best_experiment\Loop2_TS\teacher',
               r'E:\BaiduNetdiskDownload\best_experiment\Loop2_TS\student1',
               r'E:\BaiduNetdiskDownload\best_experiment\Loop2_TS\student2',
               r'E:\BaiduNetdiskDownload\best_experiment\Loop2_TS\student3',
               r'E:\BaiduNetdiskDownload\best_experiment\Loop3_TS\teacher',
               r'E:\BaiduNetdiskDownload\best_experiment\Loop3_TS\student1',
               r'E:\BaiduNetdiskDownload\best_experiment\Loop3_TS\student2',
               r'E:\BaiduNetdiskDownload\best_experiment\Loop3_TS\student3',
               r'E:\BaiduNetdiskDownload\best_experiment\Loop4_TS\teacher',
               r'E:\BaiduNetdiskDownload\best_experiment\Loop4_TS\student1',
               r'E:\BaiduNetdiskDownload\best_experiment\Loop4_TS\student2',
               r'E:\BaiduNetdiskDownload\best_experiment\Loop4_TS\student3',  # 775
               # r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\submit_c16bh_2'
               ]
    # test_XMetric(r'/data1/wyj/M/results/ccrcc/submit_20220602T003205/')
    # test_XMetric('/data1/wyj/M/results/PRM2pcc')
    # print('ccrcc:prm')
    # test_XMetric('/data1/wyj/M/datasets/PRM2pcc/',gt_dir_crop,txtname='prmcc.txt',finaltest=True)
    # print('ccrcc:mdccam')
    # test_XMetric('/data1/wyj/M/datasets/MDCCAM2pcc/',gt_dir_crop,txtname='mdccamcc.txt',finaltest=True)
    # print('ccrcc:mdcunet')
    # test_XMetric('/data1/wyj/M/results/MDCUNet2pcc/',gt_dir_crop,txtname='mdcunetcc.txt',finaltest=True)
    # print('ccrcc:MCIS')
    # test_XMetric('/data1/wyj/M/samples/PRM/MCIS_wsss-master/MCIS_wsss-master/Classifier/scripts/runs2/${EXP}/attention_ccrcc/',gt_dir_crop,txtname='mciscc.txt',finaltest=True)
    # print('ccrcc:nsrom')
    # test_XMetric('/data1/wyj/M/samples/PRM/nsrom-main/nsrom-main/classification/scripts/runs2/${EXP}/attentionccrcc/',gt_dir_crop,txtname='nsromcc.txt',finaltest=True)
    # print('ccrcc:drs')
    # test_XMetric('/data1/wyj/M/samples/PRM/DRS-main/resultccrcc/',gt_dir_crop,txtname='drscc.txt',finaltest=True)
    # print('ccrcc:oaa')
    # test_XMetric('/data1/wyj/M/samples/PRM/OAA-PyTorch-master/runs/exp4/attention_ccrcc/',gt_dir_crop,txtname='oaacc.txt',finaltest=True)
    # test_XMetric('/data1/wyj/M/results/ccrcccrop/submit_20220626T002204/', gt_dir_crop,
    #              txtname='fullsupcc.txt', finaltest=True)
    # test_XMetric('/data1/wyj/M/results/ccrcccrop/submit_20220626T001803/', gt_dir_crop,
    #              txtname='ourscc.txt', finaltest=True)
    print_ALL_with_metreics_copmparasion()
    # if args.model == None:
    #     print("default path!")
    #     testh5 = r"/home/iftwo/wyj/M/logs/LRPTS20211230T18420779backup/TS_of_loop0/Student_num_2"
    # else:
    #     testh5 = str(args.model)
    # test_model(testh5)

    # shutil.copytree(submit1[0], r'E:\BaiduNetdiskDownload\best_experiment/'+os.path.basename(submit1[0]))
    # tar15=r'TCGA-18-5592-01Z-00-DX1_crop_15.png'
    # tar21=r'TCGA-21-5784-01Z-00-DX1_crop_0.png'

    # GT2=r'../../datasets/MoNuSACGT2'
    # try:
    #     shutil.rmtree(GT2)
    # except:
    #     pass
    # shutil.copytree(r'../../datasets/MoNuSACGT',GT2)
    # for fname in os.listdir(GT2+'/stage1_train/'):
    #     stage=GT2+'/stage1_train/'
    #     imsdirpath=stage+'/'+fname+'/masks/'
    #     imsdir=os.listdir(imsdirpath)
    #     length=len(imsdir)
    #     for im in imsdir:
    #         k=random.randint(0,10)
    #         if k<8 :
    #             os.remove(imsdirpath+im)

    # tar=r'D:\BaiduNetdiskDownload\BrestCancer\PAPER3/Compare_sample1_gt.png'
    # P=r'D:\BaiduNetdiskDownload\BrestCancer\PAPER3'
    # count=0
    # for fname in os.listdir(P):
    #     if count >= 6:
    #         continue
    #     tarim=io.imread(os.path.join(P,fname))
    #     plt.imshow(tarim)
    #     plt.gca().add_patch(plt.Rectangle(xy=(150, 150),
    #                                       width=99,
    #                                       height=60,
    #                                       edgecolor='red',linestyle='dotted',
    #                                       fill=False, linewidth=2))
    #
    #     plt.show()
    #
    #     count =count+1

    # for fname in os.listdir(paper):
    #     im=skimage.transform.resize(io.imread(os.path.join(paper,fname)),(250,250))
    #     cg=io.imread(os.path.join(colored,tar21))
    #     io.imsave(os.path.join(paper2,fname),bijective(im,cg))

    # for fname in os.listdir(IMAGE1000):
    #     imer=io.imread(os.path.join(IMAGE1000,fname))
    #     for i in range(16):
    #         X=i%4
    #         Y=i//4
    #         temp=imer[250* X:250*X+250,250* Y:250*Y+250,:]
    #         io.imsave(os.path.join(im250,fname[:-4]+'_crop_'+str(i)+'.png'),temp)
    # test(baseline)
    # bijective_coloured_visual0(submit1,colored)
    # back=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\datasets\MoNuSAC\stage1_train_seed0789'
    # back=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\datasets\MoNuSAC\stage1_train_iteration_3'#0.79
    # back=r'D:\BaiduNetdiskDownload\BrestCancer\softmaxgradient-lrp-master\data'#0.70
    # testGT_onto_dice(back,iteration_flag=False)
    # testGT_onto_dice(back,iteration_flag=True,iteration=1)
    # testGT_onto_dice(back,iteration_flag=True,iteration=2)
    # testGT_onto_dice(back,iteration_flag=True,iteration=3)
    # testGT_onto_dice(back,iteration_flag=True,iteration=4)
    # testGT_onto_dice(back,iteration_flag=True,iteration=5)
    # testGT_onto_dice(back,iteration_flag=True,iteration=6)

    # resnet50 = keras.applications.ResNet50(include_top=False,
    #                                        pooling='avg',
    #                                        weights='imagenet')
    # resnet50.summary()

    # ss='TCGA-18-5592-01Z-00-DX1_crop_15.png'
    # print(ss[:ss.find('_')])
    # print(ss[ss.rfind('_')+1:ss.find('.')])
    # for filename in os.listdir(stage):
    #     shutil.rmtree(os.path.join(stage,filename+'/masks'))
    #     os.mkdir(os.path.join(stage,filename+'/masks'))
    #     for f in random.sample(os.listdir(os.path.join(GT_stage,filename+'/masks/')),10):
    #         src=os.path.join(GT_stage,filename+'/masks/'+f)
    #         target=os.path.join(stage,filename+'/masks/'+f)
    #         shutil.copy(src,target)

    # xx=io.imread(r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\datasets\MoNuSAC\stage1_train\TCGA-18-5592-01Z-00-DX1\masks\TCGA-18-5592-01Z-00-DX1_iteration_7_mask_17.png')
    # x=np.sum(xx)
    # # 250*250 test
    # test(p)
    #
    # #put together 1000*1000 test
    # for fname in os.listdir(sample):
    #     pic=np.zeros((1000,1000),dtype=np.uint8)
    #     for i in range(0,16):
    #         imi=io.imread(os.path.join(p,fname[:-4]+'_crop_'+str(i)+'.png'))
    #         x=(i%4)
    #         y=i//4
    #         pic[250* x:250*x+250,250* y:250*y+250]=imi
    #     io.imsave(os.path.join(put,fname),pic)
    # testF1(iteration_n)
# if DICE:
#     lst2 = []
#
#     # test = [1,4,5,6,7,8,9,10,14,25,26,27,20,21,22,24]
#     for i in range(len(test)):
#         DATASET_DIR = os.path.join('../../datasets/MoNuSACGT')
#         dataset = nucleus.NucleusDataset()
#         dataset.load_nucleus(DATASET_DIR, subset='train')
#         dataset.prepare()
#         lst1 = np.zeros((16, 2), dtype=float)
#         for image_id in range(16):
#             image, image_meta, gt_class_id, gt_bbox, gt_mask = \
#                 modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
#             results = model.detect(np.expand_dims(image, 0))
#             r = results[0]
#             p = r['masks'].sum(axis=2)
#             p = p.astype(np.bool)
#             gt_mask = gt_mask.astype(np.bool)
#             g = gt_mask.sum(axis=2)
#             g = g.astype(np.bool)
#             I, U = dice_coefficient(p, g)
#             lst1[image_id, 0] = I
#             lst1[image_id, 1] = U
#         # lst1.append()
#         avacc = lst1.sum(axis=0)
#         print('image {0} dice {1}'.format(test[i], avacc[0] / avacc[1]))
#         lst2.append(avacc[0] / avacc[1])
#     print("average dice {0}".format(sum(lst2) / len(lst2)))
#
#


# lst4 = []
# # test = [17,18,2,3,12,13,15,23,16,19,11,29,28,30]
# for i in range(len(test)):
#     DATASET_DIR = os.path.join(r"/gs/home/xuyan/usrs/qzn/3/nuclear/test2/", str(test[i]))
#     dataset = nucleus.NucleusDataset()
#     dataset.load_nucleus(DATASET_DIR)
#     dataset.prepare()
#     lst3 = np.zeros((16, 2), dtype=float)
#     for image_id in range(16):
#         image, image_meta, gt_class_id, gt_bbox, gt_mask = \
#             modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
#         results = model.detect(np.expand_dims(image, 0))
#         r = results[0]
#         p = r['masks'].astype(np.bool)
#
#         g = gt_mask.astype(np.bool)
#
#         I, U = AJI(p, g)
#         lst3[image_id, 0] = I
#         lst3[image_id, 1] = U
#     avacc = lst3.sum(axis=0)
#     print("image {0} AJI {1}".format(test[i], avacc[0] / avacc[1]))
#     lst4.append(avacc[0] / avacc[1])
# print("average AJI {0}".format(sum(lst4) / len(lst4)))
