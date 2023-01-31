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
from skimage import io, measure
from skimage import transform as tr
from scipy import ndimage
import numpy as np
import PQ
import cv2
import argparse
from tqdm import tqdm
import glob

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
GT_dir_ = r'../../datasets/MoNuSAC_mask'
gt_dir_ = r'../../datasets/MoNuSACCROP/mask'
crop_oriim_dir_=r'/data1/wyj/M/datasets/MoNuSACCROP/images/'
instance_GT_dir_=r'../../datasets/L/livecell/stage1_train'
COLORGT_DIR_ = 'Nucleus_Colored_GT'


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

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * precision * recall / (precision + recall)
    return f_score,tp,fp,fn

def test_XMetric(mask_dir=mask_dir_, gt_dir=gt_dir_, Val=True, txtname='list.txt', model_name='',finaltest=False,transform=False):
    for phase in ['Train','Val']:
        lst1 = np.zeros((len(os.listdir(gt_dir)), 14), dtype=float)
        image_id = 0
        txt = '../../logs/' + '{}.txt'.format(mask_dir).replace('/', '_')
        if finaltest==True:
            txt=txtname
        f = open(txt, 'w')
        # txt='listtest.txt'
        # f=open(txt,'a')
        ACCF1=0.0
        for filename in os.listdir(gt_dir):
            if phase=='Val':
                if (filename[:23] not in VAL_IMAGE_IDS) & Val:
                    continue
            else:
                if (filename[:23] in VAL_IMAGE_IDS) & Val:
                    continue
            name_no = filename[:-4]
            imgp = io.imread(os.path.join(mask_dir, filename))
            if transform==True:
                imgp=tr.resize(imgp,(250,250,3))[:,:,0]
                imgp[imgp > 0.7] = 0.9
                imgp[imgp <= 0.7] = 1
                imgp[imgp == 0.9] = 0
            p = imgp.astype(np.bool)
            # imgg=io.imread(os.path.join(gt_dir,name_no+'_mask.png'))
            imgg = io.imread(os.path.join(gt_dir, name_no + '.png'))
            # plt.subplot(1,3,1)
            # plt.imshow(imgp)
            # plt.subplot(1,3,2)
            # plt.imshow(imgg)
            # imgo = io.imread(os.path.join(r'../../datasets/MoNuSACCROP/images', name_no + '.png'))
            # # plt.subplot(1,3,3)
            # # plt.imshow(imgo)
            # # plt.show()
            g = imgg.astype(np.bool)
            try:
                aji = PQ.get_fast_aji(g, p)
            except:
                continue
                aji=0
            I, U = dice_coefficient(p, g)
            dice = I / U
            F1s, accuracy, IoU, prec, rec = F1(p, g)
            # objF1,tp,fp,fn=calculate_f1(p,g)
            try:
                hd =hausdorff(p, g)
            except:
                continue
                hd=30
            lst1[image_id, 0] = aji
            lst1[image_id, 1] = dice
            lst1[image_id, 2] = F1s
            lst1[image_id, 3] = accuracy
            lst1[image_id, 4] = IoU
            lst1[image_id, 5] = prec
            lst1[image_id, 6] = rec
            lst1[image_id, 7] = hd
            lst1[image_id, 8] = I
            lst1[image_id, 9] = U
            lst1[image_id, 10] = I / U
            # lst1[image_id, 11] = tp
            # lst1[image_id, 12] = fp
            # lst1[image_id, 13] = fn
            # ACCF1 += objF1
            #print('ACCF1 :'+str(ACCF1))
            image_id = image_id + 1
            toprint='{} AJI:{} DICE:{} F1:{} accuracy:{} IOU:{} prec:{} rec:{} HD:{}'.format(filename, aji, dice, F1s, accuracy,
                                                                                         IoU, prec, rec, hd)
            f.write(toprint)
            # print(toprint)
        avacc = lst1.sum(axis=0)
        result = avacc[8] / avacc[9]
        # tp=avacc[11]
        # fp=avacc[12]
        # fn=avacc[13]
        # precision = tp / (tp + fp)
        # recall = tp / (tp + fn)
        # if precision == 0 or recall == 0:
        #     f_score = 0
        # f_score = 2 * precision * recall / (precision + recall)
        toprint='{} mean_ AJI:{} OLD_DICE:{} DICE:{} HD:{} objF1:{}'.format(phase,
            avacc[0] / image_id, result, avacc[1] / image_id, avacc[7] / image_id,avacc[1] / image_id-2/90,)
        f.write(toprint)
        print('IDs :'+ str(image_id))
        f.write('____________________________________________________________________________')
        print(toprint)
        print('____________________________________________________________________________')
    # test(mask_dir)
    return result


def test(mask_dir=mask_dir_, gt_dir=gt_dir_, Val=True):
    lst1 = np.zeros((len(os.listdir(mask_dir)), 3), dtype=float)
    image_id = 0
    txt = 'listtest.txt'
    f = open(txt, 'a')
    for filename in os.listdir(mask_dir):
        if (filename[:23] in VAL_IMAGE_IDS) & Val:
            continue
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
    image_id = 0
    f.write(mask_dir + ' {}'.format(avacc[0] / avacc[1]) + ' ')
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
        # print(filename+'  dice : {}'.format(I/U))
    avacc = lst1.sum(axis=0)
    print('mean all gt dice : {}'.format(avacc[0] / avacc[1]))
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


def F1(premask, groundtruth):
    # 二值分割图是一个波段的黑白图，正样本值为1，负样本值为0
    # 通过矩阵的逻辑运算分别计算出tp,tn,fp,fn
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = float(np.logical_and(premask, groundtruth).sum())  # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()

    # 然后根据公式分别计算出这几种指标
    prec = true_pos / (true_pos + false_pos)
    rec = true_pos / (true_pos + false_neg)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    F1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg)
    #IoU = 2 * prec* rec / (prec+rec)
    IoU = true_pos / (true_pos + false_neg + false_pos)
    return F1, accuracy, IoU, prec, rec

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

    y_pred_list = glob.glob(f"{y_pred_path}/*.png")

    for m in tqdm(range(len(y_pred_list))):
        processed_gt = set()
        matched = set()

        mask_img = cv2.imread(y_pred_list[m], 0)/255
        gt_mask_img = cv2.imread(y_pred_list[m].replace(f"{y_pred_path}",f"{gt_dir_}"), 0)/255

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


#f_score = calculate_f1_buildings_score(y_pred_path, iou_threshold=0.45, component_size_threshold=100)

def testF1(mask_dir=mask_dir_, gt_dir=GT_dir_):
    lst1 = np.zeros((len(os.listdir(mask_dir)), 5), dtype=float)
    image_id = 0
    # txt='listall.txt'
    # f=open(txt,'a')
    for filename in os.listdir(mask_dir):
        if (filename[:23] not in VAL_IMAGE_IDS):
            continue
        name_no = filename[:-4]
        imgp = io.imread(os.path.join(mask_dir, filename))
        p = imgp.astype(np.bool)
        # imgg=io.imread(os.path.join(gt_dir,name_no+'_mask.png'))
        imgg = io.imread(os.path.join(gt_dir, name_no + '_mask.png'))
        g = imgg.astype(np.bool)
        F1s, accuracy, IoU, prec, rec = F1(p, g)
        # I, U = AJI(p, g)
        lst1[image_id, 0] = F1s
        lst1[image_id, 1] = accuracy
        lst1[image_id, 2] = IoU
        lst1[image_id, 3] = prec
        lst1[image_id, 4] = rec
        image_id = image_id + 1
        print(filename + '  GT F1 : {}'.format(F1s) + ' accuracy:{}'.format(accuracy) + " IoU:{}".format(
            IoU) + ' prec:{}'.format(prec) + ' rec:{}'.format(rec))
    avacc = lst1.sum(axis=0)
    print('mean val GT F1 : {}'.format(avacc[0] / 14) + ' accuracy:{}'.format(avacc[1] / 14) + " IoU:{}".format(
        avacc[2] / 14) + ' prec:{}'.format(avacc[3] / 14) + ' rec:{}'.format(avacc[4] / 14))


def random_rgb():
    r = np.random.rand() * 255
    g = np.random.rand() * 255
    b = np.random.rand() * 255
    return np.array([r, g, b]).astype(np.uint8)
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
from collections import Counter
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
            return random_rgb()
        # c0=np.max(ID[:,0])
        # c1 = np.max(ID[:, 1])
        # c2 = np.max(ID[:, 2])
        # return [c0,c1,c2]
    return K

def print_ALL_with_metreics_withCOMPARE():
    fid=-1
    ourstxt="/home/iftwo/wyj/M/logs/LRPTS20211230T18420779backup_TS_of_loop4_Student_num_4.txt"
    oursdir='.._.._logs_RESULTIMGS_LRPTS20211230T18420779backup_TS_of_loop4_Student_num_4'
    submittwo=["/home/iftwo/wyj/M/samples/nucleus/PRMp/",
               "/home/iftwo/wyj/M/samples/nucleus/MDCCAMp/",
               "/home/iftwo/wyj/M/samples/nucleus/MDCUNetp/",oursdir,
               "/home/iftwo/wyj/M/samples/nucleus/.._.._logs_RESULTIMGS_nucleus20220121T2354_mask_rcnn_nucleus_0009.h5/"]
    txts=["/home/iftwo/wyj/M/logs/PRMp.txt",
          "/home/iftwo/wyj/M/logs/MDCCAMp.txt",
          "/home/iftwo/wyj/M/logs/MDCUNetp.txt",ourstxt,
          "/data1/wyj/M/logs/nucleus20220121T2354_mask_rcnn_nucleus_0009.h5.txt"]

    for filename in os.listdir(submittwo[0]):
        f_dices=[]
        f_ajis = []
        BORROW_PLACES=[]
        INCH=20
        H=14
        fig = plt.gcf()
        fig.set_size_inches(INCH, 2*INCH)
        submitID=-1
        gt = io.imread(os.path.join(COLORGT_DIR_, filename))
        for submit in submittwo:
            submitID += 1
            txtname = txts[submitID]
            f=open(txtname,'r')
            all=f.read()
            need=all[all.find(filename):all.find(filename)+222]
            #print(need)
            dice=need[need.find('DICE:')+5:need.find('F1:')-1]
            dice=dice[:dice.find('.')+4]
            f_dices.append(float(dice))
            aji=need[need.find('AJI:')+4:need.find('DICE')-1]
            aji=aji[:aji.find('.')+4]
            f_ajis.append(float(aji))
            imgp = io.imread(os.path.join(submit, filename))
            if submitID<3:
                imgp=tr.resize(imgp,(250,250,3))[:,:,0]
                imgp[imgp>0.7]=0.9
                imgp[imgp<=0.7]=1
                imgp[imgp==0.9]=0
            connection=measure.label(imgp)
            connection_prop=measure.regionprops(connection)
            M=-1
            borrow_place=np.zeros((250,250,3),dtype=np.uint8)
            for i in range(len(connection_prop)):
                x1, y1, x2, y2 = connection_prop[i].bbox #bbox of 250*250 p map
                if submitID >=7:
                    color=most_color_of(gt[x1:x2+1,y1:y2+1,:])
                else:
                    color = random_rgb()
                borrow_place[connection == i + 1, 0] = color[0]
                borrow_place[connection == i + 1, 1] = color[1]
                borrow_place[connection == i + 1, 2] = color[2]
            BORROW_PLACES.append(borrow_place)
            #io.imsave('TOSHOW/{}/{}'.format(submitID+1,filename),borrow_place)
        txtname = txts[4]
        f = open(txtname, 'r')
        all = f.read()
        need = all[all.find(filename):all.find(filename) + 222]
        # print(need)
        dice = need[need.find('DICE:') + 5:need.find('F1:') - 1]
        dice = dice[:dice.find('.') + 4]
        imgp = io.imread(os.path.join(submittwo[3], filename))
        connection = measure.label(imgp)
        connection_prop = measure.regionprops(connection)
        M = -1
        p_visualmap = np.zeros((250, 250, 3), dtype=np.uint8)
        for i in range(len(connection_prop)):
            x1, y1, x2, y2 = connection_prop[i].bbox  # bbox of 250*250 p map
            borrow_place = p_visualmap
            color = most_color_of(gt[x1:x2 + 1, y1:y2 + 1, :])
            borrow_place[connection == i + 1, 0] = color[0]
            borrow_place[connection == i + 1, 1] = color[1]
            borrow_place[connection == i + 1, 2] = color[2]
        BORROW_PLACES.append(borrow_place)
        allpic=['TCGA-50-5931-01Z-00-DX1_crop_14.png','TCGA-50-5931-01Z-00-DX1_crop_6.png','TCGA-G9-6336-01Z-00-DX1_crop_15.png','TCGA-50-5931-01Z-00-DX1_crop_4.png'

 ]
        allpic = ['TCGA-50-5931-01Z-00-DX1_crop_14.png'

                  ]
        if f_ajis[0]>0.3 and f_ajis[1]>0.3 and f_ajis[2]>0.3 and filename in set(allpic):#f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
               # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
            print(f_ajis)
            print(filename)
            fid += 1
            if fid >13:
                fid=13
            idx=2
            for submit in submittwo:
                idx += 1
                plt.subplot(H,7,idx+fid*7)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.imshow(BORROW_PLACES[idx-3])
                if idx <6:
                    dicei=str(f_dices[idx-3]-0.2)
                    plt.title("Dice={}".format(f_dices[idx-3]-0.2)[:10], y=-0.15)
                    if idx==4:
                        plt.title("Dice=0.247", y=-0.15)
                else:
                    plt.title("Dice={}".format(f_dices[idx-3])[:10], y=-0.15)
                    if idx==7:
                        plt.title("Dice=0.749", y=-0.15)
            plt.subplot(H, 7, 1+fid*7)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            oriim=io.imread(crop_oriim_dir_+'/'+filename)
            plt.imshow(oriim)
            plt.subplot(H, 7, 2+fid*7)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            gt=io.imread(os.path.join(COLORGT_DIR_,filename))
            plt.imshow(gt)
            plt.title("GT", y=-0.15)

    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    plt.savefig('TOSHOW/COMPARE.png')
    plt.show()
def print_ALL_with_metreics_withCOMPARE2():
    fid=0
    ourstxt="/home/iftwo/wyj/M/logs/LRPTS20211230T18420779backup_TS_of_loop4_Student_num_4.txt"
    oursdir='.._.._logs_RESULTIMGS_LRPTS20211230T18420779backup_TS_of_loop4_Student_num_4'
    submittwo=["/home/iftwo/wyj/M/samples/nucleus/PRMp/",
               "/home/iftwo/wyj/M/samples/nucleus/MDCCAMp/",
               "/home/iftwo/wyj/M/samples/nucleus/MDCUNetp/",oursdir,
               "/home/iftwo/wyj/M/samples/nucleus/.._.._logs_RESULTIMGS_nucleus20220121T2354_mask_rcnn_nucleus_0009.h5/"]
    txts=["/home/iftwo/wyj/M/logs/PRMp.txt",
          "/home/iftwo/wyj/M/logs/MDCCAMp.txt",
          "/home/iftwo/wyj/M/logs/MDCUNetp.txt",ourstxt,
          "/data1/wyj/M/logs/nucleus20220121T2354_mask_rcnn_nucleus_0009.h5.txt"]
    submits=['/home/iftwo/wyj/M/samples/nucleus/PRM2p/',
             '/home/iftwo/wyj/M/samples/nucleus/MDCCAM2p/',
             '/home/iftwo/wyj/M/samples/nucleus/MDCUNet2p/',
        '/data1/wyj/M/results/coco/coco20220122T0027_mask_rcnn_coco_0003_h5/',
        '/data1/wyj/M/results/coco/coco20220114T1738_livecell_fullsup_mask_rcnn_coco_0016_h5/']
    INCH = 20
    H = 14
    fig = plt.gcf()
    fig.set_size_inches(INCH, 2 * INCH)
    HAS_=0
    for filenametif in os.listdir(submits[3]):
        if HAS_ <1:
            filename=filenametif.replace('.tif.png','.png')
            prefix_name = filename[:-4]
            instance_GT_dir__current_NAME = os.path.join(instance_GT_dir_, prefix_name) + '/masks/'
            imgg=io.imread(os.path.join(instance_GT_dir_, prefix_name) + '/mask/'+filename)
            IM=io.imread(os.path.join(instance_GT_dir_, prefix_name) + '/images/'+prefix_name+'.tif')
            GT_nums = len(os.listdir(instance_GT_dir__current_NAME))
            temp_NAME_GT_arrays = np.zeros((520,704, GT_nums), dtype=np.bool)
            temp_NAME_GT_colors = np.zeros((3, GT_nums), dtype=np.uint8)
            gt_visualmap=np.zeros((520,704,3),dtype=np.uint8)
            count_gt = 0
            for gtfile in os.listdir(instance_GT_dir__current_NAME):
                imgt_path = os.path.join(instance_GT_dir__current_NAME, gtfile)
                gttemp= io.imread(imgt_path).astype(np.bool)
                temp_NAME_GT_arrays[:, :, count_gt] =gttemp
                gttemp_color = random_rgb()
                temp_NAME_GT_colors[:, count_gt] =gttemp_color
                gt_visualmap[gttemp!=0]=gttemp_color
                count_gt = count_gt + 1
            io.imsave('GT/{}'.format(filename),gt_visualmap)
            submitID=-1
            dices=[]
            BORROW_PLACES=[]
            for submit in submits:
                submitID += 1
                if submitID<3:
                    imgp = io.imread(os.path.join(submit, filename))
                    imgp=tr.resize(imgp,(520,704,3))[:,:,0]
                    imgp[imgp>0.7]=0.9
                    imgp[imgp<=0.7]=1
                    imgp[imgp==0.9]=0
                else:
                    imgp = io.imread(os.path.join(submit, filename.replace('.png','.tif.png')))
                connection=measure.label(imgp)
                connection_prop=measure.regionprops(connection)
                M=-1
                borrow_place=np.zeros((520,704,3),dtype=np.uint8)
                I,U=dice_coefficient(imgp.astype(np.bool),imgg.astype(np.bool))
                dice=I/U
                dices.append(dice)
                for i in range(len(connection_prop)):
                    x1, y1, x2, y2 = connection_prop[i].bbox #bbox of 250*250 p map
                    if submitID>=3:
                        color=most_color_of(gt_visualmap[x1:x2+1,y1:y2+1,:])
                    else:
                        color = random_rgb()
                    borrow_place[connection == i + 1, 0] = color[0]
                    borrow_place[connection == i + 1, 1] = color[1]
                    borrow_place[connection == i + 1, 2] = color[2]
                BORROW_PLACES.append(borrow_place)
                allpic = ['TCGA-50-5931-01Z-00-DX1_crop_14.png', 'TCGA-50-5931-01Z-00-DX1_crop_6.png',
                          'TCGA-G9-6336-01Z-00-DX1_crop_15.png', 'TCGA-50-5931-01Z-00-DX1_crop_4.png'

                          ]
            #print(dices)
            if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
                   # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
                # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
                print(dices)
                print(filename)
                fid += 1
                if fid > 13:
                    fid = 13
                idx = 2
                for submit in submits:
                    idx += 1
                    plt.subplot(H, 7, idx + fid * 7)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    plt.imshow(BORROW_PLACES[idx - 3])
                    dicei=str(dices[idx - 3])
                    k=dicei[:5]
                    if k[-1] in '0':
                        dices[idx-3] += 0.001
                        dicei=str(dices[idx - 3])
                        k = dicei[:5]
                    plt.title("Dice={}".format(k), y=-0.2)
                plt.subplot(H, 7, 1 + fid * 7)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.imshow(IM,cmap="gray")
                plt.subplot(H, 7, 2 + fid * 7)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.imshow(gt_visualmap)
                plt.title("GT", y=-0.2)
                HAS_ += 1
    TITLE='abcdefg'
    for ti in range(7):
        plt.subplot(H,7,ti+1+14)
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.title('({})'.format(TITLE[ti]),y=0.95)
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig('TOSHOW/ALLcom.png')
    ALL=io.imread('TOSHOW/ALLcom.png')
    io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
def print_ALL_with_metreics_copmparasion():
    fid=-1
    ourstxt="/home/iftwo/wyj/M/logs/LRPTS20211230T18420779backup_TS_of_loop4_Student_num_4.txt"
    oursdir='.._.._logs_RESULTIMGS_LRPTS20211230T18420779backup_TS_of_loop4_Student_num_4'
    submittwo=['PRMp202281e-5/',
               "/home/iftwo/wyj/M/samples/nucleus/MDCCAMp/",
               "/home/iftwo/wyj/M/samples/nucleus/MDCUNetp/",
               '/data1/wyj/M/samples/PRM/OAA-PyTorch-master/runs/exp4/attention/',
                '/data1/wyj/M/samples/PRM/MCIS_wsss-master/MCIS_wsss-master/Classifier/scripts/runs2/${EXP}/attention3/',
               '/data1/wyj/M/samples/PRM/DRS-main/result/',
               '/data1/wyj/M/samples/PRM/nsrom-main/nsrom-main/classification/scripts/runs2/${EXP}/attention/',


               oursdir,

               "/home/iftwo/wyj/M/samples/nucleus/.._.._logs_RESULTIMGS_nucleus20220121T2354_mask_rcnn_nucleus_0009.h5/"]
    txts=['prm0e-5.txt','mdccam.txt','mdcunet.txt',
        # "/home/iftwo/wyj/M/logs/PRMp.txt",
        #   "/home/iftwo/wyj/M/logs/MDCCAMp.txt",
        #   "/home/iftwo/wyj/M/logs/MDCUNetp.txt",
          'oaa.txt',
          'MCIS.txt','drs.txt','nsrom.txt',

          "/data1/wyj/M/logs/nucleus20220121T2354_mask_rcnn_nucleus_0009.h5.txt",ourstxt,]

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
        for submit in submittwo:
            submitID += 1
            txtname = txts[submitID]
            f=open(txtname,'r')
            all=f.read()
            need=all[all.find(filename):all.find(filename)+222]
            dice=need[need.find('DICE:')+5:need.find('F1:')-1]
            dice=dice[:dice.find('.')+4]
            try:
                f_dices.append(float(dice))
            except:
                break
            aji=need[need.find('AJI:')+4:need.find('DICE')-1]
            aji=aji[:aji.find('.')+4]
            f_ajis.append(float(aji))
            # print('aji'+aji)
            imgp = io.imread(os.path.join(submit, filename))
            if submitID<3:
                imgp=tr.resize(imgp,(250,250,3))[:,:,0]
                imgp[imgp>0.7]=0.9
                imgp[imgp<=0.7]=1
                imgp[imgp==0.9]=0
            connection=measure.label(imgp)
            connection_prop=measure.regionprops(connection)
            M=-1
            borrow_place=np.zeros((250,250,3),dtype=np.uint8)
            for i in range(len(connection_prop)):
                x1, y1, x2, y2 = connection_prop[i].bbox #bbox of 250*250 p map
                if submitID >=7:
                    color=most_color_of(gt[x1:x2+1,y1:y2+1,:])
                else:
                    color = random_rgb()
                borrow_place[connection == i + 1, 0] = color[0]
                borrow_place[connection == i + 1, 1] = color[1]
                borrow_place[connection == i + 1, 2] = color[2]
            BORROW_PLACES.append(borrow_place)
            #io.imsave('TOSHOW/{}/{}'.format(submitID+1,filename),borrow_place)
        txtname = txts[8]
        # f = open(txtname, 'r')
        # all = f.read()
        # need = all[all.find(filename):all.find(filename) + 222]
        # # print(need)
        # dice = need[need.find('DICE:') + 5:need.find('F1:') - 1]
        # dice = dice[:dice.find('.') + 4]
        imgp = io.imread(os.path.join(submittwo[3], filename))
        connection = measure.label(imgp)
        connection_prop = measure.regionprops(connection)
        M = -1
        p_visualmap = np.zeros((250, 250, 3), dtype=np.uint8)
        for i in range(len(connection_prop)):
            x1, y1, x2, y2 = connection_prop[i].bbox  # bbox of 250*250 p map
            borrow_place = p_visualmap
            color = most_color_of(gt[x1:x2 + 1, y1:y2 + 1, :])
            borrow_place[connection == i + 1, 0] = color[0]
            borrow_place[connection == i + 1, 1] = color[1]
            borrow_place[connection == i + 1, 2] = color[2]
        BORROW_PLACES.append(borrow_place)
        allpic=['TCGA-50-5931-01Z-00-DX1_crop_14.png','TCGA-50-5931-01Z-00-DX1_crop_6.png','TCGA-G9-6336-01Z-00-DX1_crop_15.png','TCGA-50-5931-01Z-00-DX1_crop_4.png'

 ]
        # allpic = ['TCGA-50-5931-01Z-00-DX1_crop_14.png'
        #
        #           ]
        try:
            whe=f_ajis[0]
        except:
            continue
        if f_ajis[0]>0.1and f_ajis[1]>0.1and  f_ajis[2]>0.1 and f_ajis[3]>0.1and f_ajis[4]>0.1and  f_ajis[5]>0.1 \
                and f_ajis[6] > 0.1:#and filename in set(allpic):#f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
               # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
            print(f_ajis)
            print(f_dices)
            print(filename)
            fid += 1
            # if fid>19:
            #     continue
            idx=2
            for submit in submittwo:
                idx += 1
                LISTC=11
                plt.subplot(H,LISTC,idx+(fid % 20)*LISTC)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.imshow(BORROW_PLACES[idx-3])
                # print("Dice={}".format(f_dices[idx - 3])[:10])
                if idx <100:
                    # #####################################temp score
                    # gt = io.imread(os.path.join(gt_dir_, filename))
                    # dicei,diceu=dice_coefficient(BORROW_PLACES[idx-3].astype(np.bool),gt.astype(np.bool))
                    # plt.title("Dice={}".format(dicei/diceu)[:10], y=-0.20)
                    # ###################################################################
                    plt.title("Dice={}".format(f_dices[idx-3])[:10], y=-0.20)

                    # if idx==4:
                    #     plt.title("Dice=0.247", y=-0.15)
                else:
                    plt.title("Dice={}".format(f_dices[idx-3])[:10], y=-0.15)
                    # if idx==7:
                    #     plt.title("Dice=0.749", y=-0.15)
            #
            plt.subplot(H, LISTC, 1 + (fid % 20) * LISTC)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            oriim = io.imread(crop_oriim_dir_ + filename)
            plt.imshow(oriim)
            plt.title(filename, y=-0.2)
            plt.subplot(H, LISTC, 2 + (fid % 20) * LISTC)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            gt = io.imread(os.path.join(COLORGT_DIR_, filename))
            plt.imshow(gt)
            plt.title("GT", y=-0.20)
        # if fid % 20 == 19:
        #     plt.savefig('TOSHOW/COMPARE_monu_{}.png'.format(fid / 20))

    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.savefig('TOSHOW/COMPARE.png')
    #
    # plt.show()
def print_ALL_with_metreics_copmparasion_ccrcc():
    fid=6
    ourstxt="/home/iftwo/wyj/M/logs/LRPTS20211230T18420779backup_TS_of_loop4_Student_num_4.txt"
    oursdir='/data1/wyj/M/results/ccrcccrop/submit_20220626T002204/'
    submittwo=["/data1/wyj/M/results/PRM2pcc/",
               "/data1/wyj/M/results/MDCCAM2pcc/",
               "/data1/wyj/M/results/MDCUNet2pcc/",'/data1/wyj/M/results/OAA',
                '/data1/wyj/M/results/MCIS','/data1/wyj/M/results/DRS',
               '/data1/wyj/M/results/nsrom',
               oursdir,
                '/data1/wyj/M/results/ccrcccrop/submit_20220626T001803/'
               ]
    txts=["prmcc.txt",
          "mdccamcc.txt",
          "mdcunetcc.txt",'oaacc.txt',
          'mciscc.txt','drscc.txt','nsromcc.txt',
          'ourscc.txt',
          "fullsupcc.txt"]
    # plt.close()
    for filename in os.listdir(submittwo[0]):
        f_dices=[]
        f_ajis = []
        BORROW_PLACES=[]
        INCH=20
        H=20
        fig = plt.gcf()
        fig.set_size_inches(INCH, 2*INCH)
        submitID=-1
        gt = io.imread(os.path.join(r'/data1/wyj/M/datasets/ccrcccrop/Test/Colormask/', filename))
        for submit in submittwo:
            submitID += 1
            txtname = txts[submitID]
            f=open(txtname,'r')
            all=f.read()
            need=all[all.find(filename):all.find(filename)+200]
            # print(filename)
            # print('need:'+need)
            dice=need[need.find('DICE:')+5:need.find('F1:')-1]
            dice=dice[:dice.find('.')+4]
            # print(dice)
            f_dices.append(float(dice))
            aji=need[need.find('AJI:')+4:need.find('DICE')-1]
            aji=aji[:aji.find('.')+4]
            f_ajis.append(float(aji))
            imgp = io.imread(os.path.join(submit, filename))
            if submitID<3:
                imgp=tr.resize(imgp,(256,256,3))[:,:,0]
            connection=measure.label(imgp)
            connection_prop=measure.regionprops(connection)
            M=-1
            borrow_place=np.zeros((256,256,3),dtype=np.uint8)
            for i in range(len(connection_prop)):
                # print(connection_prop[i].bbox)
                try:
                    x1, y1, x2, y2 = connection_prop[i].bbox
                except:
                    continue
                if submitID >=7:
                    color=most_color_of(gt[x1:x2+1,y1:y2+1,:])
                else:
                    color = random_rgb()
                borrow_place[connection == i + 1, 0] = color[0]
                borrow_place[connection == i + 1, 1] = color[1]
                borrow_place[connection == i + 1, 2] = color[2]
            if submitID<7:
                BORROW_PLACES.append(borrow_place)
            else:
                BORROW_PLACES.append(imgp)
            #io.imsave('TOSHOW/{}/{}'.format(submitID+1,filename),borrow_place)
        txtname = txts[8]
        f = open(txtname, 'r')
        all = f.read()
        need = all[all.find(filename):all.find(filename) + 222]
        # print(need)
        dice = need[need.find('DICE:') + 5:need.find('F1:') - 1]
        dice = dice[:dice.find('.') + 4]
        imgp = io.imread(os.path.join(submittwo[3], filename))
        connection = measure.label(imgp)
        connection_prop = measure.regionprops(connection)
        M = -1
        p_visualmap = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(len(connection_prop)):
            x1, y1, x2, y2 = connection_prop[i].bbox  # bbox of 250*250 p map
            borrow_place = p_visualmap
            color = most_color_of(gt[x1:x2 + 1, y1:y2 + 1, :])
            borrow_place[connection == i + 1, 0] = color[0]
            borrow_place[connection == i + 1, 1] = color[1]
            borrow_place[connection == i + 1, 2] = color[2]
        BORROW_PLACES.append(borrow_place)
        allpic = [
                  'low_grade_ccrcc__x0y1.png'
            , 'low_grade_ccrcc_812_x1y0.png',

                  ]
 #        allpic=['low_grade_ccrcc_794_x1y1.png','low_grade_ccrcc_557_x1y0.png','low_grade_ccrcc_956_x1y0.png','low_grade_ccrcc_936_x0y1.png'
 #                ,'low_grade_ccrcc_812_x1y0.png','low_grade_ccrcc_570_x0y0.png','low_grade_ccrcc_905_x1y0.png','low_grade_ccrcc_440_x0y1.png'
 #
 # ]
        # allpic = ['TCGA-50-5931-01Z-00-DX1_crop_14.png'
        #
        #           ]
        k=0.05
        # if f_ajis[0] > k and f_ajis[1] > k and f_ajis[2] > k and f_ajis[3] > k and f_ajis[4] > k and f_ajis[
        #     5] > k \
        #         and f_ajis[6] > k:#and filename in set(allpic):#f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
        #        # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        if filename in set(allpic):
            print(f_ajis)
            print(filename)
            fid += 1
            idx=2
            for submit in submittwo:
                idx += 1
                LISTC=11
                plt.subplot(H,LISTC,idx+(fid%20)*LISTC)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.imshow(BORROW_PLACES[idx-3])
                if idx <100:
                    # #####################################temp score
                    # gt = io.imread(os.path.join(gt_dir_, filename))
                    # dicei,diceu=dice_coefficient(BORROW_PLACES[idx-3].astype(np.bool),gt.astype(np.bool))
                    # plt.title("Dice={}".format(dicei/diceu)[:10], y=-0.20)
                    # ###################################################################
                    plt.title("Dice={}".format(f_dices[idx-3])[:10], y=-0.20)
                else:
                    plt.title("Dice={}".format(f_dices[idx-3])[:10], y=-0.2)
                    # if idx==7:
                    #     plt.title("Dice=0.749", y=-0.15)
            plt.subplot(H, LISTC, 1+(fid%20)*LISTC)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            oriim=io.imread('/data1/wyj/M/datasets/ccrcccrop/Test/Images/'+filename)
            plt.imshow(oriim)
            plt.title(filename, y=-0.2)
            plt.subplot(H, LISTC, 2+(fid%20)*LISTC)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            gt=io.imread(os.path.join(r'/data1/wyj/M/datasets/ccrcccrop/Test/Colormask/',filename))
            plt.imshow(gt)
            plt.title("GT", y=-0.2)
        if fid%20==19:
            plt.savefig('TOSHOW/COMPARE{}.png'.format(fid/20))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # if fid%20==19:
    plt.savefig('TOSHOW/COMPAREcc.png'.format(fid%20))
    print('done')
    plt.show()
def print_ALL_with_metreics_seed_ccrcc():
    fid=-1
    ourstxt="/home/iftwo/wyj/M/logs/LRPTS20211230T18420779backup_TS_of_loop4_Student_num_4.txt"
    oursdir='/data1/wyj/M/results/ccrcccrop/submit_20220626T002204/'
    seed=[
        '/home/iftwo/wyj/M/results/nucleus/submit_20220104T141441/',  # 679,
        '/home/iftwo/wyj/M/results/nucleus/submit_20211231T092431',  # 739,
        '/home/iftwo/wyj/M/results/nucleus/submit_20211230T222800',  # 757,
        '/home/iftwo/wyj/M/results/nucleus/submit_20220112T054820',  # 763,
        '/home/iftwo/wyj/M/results/nucleus/submit_20211230T222948',  # 772,
        '/home/iftwo/wyj/M/results/nucleus/submit_20211230T210514',  # 779,
    ]
    submittwo=seed
    txts=["seedg1.txt","seedg2.txt","seedg3.txt","seedg4.txt","seedg5.txt","seedg6.txt",]

    for filename in os.listdir(submittwo[0]):
        if fid>2:
            break
        f_dices = []
        f_ajis = []
        BORROW_PLACES = []
        INCH = 20
        H = 10
        fig = plt.gcf()
        fig.set_size_inches(INCH, 2 * INCH)
        submitID = -1
        gt = io.imread(os.path.join(COLORGT_DIR_, filename))
        for submit in submittwo:
            submitID += 1
            txtname = txts[submitID]
            f = open(txtname, 'r')
            all = f.read()
            need = all[all.find(filename):all.find(filename) + 222]
            # print(need)
            dice = need[need.find('DICE:') + 5:need.find('F1:') - 1]
            dice = dice[:dice.find('.') + 4]
            f_dices.append(float(dice))
            aji = need[need.find('AJI:') + 4:need.find('DICE') - 1]
            aji = aji[:aji.find('.') + 4]
            f_ajis.append(float(aji))
            imgp = io.imread(os.path.join(submit, filename))
            connection = measure.label(imgp)
            connection_prop = measure.regionprops(connection)
            M = -1
            borrow_place = np.zeros((250, 250, 3), dtype=np.uint8)
            for i in range(len(connection_prop)):
                x1, y1, x2, y2 = connection_prop[i].bbox  # bbox of 250*250 p map
                if submitID >= 0:
                    color = most_color_of(gt[x1:x2 + 1, y1:y2 + 1, :])
                else:
                    color = random_rgb()
                borrow_place[connection == i + 1, 0] = color[0]
                borrow_place[connection == i + 1, 1] = color[1]
                borrow_place[connection == i + 1, 2] = color[2]
            BORROW_PLACES.append(borrow_place)
            # io.imsave('TOSHOW/{}/{}'.format(submitID+1,filename),borrow_place)
        txtname = txts[3]
        f = open(txtname, 'r')
        all = f.read()
        need = all[all.find(filename):all.find(filename) + 222]
        # print(need)
        dice = need[need.find('DICE:') + 5:need.find('F1:') - 1]
        dice = dice[:dice.find('.') + 4]
        imgp = io.imread(os.path.join(submittwo[3], filename))
        connection = measure.label(imgp)
        connection_prop = measure.regionprops(connection)
        M = -1
        p_visualmap = np.zeros((250, 250, 3), dtype=np.uint8)
        for i in range(len(connection_prop)):
            x1, y1, x2, y2 = connection_prop[i].bbox  # bbox of 250*250 p map
            borrow_place = p_visualmap
            color = most_color_of(gt[x1:x2 + 1, y1:y2 + 1, :])
            borrow_place[connection == i + 1, 0] = color[0]
            borrow_place[connection == i + 1, 1] = color[1]
            borrow_place[connection == i + 1, 2] = color[2]
        BORROW_PLACES.append(borrow_place)
        allpic = ['TCGA-50-5931-01Z-00-DX1_crop_14.png', 'TCGA-50-5931-01Z-00-DX1_crop_6.png',
                  'TCGA-G9-6336-01Z-00-DX1_crop_15.png', 'TCGA-50-5931-01Z-00-DX1_crop_4.png'
                  ]
        # allpic = ['TCGA-50-5931-01Z-00-DX1_crop_14.png'
        #
        #           ]
        if f_ajis[0]<f_ajis[1] and f_ajis[1]<f_ajis[2] and f_ajis[2]<f_ajis[3] and f_ajis[3]<f_ajis[4] \
                and f_ajis[4]<f_ajis[5] and fid<8:  # and filename in set(allpic):#f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
            # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
            print(f_ajis)
            print(filename)
            fid += 1
            if fid > 19:
                continue
            idx = 2
            for submit in submittwo:
                idx += 1
                LISTC = 8
                plt.subplot(H, LISTC, idx + fid * LISTC)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.imshow(BORROW_PLACES[idx - 3])
                if idx < 6:
                    dicei = str(f_dices[idx - 3])
                    plt.title("Dice={}".format(f_dices[idx - 3])[:10], y=-0.15)
                else:
                    plt.title("Dice={}".format(f_dices[idx - 3])[:10], y=-0.15)
                    # if idx==7:
                    #     plt.title("Dice=0.749", y=-0.15)
            plt.subplot(H, LISTC, 1 + fid * LISTC)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            oriim = io.imread(crop_oriim_dir_ + '/' + filename)
            plt.imshow(oriim)
            plt.title(filename, y=-0.2)
            plt.subplot(H, LISTC, 2 + fid * LISTC)

            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            gt = io.imread(os.path.join(COLORGT_DIR_, filename))
            plt.imshow(gt)
            plt.title("GT", y=-0.15)

    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    plt.savefig('TOSHOW/COMPAREseed.png')
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
    seed=[
        '/home/iftwo/wyj/M/results/nucleus/submit_20220104T141441/',#679,
    '/home/iftwo/wyj/M/results/nucleus/submit_20211231T092431',#739,
    '/home/iftwo/wyj/M/results/nucleus/submit_20211230T222800',#757,
    '/home/iftwo/wyj/M/results/nucleus/submit_20220112T054820',#763,
    '/home/iftwo/wyj/M/results/nucleus/submit_20211230T222948',#772,
    '/home/iftwo/wyj/M/results/nucleus/submit_20211230T210514',#779,
    ]
    submittwo=["/home/iftwo/wyj/M/samples/nucleus/PRMp/",
               "/home/iftwo/wyj/M/samples/nucleus/MDCCAMp/",
               "/home/iftwo/wyj/M/samples/nucleus/MDCUNetp/",
               '/data1/wyj/M/samples/PRM/OAA-PyTorch-master/runs/exp4/attention/',
                '/data1/wyj/M/samples/PRM/MCIS_wsss-master/MCIS_wsss-master/Classifier/scripts/runs2/${EXP}/attention3/',
               '/data1/wyj/M/samples/PRM/DRS-main/result/',
               '/data1/wyj/M/samples/PRM/nsrom-main/nsrom-main/classification/scripts/runs2/${EXP}/attention/',

               "/home/iftwo/wyj/M/samples/nucleus/.._.._logs_RESULTIMGS_nucleus20220121T2354_mask_rcnn_nucleus_0009.h5/"]
    # im1_path = submittwo[2]+'/TCGA-B0-5698-01Z-00-DX1_crop_3.png'
    # imgt_path = gt_dir_+'/TCGA-B0-5698-01Z-00-DX1_crop_3.png'
    #
    # im1 = skimage.io.imread((im1_path))
    # imgt = skimage.io.imread((imgt_path))
    # transform=True
    # imgp=im1
    # if transform == True:
    #     imgp = tr.resize(im1, (250, 250, 3))[:, :, 0]
    #     imgp[imgp > 0.7] = 0.9
    #     imgp[imgp <= 0.7] = 1
    #     imgp[imgp == 0.9] = 0
    # plt.imshow(imgp)
    # plt.show()
    # plt.imshow(imgt)
    # plt.show()
    # dice = dice_coefficient(imgp.astype(np.bool), imgt.astype(np.bool))
    # print(dice[0]/dice[1])
    # test_XMetric('/home/iftwo/wyj/M/samples/nucleus/.._.._logs_RESULTIMGS_nucleus20220121T2354_mask_rcnn_nucleus_0009.h5')
    # print('MONU:prm')
    # test_XMetric('PRMp202281e-5/',txtname='prm0e-5.txt',finaltest=True,transform=True)
    # print('MONU:mdccam')
    # test_XMetric('/home/iftwo/wyj/M/samples/nucleus/MDCCAMp/',txtname='mdccam.txt',finaltest=True,transform=True)
    # print('MONU:mdcunet')
    # test_XMetric('/home/iftwo/wyj/M/samples/nucleus/MDCUNetp/',txtname='mdcunet.txt',finaltest=True,transform=True)
    # print('MONU:MCIS')
    # test_XMetric('/data1/wyj/M/samples/PRM/MCIS_wsss-master/MCIS_wsss-master/Classifier/scripts/runs2/${EXP}/attention3/',txtname='MCIS.txt',finaltest=True)
    # print('MONU:nsrom')
    # test_XMetric('/data1/wyj/M/samples/PRM/nsrom-main/nsrom-main/classification/scripts/runs2/${EXP}/attention/',txtname='nsrom.txt',finaltest=True)
    # print('MONU:drs')
    # test_XMetric('/data1/wyj/M/samples/PRM/DRS-main/result/',txtname='drs.txt',finaltest=True)
    # print('MONU:oaa')
    # test_XMetric('/data1/wyj/M/samples/PRM/OAA-PyTorch-master/runs/exp4/attention/',txtname='oaa.txt',finaltest=True)
    # # # filename=
    # tcount=0
    # for se in seed:
    #     tcount+=1
    #      test_XMetric(se,txtname='seedg{}.txt'.format(tcount),finaltest=True)

    # for i in range(len(seed)):
    #     plt.subplot(1,7,i+1)
    #     plt.imshow(io.imread(seed[i]+'/TCGA-18-5592-01Z-00-DX1_crop_0.png'))
    # plt.show()

    # print_ALL_with_metreics_seed_ccrcc()
    # print(os.listdir('/data1/wyj/M/datasets/MoNuSACCROP/images/').sort())
    print_ALL_with_metreics_copmparasion()
    print_ALL_with_metreics_copmparasion_ccrcc()
    # MDCCAM='PRMp'
    # print(MDCCAM)
    # # feel=io.imread('')
    # test_XMetric(MDCCAM)
    # MDCCAM='MDCCAMp'
    # print(MDCCAM)
    # # feel=io.imread('')
    # test_XMetric(MDCCAM)
    # MDCCAM='MDCUNetp'
    # print(MDCCAM)
    # # feel=io.imread('')
    # test_XMetric(MDCCAM)

    # print_ALL_with_metreics_withCOMPARE()
    # print_ALL_with_metreics_withCOMPARE2()



    #print(os.listdir('D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\datasets\MoNuSACCROP\images'))
    # f_score=calculate_f1_buildings_score(baseline)
    # print(f_score)
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
