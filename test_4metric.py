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
GT_dir_ = r'../../datasets/MoNuSAC_mask'
gt_dir_ = r'../../datasets/MoNuSACCROP/mask'
instance_GT_dir_ = r'../../datasets/MoNuSACGT/stage1_train'


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
    if precision == 0 or recall == 0:
        return 0
    f_score = 2 * precision * recall / (precision + recall)
    return f_score,tp,fp,fn

def test_XMetric(mask_dir=mask_dir_, gt_dir=gt_dir_, Val=True, txtname='list.txt', model_name=''):
    for phase in ['Train','Val']:
        lst1 = np.zeros((len(os.listdir(mask_dir)), 14), dtype=float)
        image_id = 0
        txt = '../../logs/' + '{}.txt'.format(model_name[model_name.find('logs/') + 5:]).replace('/', '_')
        f = open(txt, 'a')
        # txt='listtest.txt'
        # f=open(txt,'a')
        for filename in os.listdir(mask_dir):
            if phase=='Train':
                if (filename[:23] not in VAL_IMAGE_IDS) & Val:
                    continue
            else:
                if (filename[:23] in VAL_IMAGE_IDS) & Val:
                    continue
            name_no = filename[:-4]
            imgp = io.imread(os.path.join(mask_dir, filename))
            p = imgp.astype(np.bool)
            imgg = io.imread(os.path.join(gt_dir, name_no + '.png'))
            g = imgg.astype(np.bool)
            aji = PQ.get_fast_aji(g, p)
            I, U = dice_coefficient(p, g)
            dice = I / U
            F1s, accuracy, IoU, prec, rec = F1(p, g)
            objF1,tp,fp,fn=calculate_f1(p,g)
            hd = hausdorff(p, g)
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
            print(toprint)
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
        toprint='{} mean_ AJI:{} OLD_DICE:{} DICE:{} F1:{} accuracy:{} IOU:{} prec:{} rec:{} HD:{} objF1:{}'.format(phase,
            avacc[0] / image_id, result, avacc[1] / image_id, avacc[2] / image_id, avacc[3] / image_id,
            avacc[4] / image_id, avacc[5] / image_id, avacc[6] / image_id, avacc[7] / image_id,f_score)
        f.write(toprint)
        f.write('____________________________________________________________________________')
        print(toprint)
        print('____________________________________________________________________________')
    test(mask_dir)
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


import matplotlib.pyplot as plt

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
    # test_XMetric(r'/home/iftwo/wyj/M/results/nucleus/submit_20211230T190219')
    if args.model == None:
        print("default path!")
        testh5 = r"/home/iftwo/wyj/M/logs/LRPTS20211230T18420779backup/TS_of_loop0/Student_num_2"
    else:
        testh5 = str(args.model)
    test_model(testh5)
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
