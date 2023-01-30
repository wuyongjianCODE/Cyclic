import shutil,os
target=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\samples\nucleus\shit'
sourcedir=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\images'
dd1=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\datasets\nucleus - 副本\stage1_train'
dd=r'../../datasets/MoNuSACGT\\stage1_train\\stage1_train'
import os
import sys
import keras
import random
import re
import time
import skimage
from skimage import io,measure
import numpy as np
import test_4metric

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
mask_dir_=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\submit_20201014T174123'#this is the GT baseline
#mask_dir_=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\submit_20201014T174123'#this is the GT baseline
#r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\submit_20201013T232206'
#r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\submit_20201004T212335'
# GT_dir_=r'D:\BaiduNetdiskDownload\BrestCancer\MoNuSAC-master\MoNuSAC_mask'
# gt_dir_=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\datasets\MoNuSACCROP\mask'
GT_dir_=r'../../datasets/MoNuSAC_mask'
gt_dir_=r'../../datasets/MoNuSACCROP/mask'
crop_oriim_dir_=r'/data1/wyj/M/datasets/MoNuSACCROP/images/'
instance_GT_dir_=r'../../datasets/MoNuSACGT2/stage1_train'
COLORGT_DIR_ = 'Nucleus_Colored_GT'
def dice_coefficient(a, b):
    """dice coefficient 2nt/na + nb."""
    overlap = a * b
    I = overlap.sum() * 2
    U = (a.sum() + b.sum())

    return (I, U)
def AJI0(p, g):
    p_ind = np.ones(p.shape[2])
    g_ind = np.ones(g.shape[2])
    I = 0
    U = 0
    for i in range(g.shape[2]):
        iou0 = 0
        ind = -1
        for j in range(p.shape[2]):
            iou = (g[..., i] * p[..., j]).sum() / (g[..., i] + p[..., j]).astype(np.bool).sum()
            if iou > iou0:
                iou0 = iou
                ind = j
        if ind != -1:
            p_ind[ind] = 0
            g_ind[i] = 0

            I = I + (g[..., i] * p[..., ind]).sum()

            U = U + (g[..., i] + p[..., ind]).astype(np.bool).sum()
            p[..., ind] = 0
        # else:
        #    I = I+(g[...,i]*p[...,0]).sum()

        #    U = U+(g[...,i]+p[...,0]).astype(np.bool).sum()

    U = U + p.sum() + g[..., g_ind.astype(np.bool)].sum()
    return I, U
def AJI(p0, g0):
    p=np.zeros([1000,1000,1])
    g=np.zeros([1000,1000,1])
    p[:,:,0]=p0
    p[:,:,0]=p0
    p_ind = np.ones(1)
    g_ind = np.ones(1)
    I = 0
    U = 0
    for i in range(1):
        iou0 = 0
        ind = -1
        for j in range(1):
            iou = (g[..., i] * p[..., j]).sum() / (g[..., i] + p[..., j]).astype(np.bool).sum()
            if iou > iou0:
                iou0 = iou
                ind = j
        if ind != -1:
            p_ind[ind] = 0
            g_ind[i] = 0

            I = I + (g[..., i] * p[..., ind]).sum()

            U = U + (g[..., i] + p[..., ind]).astype(np.bool).sum()
            p[..., ind] = 0
        # else:
        #    I = I+(g[...,i]*p[...,0]).sum()

        #    U = U+(g[...,i]+p[...,0]).astype(np.bool).sum()

    U = U + p.sum() + g[..., g_ind.astype(np.bool)].sum()
    return I, U
def test(mask_dir=mask_dir_,gt_dir=gt_dir_,Val=True):
    lst1=np.zeros((len(os.listdir(mask_dir)), 3), dtype = float)
    image_id=0
    txt='listtest.txt'
    f=open(txt,'a')
    for filename in os.listdir(mask_dir):
        if (filename[:23] in VAL_IMAGE_IDS) & Val:
            continue
        name_no=filename[:-4]
        imgp=io.imread(os.path.join(mask_dir,filename))
        p=imgp.astype(np.bool)
        #imgg=io.imread(os.path.join(gt_dir,name_no+'_mask.png'))
        imgg = io.imread(os.path.join(gt_dir, name_no + '.png'))
        g=imgg.astype(np.bool)
        I, U = dice_coefficient(p, g)
        #I, U = AJI(p, g)
        lst1[image_id, 0] = I
        lst1[image_id, 1] = U
        lst1[image_id, 2] = I/U
        image_id=image_id+1
        # print('{}'.format(I/U))
    avacc = lst1.sum(axis=0)
    result=avacc[0] / avacc[1]
    print('mean val gt dice : {}'.format(avacc[0] / avacc[1]))
    image_id=0
    f.write(mask_dir+' {}'.format(avacc[0] / avacc[1])+' ')
    for filename in os.listdir(mask_dir):
        name_no=filename[:-4]
        imgp=io.imread(os.path.join(mask_dir,filename))
        p=imgp.astype(np.bool)
        #imgg=io.imread(os.path.join(gt_dir,name_no+'_mask.png'))
        imgg = io.imread(os.path.join(gt_dir, name_no + '.png'))
        g=imgg.astype(np.bool)
        I, U = dice_coefficient(p, g)
        #I, U = AJI(p, g)
        lst1[image_id, 0] = I
        lst1[image_id, 1] = U
        lst1[image_id, 2] = I/U
        image_id=image_id+1
        #print(filename+'  dice : {}'.format(I/U))
    avacc = lst1.sum(axis=0)
    print('mean all gt dice : {}'.format(avacc[0] / avacc[1]))
    return result
    # print('mean  dice2 : {}'.format(lst1[:,2].mean()))
def F1(premask,groundtruth):
#二值分割图是一个波段的黑白图，正样本值为1，负样本值为0
#通过矩阵的逻辑运算分别计算出tp,tn,fp,fn
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = float(np.logical_and(premask, groundtruth).sum())  # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()

    #然后根据公式分别计算出这几种指标
    prec = true_pos / (true_pos + false_pos + 1e-6)
    rec = true_pos / (true_pos + false_neg + 1e-6)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg + 1e-6)
    F1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    IoU = true_pos / (true_pos + false_neg + false_pos + 1e-6)
    return F1,accuracy,IoU,prec,rec

def testF1(mask_dir=mask_dir_,gt_dir=GT_dir_):
    lst1=np.zeros((len(os.listdir(mask_dir)), 5), dtype = float)
    image_id=0
    # txt='listall.txt'
    # f=open(txt,'a')
    for filename in os.listdir(mask_dir):
        if (filename[:23] not in VAL_IMAGE_IDS):
            continue
        name_no=filename[:-4]
        imgp=io.imread(os.path.join(mask_dir,filename))
        p=imgp.astype(np.bool)
        #imgg=io.imread(os.path.join(gt_dir,name_no+'_mask.png'))
        imgg = io.imread(os.path.join(gt_dir, name_no + '_mask.png'))
        g=imgg.astype(np.bool)
        F1s,accuracy,IoU,prec,rec= F1(p, g)
        #I, U = AJI(p, g)
        lst1[image_id, 0] = F1s
        lst1[image_id, 1] = accuracy
        lst1[image_id, 2] = IoU
        lst1[image_id, 3] = prec
        lst1[image_id, 4] = rec
        image_id=image_id+1
        print(filename+'  GT F1 : {}'.format(F1s)+' accuracy:{}'.format(accuracy)+" IoU:{}".format(IoU)+' prec:{}'.format(prec)+' rec:{}'.format(rec))
    avacc = lst1.sum(axis=0)
    print('mean val GT F1 : {}'.format(avacc[0] /14)+' accuracy:{}'.format(avacc[1] /14)+" IoU:{}".format(avacc[2] /14)+' prec:{}'.format(avacc[3] /14)+' rec:{}'.format(avacc[4] /14))
import glob
def testGT_onto_dice(maskdir, iteration_flag=True,iteration=2,gt_dir=instance_GT_dir_):

    ALL_IMAGE_IDS=os.listdir(instance_GT_dir_)
    TRAIN_IMAGE_IDS=list(set(ALL_IMAGE_IDS) - set(VAL_IMAGE_IDS))
    IDS=[TRAIN_IMAGE_IDS,VAL_IMAGE_IDS]
    Phase=['train','val']
    for turn in range(2):
        temp_FOLDER_dices = np.zeros((30), dtype=np.float)
        INDEX = 0
        print('phase ： '+Phase[turn]+'________________________________________________________')
        for dirname in os.listdir(instance_GT_dir_):
            if (dirname[:23] not in IDS[turn]):
                continue
            if iteration_flag==True:
                focus_files=glob.glob(maskdir+'/'+dirname+'/masks/*iteration_'+str(iteration)+'*')
            else:
                not_focus_files = glob.glob(maskdir + '/' + dirname + '/masks/*iteration_*')
                all_files=glob.glob(maskdir + '/' + dirname + '/masks/*')
                focus_files=ret_list = list(set(all_files)^set(not_focus_files))
            #print(focus_files)
            instance_GT_dir__current_NAME=os.path.join(instance_GT_dir_,dirname)+'/masks'
            mask_total_nums=len(focus_files)
            temp_NAME_dices=np.zeros((mask_total_nums),dtype=np.float)
            GT_nums=len(os.listdir(instance_GT_dir__current_NAME))
            temp_NAME_GT_arrays = np.zeros((1000,1000,GT_nums), dtype=np.bool)
            count_gt=0
            for gtfile in os.listdir(instance_GT_dir__current_NAME):
                imgt_path = os.path.join(instance_GT_dir__current_NAME, gtfile)
                temp_NAME_GT_arrays[:,:,count_gt] = io.imread(imgt_path).astype(np.bool)
                count_gt=count_gt+1
            # print(dirname+' GT array generate successfully!!')
            i=0
            for file in focus_files:
                imp_path=file
                imp=io.imread(imp_path).astype(np.bool)
                connection=measure.label(imp)
                connection_prop=measure.regionprops(connection)
                x=connection_prop[0].centroid[0].astype(np.int)
                y = connection_prop[0].centroid[1].astype(np.int)
                center_point_array=temp_NAME_GT_arrays[x,y,:]
                M = np.argmax(center_point_array)
                if M==0:
                    x1, y1, x2, y2 = connection_prop[0].bbox
                    x = int((x1 + x2) / 2)
                    y = int((y1 + y2) / 2)
                    center_region_array=temp_NAME_GT_arrays[x1:x2+1,y1:y2+1,:]
                    center_region_array_sum=np.sum(center_region_array,axis=(0,1))
                    M = np.argmax(center_region_array_sum)
                    # if M==0:
                    #     print('what!!???')
                    # print(M)
                I, U = dice_coefficient(imp, temp_NAME_GT_arrays[:, :, M])
                current_i2i_dice = I / U
                temp_NAME_dices[i] = current_i2i_dice
                #print(imp_path + ' onto dice : ' + str(current_i2i_dice))
                i = i + 1
            print(dirname+' average onto dice: '+str(sum(temp_NAME_dices)/len(temp_NAME_dices))+
                  '      NONZERO onto dice: ' + str(sum(temp_NAME_dices) / len(temp_NAME_dices[temp_NAME_dices!=0])) +
                  '      false classified cell which is not GT cancer cell: ' + str(len(temp_NAME_dices[temp_NAME_dices==0]))+
                  '      this generation seed cell numbers: '+str(mask_total_nums))
            temp_FOLDER_dices[INDEX] = sum(temp_NAME_dices) / len(temp_NAME_dices[temp_NAME_dices!=0])
            INDEX=INDEX+1
        print('Average NONZERO Onto Dice: ' + str(sum(temp_FOLDER_dices) / len(temp_FOLDER_dices[temp_FOLDER_dices != 0])))


def testGT(mask_dir=mask_dir_,gt_dir=GT_dir_):
    lst1=np.zeros((len(os.listdir(mask_dir)), 3), dtype = float)
    image_id=0
    # txt='listall.txt'
    # f=open(txt,'a')
    for filename in os.listdir(mask_dir):
        if (filename[:23] not in VAL_IMAGE_IDS):
            continue
        name_no=filename[:-4]
        imgp=io.imread(os.path.join(mask_dir,filename))
        p=imgp.astype(np.bool)
        #imgg=io.imread(os.path.join(gt_dir,name_no+'_mask.png'))
        imgg = io.imread(os.path.join(gt_dir, name_no + '_mask.png'))
        g=imgg.astype(np.bool)
        I, U = dice_coefficient(p, g)
        #I, U = AJI(p, g)
        lst1[image_id, 0] = I
        lst1[image_id, 1] = U
        lst1[image_id, 2] = I/U
        image_id=image_id+1
        print(filename+'  GT dice : {}'.format(I/U))
    avacc = lst1.sum(axis=0)
    print('mean val GT dice : {}'.format(avacc[0] / avacc[1]))
    image_id=0
    for filename in os.listdir(mask_dir):
        name_no=filename[:-4]
        imgp=io.imread(os.path.join(mask_dir,filename))
        p=imgp.astype(np.bool)
        imgg=io.imread(os.path.join(gt_dir,name_no+'_mask.png'))
        #imgg = io.imread(os.path.join(gt_dir, name_no + '.png'))
        g=imgg.astype(np.bool)
        I, U = dice_coefficient(p, g)
        #I, U = AJI(p, g)
        lst1[image_id, 0] = I
        lst1[image_id, 1] = U
        lst1[image_id, 2] = I/U
        image_id=image_id+1
        print(filename+'  GT dice : {}'.format(I/U))
    avacc = lst1.sum(axis=0)
    print('mean GT dice : {}'.format(avacc[0] / avacc[1]))
    #f.write(mask_dir+' {}'.format(avacc[0] / avacc[1])+' ')
    # print('mean  dice2 : {}'.format(lst1[:,2].mean()))
def random_rgb():
    r = np.random.rand()*255
    g = np.random.rand()*255
    b = np.random.rand()*255
    return np.array([r, g, b]).astype(np.uint8)
import matplotlib.pyplot as plt
def bijective(im,cg):
    imcon=measure.label(im[:,:,0]>0)
    imcon_prop=measure.regionprops(imcon)
    newim=np.zeros((250,250,3),np.uint8)
    for i in range(len(imcon_prop)):
        x1, y1, x2, y2 = imcon_prop[i].bbox  # bbox of 250*250 p map
        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)
        color = cg[x,y,:]
        if np.max(color)==0:
            print('fail to biject')
            color=random_rgb()
        newim[imcon==i+1,0]=color[0]
        newim[imcon==i+1,1]=color[1]
        newim[imcon==i+1,2]=color[2]
    return newim

def bijective_coloured_visual0(p_dirs,cg):
    prefix_name = ''
    if not os.path.exists(cg):
        os.mkdir(cg)
    for filename in os.listdir(p_dirs[0]):
        if prefix_name == filename[:23]:
            continue
        prefix_name = filename[:23]
        instance_GT_dir__current_NAME = os.path.join(instance_GT_dir_, prefix_name) + '/masks/'
        GT_nums = len(os.listdir(instance_GT_dir__current_NAME))
        temp_NAME_GT_arrays = np.zeros((1000, 1000, GT_nums), dtype=np.bool)
        temp_NAME_GT_colors = np.zeros((3, GT_nums), dtype=np.uint8)
        gt_visualmap=np.zeros((1000,1000,3),dtype=np.uint8)
        count_gt = 0
        for gtfile in os.listdir(instance_GT_dir__current_NAME):
            imgt_path = os.path.join(instance_GT_dir__current_NAME, gtfile)
            gttemp= io.imread(imgt_path).astype(np.bool)
            temp_NAME_GT_arrays[:, :, count_gt] =gttemp
            gttemp_color = random_rgb()
            temp_NAME_GT_colors[:, count_gt] =gttemp_color
            gt_visualmap[gttemp!=0]=gttemp_color
            count_gt = count_gt + 1
        # plt.imshow(gt_visualmap)
        # plt.show()
        for i in range(16):
            X = i % 4
            Y = i// 4
            io.imsave(os.path.join(cg,prefix_name+"_crop_"+str(i)+'.png'),gt_visualmap[250 * X:250 * X + 250, 250 * Y:250 * Y + 250, :])

def bijective_coloured_visual(p_dirs,savpath='Plot_Bijective_Colored/'):
    if not os.path.exists(savpath):
        os.mkdir(savpath)
    if not os.path.exists(COLORGT_DIR_):
        os.mkdir(COLORGT_DIR_)
    for filename in os.listdir(p_dirs[0]):
        prefix_name = filename[:23]
        instance_GT_dir__current_NAME = os.path.join(instance_GT_dir_, prefix_name) + '/masks/'
        IM=io.imread(os.path.join(instance_GT_dir_, prefix_name) + '/images/'+prefix_name+'.png')
        GT_nums = len(os.listdir(instance_GT_dir__current_NAME))
        temp_NAME_GT_arrays = np.zeros((1000, 1000, GT_nums), dtype=np.bool)
        temp_NAME_GT_colors = np.zeros((3, GT_nums), dtype=np.uint8)
        gt_visualmap=np.zeros((1000,1000,3),dtype=np.uint8)
        count_gt = 0
        for gtfile in os.listdir(instance_GT_dir__current_NAME):
            imgt_path = os.path.join(instance_GT_dir__current_NAME, gtfile)
            gttemp= io.imread(imgt_path).astype(np.bool)
            temp_NAME_GT_arrays[:, :, count_gt] =gttemp
            gttemp_color = random_rgb()
            temp_NAME_GT_colors[:, count_gt] =gttemp_color
            gt_visualmap[gttemp!=0]=gttemp_color
            count_gt = count_gt + 1

        for iteri in range(len(p_dirs)):
            p_dir=p_dirs[iteri]
            regen_p_dir=p_dir.replace('submit','Colored')
            if not os.path.exists(regen_p_dir):
                os.mkdir(regen_p_dir)
            im = io.imread(os.path.join(p_dir, filename)) ##250*250 predict map
            im_should_save_here=os.path.join(regen_p_dir, filename)
            crop_id = int(filename[filename.find('crop_') + 5:-4])
            X = crop_id % 4
            Y = crop_id // 4

            connection=measure.label(im)
            connection_prop=measure.regionprops(connection)
            M=-1
            p_visualmap=np.zeros((1000,1000,3),dtype=np.uint8)
            for i in range(len(connection_prop)):
                x1, y1, x2, y2 = connection_prop[i].bbox #bbox of 250*250 p map
                x = int((x1 + x2) / 2)
                y = int((y1 + y2) / 2)
                center_region_array = temp_NAME_GT_arrays[250* X+x1:250* X+x2 + 1, 250* Y+y1:250* Y+y2 + 1, :]
                center_region_array_sum = np.sum(center_region_array, axis=(0, 1))
                M = np.argmax(center_region_array_sum)
                borrow_place=p_visualmap[250* X:250*X+250,250* Y:250*Y+250,:]
                color=temp_NAME_GT_colors[:,M]
                borrow_place[connection == i + 1, 0] = color[0]
                borrow_place[connection == i + 1, 1] = color[1]
                borrow_place[connection == i + 1, 2] = color[2]
            plt.subplot(6, 5, iteri+1)
            plt.imshow(borrow_place)
            io.imsave(im_should_save_here,borrow_place)
        plt.subplot(6, 5, len(p_dirs)+1)
        temp_sight=gt_visualmap[250* X:250*X+250,250* Y:250*Y+250,:]
        plt.imshow(temp_sight)
        io.imsave(os.path.join(COLORGT_DIR_,filename),temp_sight)
        plt.subplot(6, 5, len(p_dirs)+2)
        plt.imshow(IM[250* X:250*X+250,250* Y:250*Y+250,:])
            # plt.subplot(2, len(p_dirs), iteri+1)
            # plt.imshow(borrow_place)
            # plt.subplot(2, len(p_dirs), len(p_dirs)+1)
            # plt.imshow(gt_visualmap[250* X:250*X+250,250* Y:250*Y+250,:])
            # plt.subplot(2, len(p_dirs), len(p_dirs)+2)
            # plt.imshow(IM[250* X:250*X+250,250* Y:250*Y+250,:])
        plt.savefig(savpath+'{}_BiCol.png'.format(filename))
        print(filename)
        # for name in os.listdir(instance_GT_dir_):
        #     instance_GT_dir__current_NAME = os.path.join(instance_GT_dir_, name) + '/masks/'
        #     GT_nums = len(os.listdir(instance_GT_dir__current_NAME))
        #     temp_NAME_GT_arrays = np.zeros((1000, 1000, GT_nums), dtype=np.bool)
        #     temp_NAME_GT_colors = np.zeros((3, GT_nums), dtype=np.uint8)
        #     count_gt = 0
        #     for gtfile in os.listdir(instance_GT_dir__current_NAME):
        #         imgt_path = os.path.join(instance_GT_dir__current_NAME, gtfile)
        #         temp_NAME_GT_arrays[:, :, count_gt] = io.imread(imgt_path).astype(np.bool)
        #         temp_NAME_GT_colors[:, count_gt] = random_rgb()
        #         count_gt = count_gt + 1


def gen_submits_from_LRPTSDIR(LRPTSDIR="/data1/wyj/M/logs/LRPTS20211230T18420779backup/"):
    SUBMIT=[]
    for dirsi in range(5):
        TSdir=os.path.join(LRPTSDIR,'TS_of_loop{}'.format(dirsi))
        if os.path.isdir(TSdir):
            print(TSdir +'  searching')
            for stuid in range(5):
                for name in os.listdir(TSdir):
                    if 'submit' in name and 'student_num_{}'.format(stuid) in name:
                        SUBMIT.append(os.path.join(TSdir, name))
        # for name in dirs :
        #     if 'submit' in name and 'student_num' in name:
        #         SUBMIT.append(os.path.join(root, name))
    print(len(SUBMIT))
    print(SUBMIT)
    return SUBMIT
def print_filename_metreics(filename,submits):
    for submit in submits:
        txtname = submit[:submit.find('LRPTS')]+submit[submit.find('LRPTS'):].replace('/','_')+'.txt'
        f=open(txtname,'r')
        all=f.read()
        need=all[all.find(filename):all.find(filename)+222]
        print(need)
        dice=need[need.find('DICE:')+5:need.find('DICE:')+11]
        print(dice)
def print_ALL_with_metreics(submittwo,submits):
    fid=-1
    for filename in os.listdir(submits[0]):
        f_dices=[]
        INCH=20
        H=14
        fig = plt.gcf()
        fig.set_size_inches(INCH, 2*INCH)

        for submit in submittwo:
            txtname = submit[:submit.find('LRPTS')]+submit[submit.find('LRPTS'):].replace('/','_')+'.txt'
            f=open(txtname,'r')
            all=f.read()
            need=all[all.find(filename):all.find(filename)+222]
            #print(need)
            dice=need[need.find('DICE:')+5:need.find('F1:')-1]
            dice=dice[:dice.find('.')+4]
            f_dices.append(float(dice))
        if f_dices[1]-f_dices[0]>0.02 and f_dices[2]-f_dices[1]>0.02 and f_dices[3]-f_dices[2]>0.015 and f_dices[4]< f_dices[3]\
                and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
            print(f_dices)
            print(filename)
            fid += 1
            if fid >14:
                fid=14
            idx=2
            for submit in submits:
                idx += 1
                im = io.imread(os.path.join(submit,filename))
                plt.subplot(H,7,idx+fid*7)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.imshow(im)
                plt.title("Dice={}".format(f_dices[idx-3]), y=-0.15)
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
    TITLE='abcdefg'
    for ti in range(7):
        plt.subplot(H,7,ti+1+14)
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.title('({})'.format(TITLE[ti]),y=0.9)
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig('TOSHOW/ALL.png')
    ALL=io.imread('TOSHOW/ALL.png')
    io.imsave('TOSHOW/ALL2.png',ALL[450:950,245:1800,:])
def print_LIVEcell():
    submits=[
        '/data1/wyj/M/samples/nucleus/.._.._logs_RESULTIMGS_coco20220114T1738_livecell_fullsup_mask_rcnn_coco_0016.h5/']




if __name__ == '__main__':
    #test(r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\submit_c16bh_1')
    p=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\submit_20211019T151422'
    baseline=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results/nucleus/submit_20210112T103916'
    put=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\put'
    iteration_n=r'D:\GT10\iteration7'
    sample=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\submit_20211020T121413'
    stage=os.path.abspath(r'../../datasets/MoNuSAC/stage1_train')
    GT_stage=os.path.abspath(r'../../datasets/MoNuSACGT/stage1_train')
    IMAGE1000=r'D:\BaiduNetdiskDownload\BrestCancer\MoNuSAC-master\data\MoNuSeg Training Data\Tissue Images'
    im250=r'D:\BaiduNetdiskDownload\BrestCancer\MoNuSAC-master\im250'
    colored=r'D:\BaiduNetdiskDownload\BrestCancer\MoNuSAC-master\colored_GT'
    paper=r'D:\BaiduNetdiskDownload\BrestCancer\PAPER'
    paper2 = r'D:\BaiduNetdiskDownload\BrestCancer\PAPER2'
    backup=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\datasets\MoNuSAC - 副本\stage1_train'
    this=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\datasets\MoNuSAC-ori\stage1_train_iteration_9'#0.724
    back=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\datasets\MoNuSAC\stage1_train_DATE1022_iteration01234567_GT10-based'
    loop5=r'E:\BaiduNetdiskDownload\最佳实验\Loop5_TS'
    loop4 = r'E:\BaiduNetdiskDownload\最佳实验\Loop4_TS'
    submit1=["/data1/wyj/M/logs/LRPTS20211230T18420779backup/TS_of_loop0/submit_20211230T190219_student_num_0/",
             "/data1/wyj/M/logs/LRPTS20211230T18420779backup/TS_of_loop0/submit_20211230T190422_student_num_1/",
             "/data1/wyj/M/logs/LRPTS20211230T18420779backup/TS_of_loop0/submit_20211230T190617_student_num_2/",
             "/data1/wyj/M/logs/LRPTS20211230T18420779backup/TS_of_loop0/submit_20211230T190807_student_num_3/",
             "/data1/wyj/M/logs/LRPTS20211230T18420779backup/TS_of_loop0/submit_20211230T190948_student_num_4/"
             ]
    submit2=[
             "/data1/wyj/M/logs/LRPTS20211230T18420779backup/TS_of_loop0/Student_num_2",
             "/data1/wyj/M/logs/LRPTS20211230T18420779backup/TS_of_loop1/Student_num_2",
             "/data1/wyj/M/logs/LRPTS20211230T18420779backup/TS_of_loop2/Student_num_3",
             "/data1/wyj/M/logs/LRPTS20211230T18420779backup/TS_of_loop3/Student_num_3",
             "/data1/wyj/M/logs/LRPTS20211230T18420779backup/TS_of_loop4/Student_num_3",

             ]
    submit3=[
             "/data1/wyj/M/logs/LRPTS20211230T18420779backup/TS_of_loop0/Colored_20211230T190617_student_num_2/",
             "/data1/wyj/M/logs/LRPTS20211230T18420779backup/TS_of_loop1/Colored_20211230T194037_student_num_2/",
             "/data1/wyj/M/logs/LRPTS20211230T18420779backup/TS_of_loop2/Colored_20211230T202110_student_num_3/",
             "/data1/wyj/M/logs/LRPTS20211230T18420779backup/TS_of_loop3/Colored_20211230T210514_student_num_3/",
             "/data1/wyj/M/logs/LRPTS20211230T18420779backup/TS_of_loop4/Colored_20220116T152043_student_num_3/",

             ]
    import matplotlib
    matplotlib.use('Agg')
    print_ALL_with_metreics(submit2,submit3)
    #print_LIVEcell()
    # submit1=gen_submits_from_LRPTSDIR()

    # bijective_coloured_visual(submit1)
    #
    # files = []
    # for fname in os.listdir(instance_GT_dir_):
    #     temp_masks = os.path.join(instance_GT_dir_, fname + '/masks/')
    #     # print(temp_masks + '     {}'.format(len(os.listdir(temp_masks))))
    #     files.append(len(os.listdir(temp_masks)))
    # print('mean files : {}'.format(np.mean(files)))
    # files = []
    # for fname in os.listdir(instance_GT_dir_)[:386]:
    #     temp_masks = os.path.join(instance_GT_dir_, fname + '/masks/')
    #     # print(temp_masks + '     {}'.format(len(os.listdir(temp_masks))))
    #     files.append(len(os.listdir(temp_masks)))
    # print('TRAIN mean files : {}'.format(np.mean(files)))
    # files = []
    # for fname in os.listdir(instance_GT_dir_)[386:]:
    #     temp_masks = os.path.join(instance_GT_dir_, fname + '/masks/')
    #     # print(temp_masks + '     {}'.format(len(os.listdir(temp_masks))))
    #     files.append(len(os.listdir(temp_masks)))
    # print('VAL mean files : {}'.format(np.mean(files)))

    # files=[]
    # for fname in os.listdir(instance_GT_dir_):
    #     temp_masks=os.path.join(instance_GT_dir_,fname+'/masks/')
    #     print(temp_masks+'     {}'.format(len(os.listdir(temp_masks))))
    #     files.append(len(os.listdir(temp_masks)))
    # print('mean files : {}'.format(np.mean(files)))
    # files=[]
    # for fname in os.listdir(instance_GT_dir_):
    #     if fname not in set(VAL_IMAGE_IDS):
    #         temp_masks=os.path.join(instance_GT_dir_,fname+'/masks/')
    #         print(temp_masks+'     {}'.format(len(os.listdir(temp_masks))))
    #         files.append(len(os.listdir(temp_masks)))
    # print('TRAIN mean files : {}'.format(np.mean(files)))
    # files=[]
    # for fname in os.listdir(instance_GT_dir_):
    #     if fname in set(VAL_IMAGE_IDS):
    #         temp_masks=os.path.join(instance_GT_dir_,fname+'/masks/')
    #         print(temp_masks+'     {}'.format(len(os.listdir(temp_masks))))
    #         files.append(len(os.listdir(temp_masks)))
    # print('VAL mean files : {}'.format(np.mean(files)))

    # submit1=[r'E:\BaiduNetdiskDownload\best_experiment\Loop1_TS\student2',  #689
    #          r'E:\BaiduNetdiskDownload\best_experiment\Loop1_TS\student3',  # 716
    #
    #         r'E:\BaiduNetdiskDownload\best_experiment\Loop1_TS\teacher',  #649
    #          r'E:\BaiduNetdiskDownload\best_experiment\Loop1_TS\student1',  # 678
    #
    #
    #          r'E:\BaiduNetdiskDownload\best_experiment\Loop2_TS\teacher',
    #          r'E:\BaiduNetdiskDownload\best_experiment\Loop2_TS\student1',
    #          r'E:\BaiduNetdiskDownload\best_experiment\Loop2_TS\student2',
    #          r'E:\BaiduNetdiskDownload\best_experiment\Loop2_TS\student3',
    #          r'E:\BaiduNetdiskDownload\best_experiment\Loop3_TS\teacher',
    #          r'E:\BaiduNetdiskDownload\best_experiment\Loop3_TS\student1',
    #          r'E:\BaiduNetdiskDownload\best_experiment\Loop3_TS\student2',
    #          r'E:\BaiduNetdiskDownload\best_experiment\Loop3_TS\student3',
    #          r'E:\BaiduNetdiskDownload\best_experiment\Loop4_TS\teacher',
    #          r'E:\BaiduNetdiskDownload\best_experiment\Loop4_TS\student1',
    #          r'E:\BaiduNetdiskDownload\best_experiment\Loop4_TS\student2',
    #          r'E:\BaiduNetdiskDownload\best_experiment\Loop4_TS\student3',#775
    #          #r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\submit_c16bh_2'
    #          ]

    # fafaf=io.imread(r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\datasets\L\images\livecell_test_images\A172\A172_Phase_C7_1_00d00h00m_1.tif')
    # print(fafaf.shape())
    # test_4metric.test_XMetric(p)
    # shutil.copytree(submit1[0], r'E:\BaiduNetdiskDownload\best_experiment/'+os.path.basename(submit1[0]))
    # tar15=r'TCGA-18-5592-01Z-00-DX1_crop_15.png'
    # tar21=r'TCGA-21-5784-01Z-00-DX1_crop_0.png'


