#coding:utf-8
import os,shutil,datetime
import warnings
import test
import random
warnings.filterwarnings("ignore")
if __name__ == '__main__':
    for repeat in range(1):
        now = datetime.datetime.now()
        LRPTS_dirpath="../../logs/LRPTS{:%Y%m%dT%H%M}".format(now)
        for loop in range(0,5):
            os.system('python classification_ccrcc.py train --dataset=/data1/wyj/M/datasets/ccrccNP/ --subset=Train --weights=0 --iteration={} --LRPTS_DIR={} --LOOP={}'.format(0,LRPTS_dirpath,loop))
            GTback=r'/data1/wyj/M/datasets/ccrcccrop_backup'
            GT2 = r'/data1/wyj/M/datasets/consepcrop'
            for iter in range(0, 1):
                os.system(
                    'python nucleus3_consep.py train --dataset={} --subset=Train --weights={} --iteration={} --LRPTS_DIR={} --LOOP={}'.format(
                        GT2, 'resnet5cover.h5', str(0), LRPTS_dirpath, loop))
                os.system(
                    'python nucleus3_consep.py detect --dataset={} --subset=Test --weights=lasts --iteration={} --LRPTS_DIR={} --LOOP={}'.format(
                        GT2 , str(0), LRPTS_dirpath, loop))
