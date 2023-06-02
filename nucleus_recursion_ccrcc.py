import os, shutil, datetime
import warnings
import test
import random

warnings.filterwarnings("ignore")
if __name__ == '__main__':
    for repeat in range(1):
        now = datetime.datetime.now()
        LRPTS_dirpath = "../../logs/LRPTS{:%Y%m%dT%H%M}".format(now)
        # LRPTS_dirpath='/data1/wyj/M/logs/LRPTS20220119T1634/'
        print('!!!!!!' + LRPTS_dirpath + '!!!!!!!!')
        #LRPTS_dirpath="../../logs/LRPTS20211230T1842!!!!0779".format(now)
        for loop in range(0,1):
            os.system('python nucleus5_covercoco.py train --dataset=../../datasets/L/images/LNP/ --subset=train --weights=0 --iteration={} --LRPTS_DIR={} --LOOP={}'.format(0,LRPTS_dirpath,loop))
            L=r'../../datasets/ccrcc/'
            GT2 = r'../../datasets/ccrcc/'.format(loop)
            for iter in range(0, 1):
                os.system(
                    'python nucleus3_ccrcc.py train --dataset={} --subset=Train --weights={} --iteration={} --LRPTS_DIR={} --LOOP={}'.format(
                        GT2, 'coco', str(1), LRPTS_dirpath, loop))
                os.system(
                    'python nucleus3_ccrcc.py detect --dataset={} --subset=Train --weights={} --iteration={} --LRPTS_DIR={} --LOOP={}'.format(
                        GT2, 'coco', str(1), LRPTS_dirpath, loop))