import os,shutil,datetime
import warnings
import test
import random
warnings.filterwarnings("ignore")
if __name__ == '__main__':
    now = datetime.datetime.now()
    LRPTS_dirpath="../../logs/LRPTS{:%Y%m%dT%H%M}".format(now)

    for loop in range(5):
        os.system('python nucleus5_cover.py train --dataset=../../datasets/MyNP --subset=train --weights=0 --iteration={} --LRPTS_DIR={} --LOOP={}'.format(0,LRPTS_dirpath,loop))
        GT2=r'../../datasets/MoNuSACGT{}'.format(loop)
        try:
            shutil.rmtree(GT2)
        except:
            pass
        shutil.copytree(r'../../datasets/MoNuSACGT',GT2)
        # for fname in os.listdir(GT2+'/stage1_train/'):
        #     stage=GT2+'/stage1_train/'
        #     imsdirpath=stage+'/'+fname+'/masks/'
        #     imsdir=os.listdir(imsdirpath)
        #     length=len(imsdir)
        #     for im in imsdir:
        #         k=random.randint(0,100)
        #         if k>30+loop*10 :
        #             os.remove(imsdirpath+im)
        os.system('python nucleus5_cover_and_LRP_gen.py')
        for iter in range(0,1):
            os.system('python nucleus3_50.py train --dataset={} --subset=train --weights={} --iteration={} --LRPTS_DIR={} --LOOP={}'.format(GT2,'resnet5cover.h5',str(iter+1),LRPTS_dirpath,loop))
            os.system('python nucleus3_50.py detect --dataset=../../datasets/MoNuSACCROP --subset=stage1_train --weights=lasts --iteration={} --LRPTS_DIR={} --LOOP={}'.format(str(iter+1),LRPTS_dirpath,loop))
