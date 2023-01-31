import os, shutil, datetime
import warnings
import test
import random

warnings.filterwarnings("ignore")
if __name__ == '__main__':
    model = [
        r"/home/deeplearning/wyj/Mask_RCNN/logs/nucleus20211214T1704/mask_rcnn_nucleus_0007d791.h5",
        r'/home/deeplearning/wyj/Mask_RCNN/logs/nucleus20211104T0012_____0.724/mask_rcnn_nucleus_0001.h5',
        r'/home/deeplearning/wyj/Mask_RCNN/logs/nucleus20211104T0104/mask_rcnn_nucleus_0001.h5',
        r'/home/deeplearning/wyj/Mask_RCNN/logs/mask2.h5',
        r'/home/deeplearning/wyj/Mask_RCNN/logs/0744.h5',
        r'/home/deeplearning/wyj/Mask_RCNN/logs/LOOP2_final0719.h5',
        r'/home/deeplearning/wyj/Mask_RCNN/logs/0757.h5',
        r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\logs\nucleus20211023T0046\mask_rcnn_nucleus_0001.h5',
        r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\logs\nucleus20211023T0046\mask_rcnn_nucleus_0002.h5',
        r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\logs\nucleus20211023T0046\mask_rcnn_nucleus_0003.h5',
        r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\logs\nucleus20211023T0046\mask_rcnn_nucleus_0004.h5',

        r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\logs\nucleus20201217T2129\mask_rcnn_nucleus_0001.h5',
        r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\logs\nucleus20201220T0624\mask_rcnn_nucleus_0001.h5',
        # r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\logs\nucleus20210109T1242\mask_rcnn_nucleus_base.h5',
        # r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\logs\nucleus20210109T1242\mask_rcnn_nucleus_0001.h5',
        r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\logs\nucleus20210110T2141\mask_rcnn_nucleus_0000.h5',
        r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\logs\nucleus20210110T2141\mask_rcnn_nucleus_0001.h5',
        r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\logs\nucleus20210110T2141\mask_rcnn_nucleus_0002.h5',
        r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\logs\nucleus20210110T2141\mask_rcnn_nucleus_0003.h5',
        r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\logs\nucleus20210110T2141\mask_rcnn_nucleus_0004.h5',

    ]
    path = [
        r'resnet5cover.h5',
        r"/home/deeplearning/wyj/Mask_RCNN/logs/nucleus20211119T1634/742.h5",
        r"/home/deeplearning/wyj/Mask_RCNN/logs/nucleus20211116T2009/736.h5",
        r"/home/deeplearning/wyj/Mask_RCNN/logs/nucleus20211116T2104/716.h5",
        r"/home/deeplearning/wyj/Mask_RCNN/logs/nucleus20211116T2142/mask_rcnn_nucleus_0001.h5",
        r'/home/deeplearning/wyj/Mask_RCNN/logs/nucleus20211115T1501/mask_rcnn_nucleus_0002.h5',
        r"/home/deeplearning/wyj/Mask_RCNN/logs/nucleus20211110T2310/mask_rcnn_nucleus_0001.h5",
        r"/home/deeplearning/wyj/Mask_RCNN/logs/nucleus20211110T2310/mask_rcnn_nucleus_0002.h5",
        r"/home/deeplearning/wyj/Mask_RCNN/logs/nucleus20211110T2310/mask_rcnn_nucleus_0003.h5",
        r"/home/deeplearning/wyj/Mask_RCNN/logs/nucleus20211110T2310/mask_rcnn_nucleus_0004.h5",
        r"/home/deeplearning/wyj/Mask_RCNN/logs/nucleus20211110T2310/mask_rcnn_nucleus_0005.h5",

    ]
    # for i in range(len(model)):
    #     os.system(
    #         'python nucleus3.py detect --dataset=../../datasets/MoNuSACCROP --subset=stage1_train --weights={} --iteration='.format(model[i]) + str(
    #             0 + 1))
    # print('__________________________________________________next__________________________________________________')
    # os.system(
    #     'python nucleus4.py detect --dataset=../../datasets/MoNuSAC --subset=stage1_train --weights={} --iteration='.format(model) + str(
    #         0 + 1))
    # os.system(
    #     'python nucleus4.py detect --dataset=../../datasets/MoNuSAC --subset=stage1_train --weights=last --iteration=' + str(
    #         0 + 1))
    # os.system(
    #     r'python nucleus3.py detect --dataset=../../datasets/MoNuSACCROP --subset=stage1_train --weights=D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\logs\nucleus20201218T0630!!!\mask_rcnn_nucleus_0001.h5 --iteration=' + str(
    #         0 + 1))
    # os.system(
    #     r'python nucleus4.py detect --dataset=../../datasets/MoNuSAC --subset=stage1_train --weights=D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\logs\nucleus20201218T0630!!!\mask_rcnn_nucleus_0001.h5 --iteration=' + str(
    #         0 + 1))
    # shutil.rmtree(r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\datasets\MoNuSAC\stage1_train')
    # shutil.copytree(r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\datasets\MoNuSAC\stage1_train0000',r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\datasets\MoNuSAC\stage1_train')
    # os.system(
    #     'python nucleus3.py train --dataset=../../datasets/MoNuSAC --subset=train --weights={} --iteration='.format(
    #         model[2]) + str(0 + 1))
    # for iter in range(0, 7):
    #     os.system(
    #         'python nucleus3.py detect --dataset=../../datasets/MoNuSACCROP --subset=stage1_train --weights={} --iteration='.format(
    #             model[iter]) + str(0 + 1))
    # for iter in range(6,7):
    #     os.system(
    #         'python nucleus4.py detect --dataset=../../datasets/MoNuSAC --subset=stage1_train --weights={} --iteration='.format(model[iter]) + str(
    #             iter + 1))
    # for iter in range(0,8):
    #     os.system('python nucleus4.py detect --dataset=../../datasets/MoNuSAC --subset=stage1_train --weights={} --iteration='.format(model[iter]) + str(iter + 1))

    # for iter in range(0,1):
    #     os.system('python nucleus3.py detect --dataset=../../datasets/MoNuSACCROP --subset=stage1_train --weights=last --iteration='+ str(iter + 1))
    #     os.system('python nucleus4.py detect --dataset=../../datasets/MoNuSAC --subset=stage1_train --weights=last --iteration=' + str(iter + 1))
    #
    #     os.system('python inspect_nucleus_model_by_epoches.py')
    #     os.system('python refinement.py --iteration_num='+str(iter+1))

    # os.system('python example-resnet.py')
    # os.system('python nucleus5_cover.py train --dataset=../../datasets/MyNP --subset=train --weights=0 --iteration=0')

    # os.system('python test.py')
    # for iter in range(0,1):
    # os.system('python nucleus3.py train --dataset=../../datasets/MoNuSAC --subset=train --weights={} --iteration='.format(model[0])+str(iter+1))
    # os.system('python nucleus3.py detect --dataset=../../datasets/MoNuSACCROP --subset=stage1_train --weights={} --iteration='.format(model[1]) + str(iter + 1))
    # os.system('python nucleus6.py detect --dataset=../../datasets/MoNuSACCROP --subset=stage1_train --weights={} --iteration='.format(model[5]) + str(0 + 1))
    # os.system('python inspect_nucleus_model_by_epoches.py')
    # os.system('python refinement.py --iteration_num='+str(iter+1))

    # for iter in range(0,3):
    #     os.system('python nucleus3.py train --dataset=../../datasets/MoNuSAC --subset=train --weights=coco --iteration='.format(model[5])+str(iter+1))
    # os.system('python nucleus3.py detect --dataset=../../datasets/MoNuSACCROP --subset=stage1_train --weights=lasts --iteration=' + str(0 + 1))
    #     os.system('python nucleus6.py detect --dataset=../../datasets/MoNuSACCROP --subset=stage1_train --weights=last --iteration=' + str(iter + 1))
    #     os.system('python inspect_nucleus_model_by_epoches.py')
    #     os.system('python refinement.py --iteration_num='+str(iter+1))
    # test.testGT_onto_dice(r'../../datasets/MoNuSAC/stage1_train/',iteration_flag=True,iteration=2)
    # for iter in range(0,1):
    # os.system('python nucleus3.py train --dataset=../../datasets/MoNuSAC --subset=train --weights=coco --iteration='.format('resnet5cover.h5')+str(iter+1))
    # os.system('python nucleus3.py detect --dataset=../../datasets/MoNuSACCROP --subset=stage1_train --weights={} --iteration='.format(path[iter]) + str(iter + 1))
    # os.system('python nucleus6.py detect --dataset=../../datasets/MoNuSACCROP --subset=stage1_train --weights={} --iteration='.format(path[1]) + str(iter + 1))
    # os.system('python inspect_nucleus_model_by_epoches.py')
    # os.system('python refinement.py --iteration_num='+str(iter+1))


    for repeat in range(1):
        now = datetime.datetime.now()
        LRPTS_dirpath = "../../logs/LRPTS{:%Y%m%dT%H%M}".format(now)
        # LRPTS_dirpath='/data1/wyj/M/logs/LRPTS20220119T1634/'
        print('!!!!!!' + LRPTS_dirpath + '!!!!!!!!')
        #LRPTS_dirpath="../../logs/LRPTS20211230T1842!!!!0779".format(now)
        for loop in range(0,1):
            # os.system('python nucleus5_covercoco.py train --dataset=../../datasets/L/images/LNP/ --subset=train --weights=0 --iteration={} --LRPTS_DIR={} --LOOP={}'.format(0,LRPTS_dirpath,loop))
            L=r'../../datasets/ccrcc/'
            GT2 = r'../../datasets/ccrcc/'.format(loop)
            for iter in range(0, 1):
                os.system(
                    'python nucleus3_ccrcc.py train --dataset={} --subset=Train --weights={} --iteration={} --LRPTS_DIR={} --LOOP={}'.format(
                        GT2, 'coco', str(1), LRPTS_dirpath, loop))
                os.system(
                    'python nucleus3_ccrcc.py detect --dataset={} --subset=Train --weights={} --iteration={} --LRPTS_DIR={} --LOOP={}'.format(
                        GT2, 'coco', str(1), LRPTS_dirpath, loop))


    # for loop in range(1):
    #     L=r'../../datasets/L/livecell/'
    #     GT2 = r'../../datasets/L/LIVE2/'
    #     # try:
    #     #     shutil.rmtree(GT2)
    #     # except:
    #     #     pass
    #     # try:
    #     #     os.mkdir(GT2)
    #     #     os.mkdir(GT2+'stage1_train/')
    #     # except:
    #     #     pass
    #     # for fname in os.listdir(L+'/stage1_train/'):
    #     #     if fname.startswith('BV2'):
    #     #         shutil.copytree(L+'/stage1_train/'+fname,GT2+'/stage1_train/'+fname)
    #     for iter in range(0, 1):
    #         os.system(
    #             'python nucleus3reedit.py train --dataset={} --subset=train --weights={} --iteration={} --LRPTS_DIR={} --LOOP={}'.format(
    #                 GT2, 'last', str(0), LRPTS_dirpath, loop))
    #         os.chdir("../coco")
    #         os.system('python ../coco/coco.py evaluate --model=lasts')
    #         os.chdir("../nucleus")
    # testh5 = r"/home/iftwo/wyj/M/logs/LRPTS20211230T1842!!!!0779/TS_of_loop3/Student_num_3"
    # os.system('python nucleus3metric.py detect --dataset=../../datasets/MoNuSACCROP --subset=stage1_train --weights={} --iteration={}'.format(testh5,0))
    # for iter in range(1,2):
    # os.system('python nucleus3.py train --dataset=../../datasets/MoNuSACGT3 --subset=train --weights={} --iteration='.format(model[0])+str(iter+1))
    # os.system('python nucleus3.py detect --dataset=../../datasets/MoNuSACCROP --subset=stage1_train --weights=lasts --iteration='.format(path[iter]) + str(iter + 1))
    # os.system('python nucleus6.py detect --dataset=../../datasets/MoNuSACCROP --subset=stage1_train --weights=last --iteration=' + str(iter + 1))
    # os.system('python inspect_nucleus_model_by_epoches.py')
    # os.system('python refinement.py --iteration_num='+str(iter+1))

# #os.system('python nucleus.py detect --dataset=../../datasets/MoNuSAC --subset=train --weights=D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\logs\default20200930T0804\mask_rcnn_default_0002.h5 --iteration=' + str(1 + 1))
