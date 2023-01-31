import shutil
import numpy as np
from skimage import io ,measure
import matplotlib.image as mi
import matplotlib.pyplot as plt
import os
import replace_maskdir
# x=np.zeros([3,9])
# im=mi.imread('../mask/1_Region_1.bmp')
# img=np.array(im,dtype=np.uint8)
# print(im.shape)
# #plt.imshow(im)
# con=measure.label(im,connectivity=1)
# conp=measure.regionprops(con)
# masker=np.zeros(con.shape)#im
# masker=img
# masker[con!=1]=0
# masker[con==1]=255
# plt.imshow(im)
# #plt.imshow(masker)
# plt.show()
#print(con)
maskpath_='../mask-new/'
savdir_='../masks/'
rootdir_=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\datasets\MoNuSAC\stage1_train'
def seg_mask(maskpath=maskpath_,savdir=savdir_,iteration_num=0):
    try:
        shutil.rmtree(savdir)
    except:
        pass
    try:
        os.mkdir(savdir)
    except:
        pass
    for wholename in os.listdir(maskpath):
        filename = os.path.splitext(wholename)[0]
        im = mi.imread(os.path.join(maskpath,wholename))
        print(filename)
        con = measure.label(im, connectivity=1)
        conp = measure.regionprops(con)
        try:
            shutil.rmtree(savdir + filename)
        except:
            pass
        try:
            os.mkdir(savdir + '/'+filename)
        except:
            pass
        for i in range(1,len(conp)+1):
            masker = np.array(im,dtype=np.uint8)
            masker[con != i] = 0
            masker[con == i] = 255
            savname=savdir +'/'+ filename + '/' +filename + '_iteration_'+str(iteration_num)+'_mask_' +str(i)+'.png'
            io.imsave(savname,masker)
            #chenkingim=mi.imread(savname)
            # plt.imshow(chenkingim)
            # plt.show()
            # chenkingim=masker

    #os.system('python replace_maskdir.py')
    replace_maskdir.replace_maskdir(rootdir=rootdir_,savdir=savdir,iteration_num=iteration_num)

def seg_mask2(maskpath=maskpath_, savdir=savdir_, iteration_num=0):
    try:
        shutil.rmtree(savdir)
    except:
        pass
    try:
        os.mkdir(savdir)
    except:
        pass
    for wholename in os.listdir(maskpath):
        filename = os.path.splitext(wholename)[0]
        im = mi.imread(os.path.join(maskpath, wholename))
#        print(filename+'   seg_mask  running')
        con = measure.label(im, connectivity=1)
        conp = measure.regionprops(con)
        try:
            shutil.rmtree(savdir + filename)
        except:
            pass
        try:
            os.mkdir(savdir + '/' + filename)
        except:
            pass
        for i in range(1, len(conp) + 1):
            masker = np.array(im, dtype=np.uint8)
            masker[con != i] = 0
            masker[con == i] = 255
            savname = savdir + '/' + filename + '/' + filename + '_iteration_' + str(
                iteration_num) + '_mask_' + str(i) + '.png'
            io.imsave(savname, masker)
            # chenkingim=mi.imread(savname)
            # plt.imshow(chenkingim)
            # plt.show()
            # chenkingim=masker

    # os.system('python replace_maskdir.py')
    #replace_maskdir.replace_maskdir(rootdir=rootdir_, savdir=savdir, iteration_num=iteration_num)

if __name__ == '__main__':
    seg_mask(maskpath_,savdir_)
    #replace_maskdir.replace_maskdir(savdir=savdir_)