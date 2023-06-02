import shutil
import numpy as np
from skimage import io ,measure
import matplotlib.image as mi
import matplotlib.pyplot as plt
import os
from skimage.color import rgb2gray
from skimage.transform import resize
#import test
import warnings
from collections import Counter
warnings.filterwarnings("ignore")
def dice(a, b):
    """dice coefficient 2nt/na + nb."""
    overlap = a * b
    I = overlap.sum() * 2
    U = (a.sum() + b.sum())

    return I/U
def TP(gt, mask):
    """dice coefficient 2nt/na + nb."""
    overlap = gt * mask
    I = overlap.sum()
    U = mask.sum()
    return I/U
def Recall(gt, mask):
    """dice coefficient 2nt/na + nb."""
    return TP(mask,gt)
# Dirpath=r'D:\BaiduNetdiskDownload\BrestCancer\softmaxgradient-lrp-master\vgg\Work'
# lrp=os.listdir(Dirpath+'/LRP1')
# lrpa=os.listdir(Dirpath+'/LRPA1')
# im=os.listdir(Dirpath+'/im')
# gt=os.listdir(Dirpath+'/gt')
# gbp=os.listdir(Dirpath+'/GBP1')
# sglrp=os.listdir(Dirpath+'/SGLRPA1')
# countim=0
# picno_=0
# for id in range(len(lrp)):
#     LRP=io.imread(Dirpath+'/LRP1/'+lrp[id])
#     if np.min(LRP)<250:
#         plt.subplot(5,5,5*countim+2)
#         plt.imshow(LRP)
#         plt.xticks([])
#         plt.yticks([])
#         plt.subplot(5, 5, 5 * countim + 1)
#         GBP=io.imread(Dirpath+'/GBP1/'+gbp[id])
#         plt.imshow(GBP)
#         plt.xticks([])
#         plt.yticks([])
#         plt.subplot(5, 5, 5 * countim + 3)
#         LRPA=io.imread(Dirpath+'/LRPA1/'+lrpa[id])
#         plt.imshow(LRPA)
#         plt.xticks([])
#         plt.yticks([])
#         plt.subplot(5, 5, 5 * countim + 4)
#         GT=io.imread(Dirpath+'/gt/'+gt[id])
#         plt.imshow(GT)
#         plt.xticks([])
#         plt.yticks([])
#         plt.subplot(5, 5, 5 * countim + 5)
#         IM=io.imread(Dirpath+'/im/'+im[id])
#         plt.imshow(IM)
#         plt.xticks([])
#         plt.yticks([])
#         countim=countim+1
#         if countim==5:
#             countim=0
#             picno_=picno_+1
#             plt.axis('off')
#             plt.xticks([])
#             plt.yticks([])
#             plt.gca().xaxis.set_major_locator(plt.NullLocator())
#             plt.gca().yaxis.set_major_locator(plt.NullLocator())
#             plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
#             plt.margins(0,0)
#             plt.savefig('compare/{}.png'.format(picno_),pad_inches = 0)
#             plt.close()
#         #plt.show()

def fix(LLLdir=''):
    colours=[0.4,0.2,0.5,0.38,0.2,0.4,0.2,0.4,0.3,0.2,0.2,0.25,0.3,0.25,0.32,0.2,0.45,0.5,0.25,0.45,0.4,0.35,0.4,0.45,0.3,0.25,0.3,0.2,0.19,0.2]
    # print(len(colours))
    SAVPATH =os.path.join(LLLdir,'SEEDMAPS')
    if not os.path.exists(SAVPATH):
        os.mkdir(SAVPATH)
    current_ids=[[37,14,18,39,48,71,75,102,153,257,229,197,275,131,271,101,181],
                 [218,104,34,239,212,200,44,60,207,185,111,184,230,43,160,192,180],
                 [80,101,26,11,128,57,90,52,116,23],
                 [85,87,121,90,79,117,65,36,21,123,112,68,109,14,18],
                 [15,8,29,30,36,31,33,35,10,8,12],
                 [120,132,134,98,104,128,46,57,58,29,54,12],
                 [94,37,38,45,70,68,36,30,31,63,218,230,156,143,199,166,239,240],
                 [22,21,31,28,61,78,87,101,112,123,122,130,132,148,165,172,124],
                 [37,41,44,22,20,25,38,44,74,104,108,138,124,137,144],
                 [163,43,24,74,87,105,145,176,198,258,246,260,277,257],
                 [43,40,52,59,86,141,171,216,190,267,301,338,219],
                 [50,48,62,150,178,247,288,278,243,289,217],
                 [54,81,152,251,297,345,318,342,120,266,257],
                 [249,60,90,172,157,228,233,227,188,195,313,401,426,409,459,415,58,62],
                 [46,42,61,60,55,125,131,209,241,239,235,264,255],
                 [4,9,7,12,22,19,],
                 [109,128,142,153,219,253,293,182,209,123],
                 [125,99,277,262,245,291,185,146,136,139,271],
                 [38,86,74,94,91,41,38,11,57,58],
                 [11,10,15,26,19,45,43,40,29],
                 [111,46,97,55,114],
                 [64,41,36,69,70,39,65,58,47,130,172,212,248,258,233,253,269],
                 [51,50,65,44,62,114,123,143,163,171,32,53,110],
                 [47,44,56,77,80,21],
                 [27,35,43,140,153,189,87,194,212,79,177],
                 [61,73,76,70,80,90,71,79,35,23],
                 [79,74,104,170,283,293,298,460,467,438,272,269,234,505,483,126,132,323],
                 [280,231,260,293,235,177,285,82,123,171,174,188,110],
                 [17,21,14,77,80,87,69,53,95,93,48,59],
                 [119,112,74,48,82,116,117,128,223,253,208,110]
                 ]
    # print(len(current_ids))
    order=np.array([0,1,10,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,3,4,5,6,7,8,9])
    # b = np.where(order==4)
    # print(b[0][0])
    # order2=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    # order3=[0,1,1]
    # imgdir=r'D:\BaiduNetdiskDownload\BrestCancer\MoNuSAC-master\data\MoNuSeg Training Data\Tissue Images'
    #gtp=r'D:\BaiduNetdiskDownload\BrestCancer\softmaxgradient-lrp-master\vgg\Work\gt\21_8.png'
    gtp=r'../../datasets/MoNuSAC_mask/'
    # lrpim_path=r'D:\BaiduNetdiskDownload\BrestCancer\softmaxgradient-lrp-master\vgg\Work\LRP1\S21_LRP_8.jpg'
    #lrpim_path=r'D:\BaiduNetdiskDownload\BrestCancer\softmaxgradient-lrp-master\Mapping_GT'
    #mappath=r'D:\BaiduNetdiskDownload\BrestCancer\softmaxgradient-lrp-master\Mapping_GT'
    idd=0
    stage = os.path.abspath(r'../../datasets/MoNuSAC/stage1_train')
    LRPMAPS='LRPmaps'
    seed_path=r'../../datasets/MoNuSAC/seed_visual/'
    result=r'../../results\nucleus\submit_20201219T025258'
    VISUALIZE=True
    mean_area=600
    STAGE_NAMES=['TCGA-18-5592-01Z-00-DX1', 'TCGA-21-5784-01Z-00-DX1', 'TCGA-21-5786-01Z-00-DX1', 'TCGA-38-6178-01Z-00-DX1',
                 'TCGA-49-4488-01Z-00-DX1', 'TCGA-50-5931-01Z-00-DX1', 'TCGA-A7-A13E-01Z-00-DX1', 'TCGA-A7-A13F-01Z-00-DX1',
                 'TCGA-AR-A1AK-01Z-00-DX1', 'TCGA-AR-A1AS-01Z-00-DX1', 'TCGA-AY-A8YK-01A-01-TS1', 'TCGA-B0-5698-01Z-00-DX1',
                 'TCGA-B0-5710-01Z-00-DX1', 'TCGA-B0-5711-01Z-00-DX1', 'TCGA-CH-5767-01Z-00-DX1', 'TCGA-DK-A2I6-01A-01-TS1',
                 'TCGA-E2-A14V-01Z-00-DX1', 'TCGA-E2-A1B5-01Z-00-DX1', 'TCGA-G2-A2EK-01A-02-TSB', 'TCGA-G9-6336-01Z-00-DX1',
                 'TCGA-G9-6348-01Z-00-DX1', 'TCGA-G9-6356-01Z-00-DX1', 'TCGA-G9-6362-01Z-00-DX1', 'TCGA-G9-6363-01Z-00-DX1',
                 'TCGA-HE-7128-01Z-00-DX1', 'TCGA-HE-7129-01Z-00-DX1', 'TCGA-HE-7130-01Z-00-DX1', 'TCGA-KB-A93J-01A-01-TS1',
                 'TCGA-NH-A8F7-01A-01-TS1', 'TCGA-RD-A8N9-01A-01-TS1']
    for filename in os.listdir(stage):
        for file in os.listdir(os.path.join(stage, filename + '/masks')):
            os.remove(os.path.join(stage, filename + '/masks/' + file))
    for nam in STAGE_NAMES:
        AREA_WEIGHT=0
        print(nam)
        #nam = r'TCGA-HE-7129-01Z-00-DX1'
        if nam==r'TCGA-HE-7129-01Z-00-DX1' or nam=='TCGA-HE-7128-01Z-00-DX1' :
            AREA_WEIGHT=50
            mean_area=200
        else:
            AREA_WEIGHT=0
            mean_area=600
        name=nam+'.png'
        lrpimc=io.imread(stage +'/'+nam+'/images/'+name)
        oriim=io.imread(stage +'/'+nam+'/images/'+name)
        #res=io.imread(result +'/'+name)
        # plt.subplot(1, 3, 2)
        # plt.imshow(lrpimc)
        # plt.subplot(1, 3, 1)
        # plt.imshow(oriim)
        # plt.subplot(1, 3, 3)
        # plt.imshow(res)
        # plt.show()
        # ddir = r'D:\BaiduNetdiskDownload\BrestCancer\softmaxgradient-lrp-master\vgg\Work\compare_4/'
        #plt.savefig(ddir + name , bbox_inches='tight', dpi=900)
        # lrpimc=np.array(resize(lrpim,(224,224)))
        # plt.imshow(lrpimc)
        # plt.show()
        gt=np.array(io.imread(gtp+'/'+name[:-4]+'_mask.png'))
        gray=rgb2gray(oriim)
        # plt.imshow(gray)
        # plt.show()
        # print(np.mean(gray))
        final_mask=np.zeros(lrpimc.shape[0:2],dtype=np.int16)
        f_step3 = np.ones(lrpimc.shape[0:2], dtype=np.int16)
        final_responding_gt=np.zeros(lrpimc.shape[0:2])
        mask = np.zeros(lrpimc.shape[0:2])
        blank_result = np.zeros(lrpimc.shape[0:2])
        mask88 = np.zeros(lrpimc.shape[0:2])
        max_convex_map = np.zeros(lrpimc.shape[0:2])
        gtmaskblank_result = np.zeros(lrpimc.shape[0:2])
        gt_respoding_instance = np.zeros(lrpimc.shape[0:2],dtype=np.int16)
        double_contour_map = np.zeros(lrpimc.shape[0:2])

        threshold = colours[idd]#(np.mean(gray))
        if idd==0:
            threshold = 0.35
        mask[gray <= threshold] = 255
        con = measure.label(mask, connectivity=1)
        conp = measure.regionprops(con)
        gtcon = measure.label(gt, connectivity=1)
        gtconp = measure.regionprops(gtcon)
        for conpi in conp:
            if conpi.area < 100:
                con[con == conpi.label] = 0
        con_sq = measure.label(con)
        con_sqp=measure.regionprops(con_sq)
        # plt.imshow(con_sq)
        # plt.show()
        score=np.zeros(len(con_sqp))
        f_step3=f_step3*255

        for i in range(len(con_sqp)) :
            convex = con_sqp[i - 1].convex_image
            [x1, y1, x2, y2] = con_sqp[i - 1].bbox
            use = con[x1:x2, y1:y2]
            usef_step3 = f_step3[x1:x2, y1:y2]
            usef_step3[convex == True] = 0
            usef_step3[use!=0] = 255
            use[convex == True] = 255
            convex_rate=con_sqp[i].filled_area / con_sqp[i].convex_area
            if convex_rate > 0.5:
                weight=(convex_rate - 0.5)
            else:weight=0
            [x1, y1, x2, y2] = con_sqp[i].bbox
            X=x2-x1
            Y=y2-y1
            #score[i] = 300 * convex_rate - AREA_WEIGHT * abs(con_sqp[i].convex_area - 600) - 200 * weight * (((X-Y)/(X+Y))*((X-Y)/(X+Y))) #L3 0789
            score[i] = 300 * convex_rate - AREA_WEIGHT*abs(con_sqp[i].convex_area - mean_area) #- 200 * weight * (((X-Y)/(X+Y))*((X-Y)/(X+Y)))
            if nam == r'TCGA-HE-7129-01Z-00-DX1' or nam == 'TCGA-HE-7128-01Z-00-DX1' :
                pass
            else:
                if con_sqp[i].area<200 or con_sqp[i].area>1300:
                    score[i]=score[i]-1000
                if con_sqp[i].area <= 150 or con_sqp[i].area >= 2000:
                    score[i] = score[i] - 1000000

            if x1==0 or y1==0 or x2>=lrpimc.shape[0]-1 or y2>=lrpimc.shape[1]-1:
                score[i] =score[i]-10000
        rank=np.argsort(score)
        LENGTH=20
        IDS=np.zeros(LENGTH+1,dtype=np.int)
        for i in range(LENGTH+1):
            IDS[i]=con_sqp[rank[-i]].label

        # plt.imshow(con_sq)
        # plt.show()
        cid=0
        for current_id in IDS:
            # gtmaskblank_result[gtcon == 9] = 255

            convex=con_sqp[current_id-1].convex_image
            if con_sqp[current_id-1].area<2000 and cid<LENGTH:
                blank= np.zeros(lrpimc.shape[0:2])
                [x1,y1,x2,y2]=con_sqp[current_id-1].bbox
                use = blank[x1:x2, y1:y2]
                use[convex == True] = 255
                io.imsave(r'../../datasets/MoNuSAC/stage1_train/{}'.format(name[:-4])+'/masks/'+'{}.png'.format(name[:-4]+'_'+str(current_id)),blank.astype(np.uint8))
                cid = cid + 1
                use=final_mask[x1:x2,y1:y2]
                use[convex==True]=cid

                # contours=measure.find_contours(max_convex_map,254)
                # double_contour_map[con==current_id]=255
                # # # mask[]
                # gt_responding_value=np.max(gtcon[x1:x2,y1:y2])
                # V=Counter(gtcon[x1:x2,y1:y2]).most_common(1)[0][0]
                # if gt_responding_value!=0:
                #     final_responding_gt[V]=255
        # print('Recall: {}'.format(Recall(final_responding_gt != 0, final_mask != 0)))
        # print('Precise : {}'.format(TP(final_responding_gt != 0, final_mask != 0)))
        # print('dice : {}'.format(dice(final_responding_gt != 0, final_mask != 0)))
        # io.imsave('D:\BaiduNetdiskDownload\BrestCancer\softmaxgradient-lrp-master\MASK/{}'.format(name),final_mask.astype(np.uint8))
        # io.imsave('D:\BaiduNetdiskDownload\BrestCancer\softmaxgradient-lrp-master\Mapping_GT/{}'.format(name), final_responding_gt.astype(np.uint8))
        maper = np.array(final_mask)
        maper[gt != 0]=100
        maper[final_mask!=0]=maper[final_mask!=0]+200
        contours = measure.find_contours(con, 1)
        if VISUALIZE:
            plt.subplot(1, 4, 2)
            plt.imshow(final_mask)
            plt.subplot(1, 4, 1)
            plt.imshow(maper)
            plt.subplot(1, 4, 3)
            plt.imshow(mask)
            plt.subplot(1, 4, 4)
            plt.imshow(oriim)
            # plt.savefig(seed_path + name , bbox_inches='tight', dpi=900)
            plt.show()
        for cropid in range(16):
            cx=cropid%4
            cy=cropid//4
            mask_crop=final_mask[cx*250:cx*250+250,cy*250:cy*250+250]
            mid_crop=mask[cx*250:cx*250+250,cy*250:cy*250+250]
            step2=con[cx*250:cx*250+250,cy*250:cy*250+250]
            step3 = f_step3[cx * 250:cx * 250 + 250, cy * 250:cy * 250 + 250]
            io.imsave(SAVPATH+'/{}_crop_{}_all_seeds_step4.png'.format(nam,cropid),mask_crop)

            plt.close()
            # plt.figure(figsize=(10,10),dpi=100)
            fig=plt.gcf()
            fig.set_size_inches(10,10)
            plt.imshow(mask,plt.cm.gray)
            for n,contour in enumerate(contours):
                plt.plot(contour[:,1],contour[:,0],linewidth=2)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.axis('off')
            plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)
            plt.margins(0,0)
            plt.savefig('temp.png')
            # plt.show()

            imstep=io.imread('temp.png')
            print(imstep.shape)
            step3=imstep[cx*250:cx*250+250,cy*250:cy*250+250]
            plt.close()
            # plt.imshow(step3)
            # plt.show()
            io.imsave(SAVPATH + '/{}_crop_{}_all_seeds_step2.png'.format(nam, cropid), step3)
            io.imsave(SAVPATH + '/{}_crop_{}_all_seeds_step3.png'.format(nam, cropid), step2)
            io.imsave(SAVPATH + '/{}_crop_{}_all_seeds_step1.png'.format(nam, cropid), mid_crop)
        idd=idd+1
    #test.testGT_onto_dice(stage,iteration_flag=False)

if __name__ == '__main__':
    fix()
        # for ix in range(5):
        #     for iy in range(5):
        #         trylrp=io.imread(r'D:\BaiduNetdiskDownload\BrestCancer\softmaxgradient-lrp-master\vgg\Work\LRP1' + '/S' + str(
        #             idd) + '_LRP_' + str(np.where(order == 5 * ix + iy)[0][0]) + '.jpg')
        #         if np.max(trylrp)!=0:
        #             plt.subplot(1,3,1)
        #             plt.imshow(io.imread(r'D:\BaiduNetdiskDownload\BrestCancer\softmaxgradient-lrp-master\vgg\Work\im'+'/'+str(idd)+'_'+str(np.where(order==5*ix+iy)[0][0])+'.png'))
        #             plt.subplot(1,3,2)
        #             plt.imshow(io.imread(r'D:\BaiduNetdiskDownload\BrestCancer\softmaxgradient-lrp-master\vgg\Work\LRP1'+'/S'+str(idd)+'_LRP_'+str(np.where(order==5*ix+iy)[0][0])+'.jpg'))
        #             print(order[5 * ix + iy])
        #             plt.subplot(1,3,3)
        #             plt.imshow(lrpimc[iy*194:iy*194+224,ix*194:ix*194+224])
        #             ddir=r'D:\BaiduNetdiskDownload\BrestCancer\softmaxgradient-lrp-master\vgg\Work\compare_3/'
        #             plt.savefig(ddir+name+'{}_{}.png'.format(ix,iy),bbox_inches='tight',dpi=900)
        #             # plt.show()

        #final_responding_gt[gt_respoding_instance!=0]=255


        # threshold_green=170
        # threshold_blue=170
        # lrpimc_green=lrpimc[:,:,1]
        # lrpimc_blue=lrpimc[:,:,2]
        # gb=lrpimc_green+lrpimc_blue
        # mask[lrpimc_green>threshold_green]=0
        # mask[lrpimc_green<=threshold_green]=255
        # mask[lrpimc_blue<=threshold_blue]=255
        # mask[gb<=100]=255
        # plt.imshow(mask)
        # plt.show()


        # new_con=sorted(conp,key=lambda item:item.area,reverse=True)
        # for i in range(10):
        #     label=new_con[i].label
        #     label_toshow=np.zeros(lrpimc.shape[0:2])
        #     label_toshow[con==label]=255
        #     plt.imshow(label_toshow)
        #     plt.ion()
        #     plt.pause(0.6)  # 显示秒数
        #     plt.close()


        # gtmaskblank_result[gtcon==9]=255
        # convex=conp[current_id-1].convex_image
        # [x1,y1,x2,y2]=conp[current_id-1].bbox
        # use=max_convex_map[x1:x2,y1:y2]
        # use[convex==True]=255
        # contours=measure.find_contours(max_convex_map,254)
        # double_contour_map[con==current_id]=255
        # # mask[]
        # gt_responding_value=gtcon[(x2+x1)//2,(y2+y1)//2]
        # gt_respoding_instance[gtcon==gt_responding_value]=100

        # for n, contour in enumerate(contours):
        #     for x,y in contour:
        #         double_contour_map[int(x+0.5),int(y+0.5)]=255
        # # for n, contour in enumerate(contours):
        # #     plt.plot(contour[:, 1], contour[:, 0], linewidth=0.5,color='yellow')
        # # plt.show()
        # double_contour_map[double_contour_map==0]=1
        # double_contour_map[double_contour_map==255]=0
        # double_contour_map[double_contour_map==1]=255
        # cmcon=measure.label(double_contour_map)
        # blank_result[cmcon==2]=200
        # mask88[con==88]=255
        # print(dice(gt_respoding_instance!=0,blank_result!=0))
        # plt.imshow(gt_respoding_instance)
        # plt.show()
        # gt_respoding_instance=gt_respoding_instance+blank_result
        # plt.subplot(1,2,1)
        # plt.imshow(con)
        # plt.subplot(1,2,2)
        # plt.imshow(gt[:,:,0])
        # plt.show()




        #
        # plt.imshow(gtmaskblank_result)
        # plt.show()
        # plt.imshow(blank_result)
        # plt.show()
        # plt.imshow(double_contour_map)
        # plt.show()
        # plt.imshow(max_convex_map)
        # plt.show()
        # max_convex_map=max_convex_map+gt_respoding_instance
        # plt.imshow(max_convex_map)
        # plt.show()
        # plt.imshow(gray)
        # plt.show()
        # plt.imshow(lrpimc)
        # plt.show()

    # final_visual=np.zeros((lrpimc.shape[0],lrpimc.shape[1],3))
    # f1=final_visual[:,:,0]
    # f1[final_mask!=0]=255
    # final_visualgt=np.zeros((lrpimc.shape[0],lrpimc.shape[1],3))
    # f2=final_visual[:,:,1]
    # f2[final_responding_gt!=0]=255
    # plt.imshow(final_visual)
    # plt.show()
    # plt.imshow(final_mask)
    # plt.show()
    # plt.pause(0.6)  # 显示秒数
    # plt.close()
    # plt.imshow(final_responding_gt)
    # plt.show()
    # plt.pause(0.6)  # 显示秒数
    # plt.close()
    # print(dice(final_responding_gt != 0, final_mask != 0))
