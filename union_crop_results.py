import numpy as np
import warnings
warnings.filterwarnings("ignore")
from skimage import io ,measure
import matplotlib.image as mi
import matplotlib.pyplot as plt
from scipy import misc
import math
import os,shutil
from PIL import Image
VAL_IMAGE_IDS =  [
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
Monudir='/data1/wyj/M/datasets/MoNuSACGT/stage1_train/'
ALL_IMAGE_IDS=os.listdir(Monudir)
TRAIN_IMAGE_IDS=list(set(ALL_IMAGE_IDS) - set(VAL_IMAGE_IDS))
print(TRAIN_IMAGE_IDS)
IDS=[TRAIN_IMAGE_IDS,VAL_IMAGE_IDS]
for filename in os.listdir(Monudir):
    oriim=io.imread(Monudir+'{}/images/{}.png'.format(filename,filename))
    masksdir=os.path.join(Monudir,filename+'/masks/')
    maskall=np.zeros((1000,1000),dtype=np.int32)
    count=0
    for maskfile in os.listdir(masksdir):
        count+=1
        mask=io.imread(masksdir+maskfile)
        maskall[mask!=0]=count
    io.imsave('/data1/wyj/M/samples/PRM/WeakNucleiSeg/data/MO/images/'+filename+'.png',oriim)
    im=Image.fromarray(maskall)
    im.save('/data1/wyj/M/samples/PRM/WeakNucleiSeg/data/MO/labels_instance/'+filename+'_label.png')