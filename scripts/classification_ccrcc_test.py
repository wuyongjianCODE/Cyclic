"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018/

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate submission file
    python3 nucleus.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import slic2mask2
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from keras.layers import Dense, GlobalAveragePooling2D,Flatten
import keras.layers as KL
from keras import metrics
from keras import backend as K
import numpy as np
from skimage import io
from utils.visualizations import GradCAM, GuidedGradCAM, GBP
from utils.visualizations import LRP, CLRP, LRPA, LRPB, LRPE
from utils.visualizations import SGLRP, SGLRPSeqA, SGLRPSeqB
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from utils.helper import heatmap
import innvestigate.utils as iutils
from skimage import io
import os,sys
from keras.preprocessing.image import img_to_array, load_img,ImageDataGenerator
import os
import sys
import json
import datetime
import numpy as np
from skimage import io,measure
import test
from imgaug import augmenters as iaa
import tensorflow as tf
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from keras.preprocessing.image import img_to_array, load_img,ImageDataGenerator
from mrcnn import visualize
from keras import optimizers
import os,shutil
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
epoch_of_current_iteration=np.array([10,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/nucleus/")

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to surve as a validation set.
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
subset1 = [
                "TCGA-G9-6362-01Z-00-DX1",
                "TCGA-DK-A2I6-01A-01-TS1",
                "TCGA-G2-A2EK-01A-02-TSB",
                "TCGA-AY-A8YK-01A-01-TS1",
                "TCGA-NH-A8F7-01A-01-TS1",
                "TCGA-KB-A93J-01A-01-TS1",
                "TCGA-RD-A8N9-01A-01-TS1",
            ]
subset2 = [
                "TCGA-E2-A1B5-01Z-00-DX1",
                "TCGA-E2-A14V-01Z-00-DX1",
                "TCGA-21-5784-01Z-00-DX1",
                "TCGA-21-5786-01Z-00-DX1",
                "TCGA-B0-5698-01Z-00-DX1",
                "TCGA-B0-5710-01Z-00-DX1",
                "TCGA-CH-5767-01Z-00-DX1",
            ]

############################################################
#  Configurations
############################################################
def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.

    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)
def metric_precision(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=TP/(TP+FP)
    return precision

def metric_recall(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    recall=TP/(TP+FN)
    return recall

def metric_F1score(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1score=2*precision*recall/(precision+recall)
    return F1score


############################################################
#  RLE Encoding
############################################################
def rescale(img):
    img2=np.copy(img).astype(np.float)
    max=np.max(img2)
    min=np.min(img2)
    img2=img2-min
    img2=img2*255/(max-min)
    return img2.astype(np.int)
def set_threshold(img,thres):
    max=np.max(img)
    min=np.min(img)
    img2=np.copy(img)
    value=max-(max-min)*thres
    img2[img<=value]=-1
    img2[img>value]=0
    img2[img==-1]=255
    return img2
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    parser.add_argument('--iteration', required=False,
                        default=0,
                        metavar="the iteration num",
                        help='as shown')
    parser.add_argument('--LRPTS_DIR', required=False,
                        metavar="the LRPTS dir path",
                        help='as shown')
    parser.add_argument('--LOOP', required=False,
                        metavar="the LOOP number",
                        help='as shown')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    # if args.command == "train":
    #     config = NucleusConfig()
    # else:
    #     config = NucleusInferenceConfig()
    # config.display()

    # Create model
    # if args.command == "train":
    #     Tempmodel = modellib.MaskRCNN(mode="training", config=config,
    #                               model_dir=args.logs)
    # else:
    #     Tempmodel = modellib.MaskRCNN(mode="inference", config=config,
    #                               model_dir=args.logs)
    input_image = KL.Input(
        shape=[None, None, 3], name="input_image")
    x=modellib.buildNP(input_image)
    x = GlobalAveragePooling2D()(x)
    # x = Flatten()(x)
    # 添加一个全连接层
    x = Dense(1024, activation='relu')(x)

    # 添加一个分类器，假设我们有2个类
    predictions = Dense(2, activation='softmax')(x)

    # 构建我们需要训练的完整模型
    model = Model(inputs=input_image, outputs=predictions)
    sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,  # 'rmsprop',
                  loss='categorical_crossentropy',
                  metrics=[metrics.mae, metrics.categorical_accuracy, metric_precision, metric_recall, metric_F1score,
                           precision, recall, fmeasure, fbeta_score])

    # if args.weights.lower() == "coco":
    #     weights_path = COCO_WEIGHTS_PATH
    #     # Download weights file
    #     if not os.path.exists(weights_path):
    #         utils.download_trained_weights(weights_path)
    # elif args.weights.lower() == "last":
    #     # Find last trained weights
    #     weights_path = model.find_last()
    # elif args.weights.lower() == "imagenet":
    #     # Start from ImageNet trained weights
    #     weights_path = model.get_imagenet_weights()
    # else:
    #     weights_path = args.weights

    # Load weights
    weights_path = COCO_WEIGHTS_PATH
    if int(args.LOOP)>0:
        weights_path = '../../best.h5'
    if not os.path.exists(args.LRPTS_DIR):
        os.mkdir(args.LRPTS_DIR)
    tpc=os.path.join(args.LRPTS_DIR,'TS_of_loop'+str(args.LOOP))
    tp=os.path.join(args.LRPTS_DIR,'TS_of_loop'+str(int(args.LOOP)-1)+'/')
    if not os.path.exists(tpc):
        os.mkdir(tpc)
    print("Loading weights ", weights_path)
    # if args.weights.lower() == "coco":
    #     # Exclude the last layers because they require a matching
    #     # number of classes
    #     model.load_weights(weights_path, by_name=True, exclude=[
    #         "mrcnn_class_logits", "mrcnn_bbox_fc",
    #         "mrcnn_bbox", "mrcnn_mask"])
    # else:
    model.load_weights(weights_path, by_name=True)
    #model.summary()
    # model.uses_learning_phase=True
    #
    # model=modellib.remodel(model)
    # for layer in model.layers[:-3     z]:
    #     layer.trainable = False
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    NP = args.dataset
    NP_remained = args.dataset
    train_generator = train_datagen.flow_from_directory(
        NP,
        target_size=(224, 224),
        batch_size=10)

    validation_generator = test_datagen.flow_from_directory(
        NP_remained,
        target_size=(224, 224),
        batch_size=10)
    MaskrcnnPath = "/data1/wyj/M/logs/LRPTS20220612T2022/classification_model_of_loop_1.h5"
    MaskrcnnPath = "resnet5cover.h5"
    model.load_weights(os.path.abspath(MaskrcnnPath), by_name=True)
    # score = model.evaluate_generator(generator=validation_generator,
    #                                  workers=1,
    #                                  use_multiprocessing=False,
    #                                  verbose=0)
    #
    # print('%s: %.2f' % (model.metrics_names[0], score[0]))  # Loss
    # print('%s: %.2f%%' % (model.metrics_names[1], score[1] * 100))  # metrics1
    # print('%s: %.2f%%' % (model.metrics_names[2], score[2] * 100))  # metrics2
    # print('%s: %.2f%%' % (model.metrics_names[3], score[3] * 100))  # metrics3
    # print('%s: %.2f%%' % (model.metrics_names[4], score[4] * 100))  # metrics3
    # print('%s: %.2f%%' % (model.metrics_names[5], score[5] * 100))  # metrics3
    # print('%s: %.2f%%' % (model.metrics_names[6], score[6] * 100))  # metrics3
    # print('%s: %.2f%%' % (model.metrics_names[7], score[7] * 100))  # metrics3
    # print('%s: %.2f%%' % (model.metrics_names[8], score[8] * 100))  # metrics3
    # print('%s: %.2f%%' % (model.metrics_names[9], score[9] * 100))  # metrics3

    partial_model = Model(
        inputs=model.inputs,
        outputs=iutils.keras.graph.pre_softmax_tensors(model.outputs),
        name=model.name,
    )
# # These values are set due to Keras Applications. Change this to a range suitable for your model.
    max_input = 255
    min_input = -255
    BASEDIR = '/data1/wyj/M/datasets/ccrcccrop/Test/Images/'
    Maindir = '/data1/wyj/M/datasets/ccrcccrop/Test/Images/'#/data1/wyj/M/datasets/ccrcccrop/Test/Images/
    logdir = 'LRPmaps_ccrcc'
    target_class = -1  # ImageNet "zebra"
    # GradCAM and GuidedGradCAM requires a specific layer
    target_layer = "global_average_pooling2d_1"  # VGG only
    use_relu = True
    lrp_clsn = []
    for i in range(2):
        lrp_clsn.append(LRPA(
            partial_model,
            target_id=i,
            relu=use_relu,
        ))
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    GENERATION=0
    STAGE_TRAIN_Gen=r'/data1/wyj/M/datasets/ccc/'
    if not os.path.exists(STAGE_TRAIN_Gen):
        os.mkdir(STAGE_TRAIN_Gen)
    for name in os.listdir(Maindir):
        target_class = 1
        name_dir=STAGE_TRAIN_Gen
        name_dir_imges=name_dir+'/images'
        name_dir_mask=name_dir+'/mask'
        name_dir_masks=name_dir+'/masks'
        name_dir_LRP=name_dir+'/LRP'
        if not os.path.exists(name_dir):
            os.mkdir(name_dir)
        if not os.path.exists(name_dir_imges):
            os.mkdir(name_dir_imges)
        if not os.path.exists(name_dir_masks):
            os.mkdir(name_dir_masks)
        if not os.path.exists(name_dir_mask):
            os.mkdir(name_dir_mask)
        if not os.path.exists(name_dir_LRP):
            os.mkdir(name_dir_LRP)
        try:
            orig_imgs = [img_to_array(load_img(Maindir+name, target_size=(256, 256)))]
        except:
            continue
        # gt_imgs = [img_to_array(load_img(Maindir+fname+'/mask/'+fname+'.png', target_size=(520, 704)))]
        input_imgs = np.copy(orig_imgs)
        # preprocess input for model
        input_imgs = preprocess_input(input_imgs)  # for built in keras models
        example_id = 0
        imtoshow = input_imgs[0]
        # plt.imshow(imtoshow)
        # io.imsave(logdir+'/{}_MID.png'.format(fname[:-4]),imtoshow)

        predictions = model.predict(input_imgs)
        pred_id = np.argmax(predictions[example_id])
        # print(decode_predictions(predictions))
        if pred_id != target_class:
            print(name)
        # partial_gradcam_analyzer = GradCAM(
        #     model=partial_model,
        #     target_id=target_class,
        #     layer_name=target_layer,
        #     relu=use_relu,
        # )
        # analysis_partial_grad_cam = partial_gradcam_analyzer.analyze(input_imgs)
        # heatmap(analysis_partial_grad_cam[example_id].sum(axis=(2)))
        # plt.show()
        INCH1 = 32
        INCH2 = 32
        DPI = 8

        analysis_lrpa = lrp_clsn[pred_id].analyze(input_imgs)
        heatmap(analysis_lrpa[example_id].sum(axis=(2)))
        fig = plt.gcf()
        fig.set_size_inches(INCH1, INCH2)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        # plt.show()
        fig.savefig(name_dir_LRP+'/{}_LRPA.png'.format(name), format='png', transparent=True, dpi=DPI,
                    pad_inches=0)
        print(name_dir_LRP+'/{}_LRPA.png'.format(name))
    TRAINDIR = '/data1/wyj/M/datasets/ccrcccrop/Train/Images/'
    for name in os.listdir(name_dir_LRP):
        print(name+'  generating!!')
        name_dir=STAGE_TRAIN_Gen+name
        name_dir_imges=name_dir+'/images'
        name_dir_mask=name_dir+'/mask'
        name_dir_masks=name_dir+'/masks'
        name_dir_LRP=name_dir+'/LRP'
        try:
            LRPa=io.imread(os.path.join(name_dir_LRP,name+'_LRPA.png'))[:,:,1]
        except:
            continue
        LRPA_thre=set_threshold(LRPa,0.1)
        LRPA_con=measure.label(LRPA_thre)
        LRPA_conprop=measure.regionprops(LRPA_con)
        # plt.imshow(LRPA_thre)
        # plt.show()
        # plt.imshow(LRPA_con)
        # plt.show()
        #declare all nparrays needed!!!!!!!!
        score = np.zeros(len(LRPA_conprop))
        final_mask=np.zeros(LRPa.shape[0:2],dtype=np.int16)
        final_responding_gt=np.zeros(LRPa.shape[0:2])
        mask = np.zeros(LRPa.shape[0:2])
        blank_result = np.zeros(LRPa.shape[0:2])
        mask88 = np.zeros(LRPa.shape[0:2])
        max_convex_map = np.zeros(LRPa.shape[0:2])
        gtmaskblank_result = np.zeros(LRPa.shape[0:2])
        gt_respoding_instance = np.zeros(LRPa.shape[0:2],dtype=np.int16)
        double_contour_map = np.zeros(LRPa.shape[0:2])
        #end declare!!!!!!!!!
        AREA_WEIGHT=1
        mean_area=300
        for i in range(len(LRPA_conprop)) :
            if LRPA_conprop[i].area<100:
                continue
            # convex_rate=LRPA_conprop[i].filled_area / LRPA_conprop[i].convex_area
            # if convex_rate > 0.5:
            #     weight=(convex_rate - 0.5)
            # else:weight=0
            # [x1, y1, x2, y2] = LRPA_conprop[i].bbox
            # X=x2-x1
            # Y=y2-y1
            #score[i] = 300 * convex_rate - AREA_WEIGHT * abs(LRPA_conprop[i].convex_area - 600) - 200 * weight * (((X-Y)/(X+Y))*((X-Y)/(X+Y))) #L3 0789
            score[i] =AREA_WEIGHT*abs(LRPA_conprop[i].area - mean_area) #- 200 * weight * (((X-Y)/(X+Y))*((X-Y)/(X+Y)))
        rank=np.argsort(score)
        LENGTH=20
        IDS=np.zeros(LENGTH,dtype=np.int)
        for cid in range(LENGTH):
            # gtmaskblank_result[gtcon == 9] = 255
            try:
                current_id=LRPA_conprop[rank[-1-cid]].label
            except:
                print('{} has only {} pseudo masks,less than required 20+ masks!'.format(name,cid))
                break
            convex=LRPA_conprop[current_id-1].convex_image
            if LRPA_conprop[current_id-1].area<1000 and cid<LENGTH:
                blank= np.zeros(LRPa.shape[0:2])
                [x1,y1,x2,y2]=LRPA_conprop[current_id-1].bbox
                use = blank[x1:x2, y1:y2]
                use[convex == True] = 255
                io.imsave(name_dir_masks+'/{}_{}.png'.format(name,cid),blank.astype(np.uint8))
                use=final_mask[x1:x2,y1:y2]
                use[convex==True]=cid
        # plt.subplot(1,3,1)
        # plt.imshow(final_mask)
        # plt.subplot(1,3,2)
        # plt.imshow(LRPa)
        # plt.show()
        io.imsave(name_dir_mask+'/{}.png'.format(name),final_mask!=0)
