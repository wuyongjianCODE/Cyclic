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
# from utils.visualizations import GradCAM, GuidedGradCAM, GBP
# from utils.visualizations import LRP, CLRP, LRPA, LRPB, LRPE
# from utils.visualizations import SGLRP, SGLRPSeqA, SGLRPSeqB
# from utils.helper import heatmap
# import innvestigate.utils as iutils
from skimage import io
import os,sys
from keras.preprocessing.image import img_to_array, load_img,ImageDataGenerator
import os
import sys
import json
import datetime
import numpy as np
import skimage.io
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
    MaskrcnnPath = '../../mask_rcnn_coco.h5'
    model.load_weights(os.path.abspath(MaskrcnnPath), by_name=True)
    model.fit_generator(
        train_generator,
        steps_per_epoch=4000* (int(args.LOOP)+1),
        epochs=1,
        validation_data=validation_generator,
        validation_steps=10)
    model.save('resnet5cover.h5')
    model.save(os.path.join(args.LRPTS_DIR,'classification_model_of_loop_'+str(args.LOOP)+'.h5'))
    score = model.evaluate_generator(generator=validation_generator,
                                     workers=1,
                                     use_multiprocessing=False,
                                     verbose=0)

    print('%s: %.2f' % (model.metrics_names[0], score[0]))  # Loss
    print('%s: %.2f%%' % (model.metrics_names[1], score[1] * 100))  # metrics1
    print('%s: %.2f%%' % (model.metrics_names[2], score[2] * 100))  # metrics2
    print('%s: %.2f%%' % (model.metrics_names[3], score[3] * 100))  # metrics3
    print('%s: %.2f%%' % (model.metrics_names[4], score[4] * 100))  # metrics3
    print('%s: %.2f%%' % (model.metrics_names[5], score[5] * 100))  # metrics3
    print('%s: %.2f%%' % (model.metrics_names[6], score[6] * 100))  # metrics3
    print('%s: %.2f%%' % (model.metrics_names[7], score[7] * 100))  # metrics3
    print('%s: %.2f%%' % (model.metrics_names[8], score[8] * 100))  # metrics3
    print('%s: %.2f%%' % (model.metrics_names[9], score[9] * 100))  # metrics3


