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
# if __name__ == '__main__':
#     import matplotlib
#     matplotlib.get_backend()
import shutil

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
from skimage import io,measure
from utils.visualizations import GradCAM, GuidedGradCAM, GBP
from utils.visualizations import LRP, CLRP, LRPA, LRPB, LRPE
from utils.visualizations import SGLRP, SGLRPSeqA, SGLRPSeqB
from utils.helper import heatmap
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import innvestigate.utils as iutils
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

class NucleusConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "nucleus"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 320// IMAGES_PER_GPU#(657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet152"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


class NucleusInferenceConfig(NucleusConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################

class NucleusDataset(utils.Dataset):

    def load_nucleus(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("nucleus", 1, "nucleus")

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", "stage1_train", "stage1_test", "stage2_test",'subset1','subset2']
        subset_dir = "stage1_train" if subset in ["train", "val"] else subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        else:
            # Get image ids from directory names
            image_ids = next(os.walk(dataset_dir))[1]
            if subset == "train":
                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))
        if subset == "subset1":
            image_ids = subset1
        if subset == "subset2":
            image_ids = subset2

        # Add images
        for image_id in image_ids:
            self.add_image(
                "nucleus",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id, "images/{}.png".format(image_id)))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")

        # Read mask files from .png image
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model, dataset_dir, subset,iter=0):
    """Train the model."""
    # Training dataset.
    dataset_train = NucleusDataset()
    dataset_train.load_nucleus(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NucleusDataset()
    dataset_val.load_nucleus(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epoch_of_current_iteration[iter],
                augmentation=augmentation,
                layers='heads')

    # print("Train all layers")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=40,
    #             augmentation=augmentation,
    #             layers='all')


############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset, iter=0):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = NucleusDataset()
    dataset.load_nucleus(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    # mask_count=[]
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        # visualize.display_instances(
        #     image, r['rois'], r['masks'], r['class_ids'],
        #     dataset.class_names, r['scores'],
        #     show_bbox=False, show_mask=True,
        #     title="Predictions")
        # plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))
        mask_result=np.zeros((image.shape[0],image.shape[1]))
        # mask_count= np.zeros((image.shape[0], image.shape[1]))
        masks=r['masks']
        scores=r['scores']
        for i in range(masks.shape[2]):
            mask_result[masks[:,:,i]==1]=255
             # mask_count[masks[:, :, i] == 1] = mask_count[masks[:, :, i] == 1]+1
            # if mask_count.max()>1:
            #     swit=False
            mask_temp=np.array(masks[:,:,i])
            mask_temp[mask_temp==1]=255
            try:
                os.mkdir(r'./masks')
            except:
                pass
            ID=dataset.image_info[image_id]["id"]
            name_path = r'../../datasets/MoNuSAC/stage1_train/' + ID + '/masks'
            if scores[i]>0.95:
                skimage.io.imsave("{}/{}.png".format(os.path.abspath(name_path),dataset.image_info[image_id]["id"]+'_mask_'+str(i)),mask_temp)
        skimage.io.imsave("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]).replace(' ','_'),mask_result)
#   test.test(submit_dir,r'D:\BaiduNetdiskDownload\BrestCancer\MoNuSAC-master\MoNuSAC_mask')
    print("useless test GT:----------------------------------------------------------")
    test.testGT(submit_dir)
    print("useless test GT:directly detect-----------------------------------------------------done")
    # Save to csv file
    #slic2mask2.union(submit_dir)
    #slic2mask2.slic2mask3(iteration_num=iter)
    #savdir=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\datasets'
    #submit_dir=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\submit_20200920T133550'
    #seg_mask.seg_mask(submit_dir,os.path.join(savdir,'masks'),iteration_num=iter)


############################################################
#  Command Line
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

    IMG=io.imread("/data1/wyj/M/samples/nucleus/LRPmaps_8COCO/SHSY5Y_Phase_A10_2_02d08h00m_3_ORI.png")
    LRPAPATH=io.imread("/data1/wyj/M/samples/nucleus/LRPmaps_8COCO/SHSY5Y_Phase_A10_2_02d08h00m_3_LRPA.png")
    LRPDIR='/data1/wyj/M/samples/nucleus/LRPmaps_8COCOORI/'
    SEG_TRAIN_DATASET='/data1/wyj/M/datasets/L/LIVElrp/stage1_train'
    SAVE_TEMP_DATASET='/data1/wyj/M/datasets/L/LIVElrp/stage1_train_LRP'
    AREA_WEIGHT=1
    mean_area = 600
    # shutil.copytree(SEG_TRAIN_DATASET,SAVE_TEMP_DATASET)
        # plt.subplot(1,3,3)
        # plt.imshow(final_mask)
        # plt.imshow(con_sq)
        # plt.show()
    # IMG_thre=set_threshold(IMG,0.35)
    # plt.imshow(IMG_thre)
    # plt.show()
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
    model.compile(optimizer=sgd,#'rmsprop',
                  loss='categorical_crossentropy',
                  metrics=[metrics.mae, metrics.categorical_accuracy,metric_recall,metric_precision,metric_F1score])
    weights_path='resnet8COCO2.h5'
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)
    #model.summary()
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    NP = r'../../datasets/MyNP'
    NP_remained = r'../../datasets/MyNP'
    train_generator = train_datagen.flow_from_directory(
        NP,
        target_size=(224, 224),
        batch_size=10)

    validation_generator = test_datagen.flow_from_directory(
        NP_remained,
        target_size=(224, 224),
        batch_size=10)
    # MaskrcnnPath = '../../mask_rcnn_coco.h5'
    # model.load_weights(os.path.abspath(MaskrcnnPath), by_name=True)
    # model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=4000,
    #     epochs=1,
    #     validation_data=validation_generator,
    #     validation_steps=10)
    # model.save('resnet5cover.h5')
    # model.save(os.path.join(args.LRPTS_DIR,'classification_model_of_loop_'+str(args.LOOP)+'.h5'))
    # score = model.evaluate_generator(generator=validation_generator,
    #                                  workers=1,
    #                                  use_multiprocessing=False,
    #                                  verbose=0)
    #
    # print('%s: %.2f' % (model.metrics_names[0], score[0]))  # Loss
    # print('%s: %.2f%%' % (model.metrics_names[1], score[1] * 100))  # metrics1
    # print('%s: %.2f%%' % (model.metrics_names[2], score[2] * 100))  # metrics2
    partial_model = Model(
        inputs=model.inputs,
        outputs=iutils.keras.graph.pre_softmax_tensors(model.outputs),
        name=model.name,
    )
# # These values are set due to Keras Applications. Change this to a range suitable for your model.
    max_input = 255
    min_input = -255
    BASEDIR = r'/data1/wyj/M/datasets/L/images/livecell_test_images/'
    Maindir = r'/data1/wyj/M/datasets/L/images/livecell_test_images/'
    Maskdir = r'/data1/wyj/M/datasets/L/LIVE4/stage1_train/'
    nam = Maskdir + '{}'
    logdir = 'LRPmaps_8COCOORI'
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
    STAGE_TRAIN_Gen=r'/data1/wyj/M/datasets/L/LIVElrp/stage1_train/'.format(GENERATION)
    if not os.path.exists(STAGE_TRAIN_Gen):
        os.mkdir(STAGE_TRAIN_Gen)
    # for name in os.listdir(SEG_TRAIN_DATASET):
    #     target_class = 1 if name.startswith('BV2') else 0
    #     name_dir=STAGE_TRAIN_Gen+name
    #     name_dir_imges=name_dir+'/images'
    #     name_dir_mask=name_dir+'/mask'
    #     name_dir_masks=name_dir+'/masks'
    #     name_dir_LRP=name_dir+'/LRP'
    #     if not os.path.exists(name_dir):
    #         os.mkdir(name_dir)
    #     if not os.path.exists(name_dir_imges):
    #         os.mkdir(name_dir_imges)
    #     if not os.path.exists(name_dir_masks):
    #         os.mkdir(name_dir_masks)
    #     if not os.path.exists(name_dir_mask):
    #         os.mkdir(name_dir_mask)
    #     if not os.path.exists(name_dir_LRP):
    #         os.mkdir(name_dir_LRP)
    #     Maindir = os.path.join(SEG_TRAIN_DATASET, name)
    #     shutil.copy(Maindir+'/images/'+name+'.tif',name_dir_imges+'/'+name+'.tif')
    #     try:
    #         orig_imgs = [img_to_array(load_img(Maindir+'/images/'+name+'.tif', target_size=(520, 704)))]
    #     except:
    #         continue
    #     # gt_imgs = [img_to_array(load_img(Maindir+fname+'/mask/'+fname+'.png', target_size=(520, 704)))]
    #     input_imgs = np.copy(orig_imgs)
    #     # preprocess input for model
    #     input_imgs = preprocess_input(input_imgs)  # for built in keras models
    #     example_id = 0
    #     imtoshow = input_imgs[0]
    #     # plt.imshow(imtoshow)
    #     # io.imsave(logdir+'/{}_MID.png'.format(fname[:-4]),imtoshow)
    #
    #     predictions = model.predict(input_imgs)
    #     pred_id = np.argmax(predictions[example_id])
    #     # print(decode_predictions(predictions))
    #     if pred_id != target_class:
    #         print(name)
    #     # partial_gradcam_analyzer = GradCAM(
    #     #     model=partial_model,
    #     #     target_id=target_class,
    #     #     layer_name=target_layer,
    #     #     relu=use_relu,
    #     # )
    #     # analysis_partial_grad_cam = partial_gradcam_analyzer.analyze(input_imgs)
    #     # heatmap(analysis_partial_grad_cam[example_id].sum(axis=(2)))
    #     # plt.show()
    #     INCH1 = 88
    #     INCH2 = 65
    #     DPI = 8
    #
    #     analysis_lrpa = lrp_clsn[pred_id].analyze(input_imgs)
    #     heatmap(analysis_lrpa[example_id].sum(axis=(2)))
    #     fig = plt.gcf()
    #     fig.set_size_inches(INCH1, INCH2)
    #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #     plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #     plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    #     plt.margins(0, 0)
    #     fig.savefig(name_dir_LRP+'/{}_LRPA.png'.format(name), format='png', transparent=True, dpi=DPI,
    #                 pad_inches=0)
    for name in os.listdir(STAGE_TRAIN_Gen):
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
# metrics1
    # train_datagen = ImageDataGenerator(
    #     rescale=1. / 255,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True)
    #
    # test_datagen = ImageDataGenerator(rescale=1. / 255)
    # NP = r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\samples\nucleus\MyNP'
    # train_generator = train_datagen.flow_from_directory(
    #     NP,
    #     target_size=(224, 224),
    #     batch_size=20)
    #
    # validation_generator = test_datagen.flow_from_directory(
    #     NP,
    #     target_size=(224, 224),
    #     batch_size=20)
    #
    # model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=30000,
    #     epochs=1,
    #     validation_data=validation_generator,
    #     validation_steps=100)
    # model.save('vgg6w.h5')


    # Train or evaluate
    # if args.command == "train":
    #     train(model, args.dataset, args.subset,iter=int(args.iteration))
    # elif args.command == "detect":
    #     detect(model, args.dataset, args.subset,iter=int(args.iteration))
    # else:
    #     print("'{}' is not recognized. "
    #           "Use 'train' or 'detect'".format(args.command))