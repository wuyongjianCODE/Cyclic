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
import imageio

if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import slic2mask2
import os,shutil
import sys
import json
import datetime
import numpy as np
import skimage.io
import skimage.measure as measure
import test_4metric_cc as testcc
from imgaug import augmenters as iaa
import fix4
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
from scipy.io import loadmat
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
from scipy.io import loadmat
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#epoch_of_current_iteration=np.array([10,8,4,4,4,4,3,1,1,1,1,1,1,1,1,1,1])
epoch_of_current_iteration=np.array([25,25,5,5,5,1,3,1,2,3,4,4,4,4,4,1,1])
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/ccrcccrop/")
TAKE_5_STEP=False
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


############################################################
#  Configurations
############################################################
def updateTargetModel(model,targetmodel):
    modelWeights=model.trainable_weights
    targetmodelweights=targetmodel.trainable_weights
    for i in range(len(targetmodelweights)):
        targetmodelweights[i].assign(modelWeights[i])
class NucleusConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "ccrcccrop"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1
    #LEARNING_RATE = 0.0001  #default
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus
    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 28// IMAGES_PER_GPU#(657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet101"

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
    #MEAN_PIXEL = np.array([43.53, 39.56, 48.22])
    MEAN_PIXEL = np.array([189.44, 136.76, 169.23])
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
    # TRAIN_ROIS_PER_IMAGE = 256
    #
    # # Maximum number of ground truth instances to use in one image
    # MAX_GT_INSTANCES = 400
    #
    # # Max number of final detections per image
    # DETECTION_MAX_INSTANCES = 800


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
        self.add_class("ccrcc", 1, "ccrcc1")
        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        image_ids = next(os.walk(dataset_dir+'/{}/Images'.format(subset)))[2]

        # Add images
        for image_id in image_ids:
            self.add_image(
                "ccrcc",
                image_id=image_id,
                path=os.path.join(dataset_dir+'/{}/'.format(subset), "Images/{}".format(image_id)))
        print('done load_nucleus')
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        path=os.path.dirname(info['path'])
        mask_dir = os.path.join(os.path.dirname(path[:path.rfind('Images')]), "Labels")
        instance_mask_dir=os.path.join(os.path.dirname(path[:path.rfind('Images')]), "Instance_mask")
        # Read mask files from .png image
        mask = []
        mask2=[]
        classes=[]
        f=info['id'].replace('.png','.mat')
        # mat = loadmat(os.path.join(mask_dir, f))
        # ins_map=mat['instance_map']
        # cls_map = mat['class_map']
        # for i in range(1,np.max(ins_map)+1):
        #     jac = ins_map==i
        #     tester=np.max(jac)
        #     if tester==False:
        #         continue
        #     mask.append(jac)
        #     cls=np.max(cls_map[jac])
        #     classes.append(cls)
        for id in os.listdir(instance_mask_dir+'/'+f[:-4]):
            inm=skimage.io.imread(os.path.join(instance_mask_dir+'/'+f[:-4],id))
            jac= inm!=0
            mask.append(jac)

        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask,np.ones([mask.shape[-1]], dtype=np.int32)# np.array(classes)

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
    dataset_val.load_nucleus(dataset_dir, "Valid")
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
    # print("Train network heads")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=epoch_of_current_iteration[iter],
    #             augmentation=augmentation,
    #             layers='heads')
    print("Train all layer")
    if TAKE_5_STEP==True:
        for i in range(5):
            model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE,
                        epochs=(epoch_of_current_iteration[iter])/5,
                        augmentation=augmentation,
                        layers='all')
            fix4.fix()
    else:
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=epoch_of_current_iteration[iter],
                    augmentation=augmentation,
                    layers='all')
    # print("Train all layer")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=epoch_of_current_iteration[iter],
    #             augmentation=augmentation,
    #             layers='all')
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
def random_rgb():
    r = np.random.rand()*255
    g = np.random.rand()*255
    b = np.random.rand()*255
    if r==0 and g==0 and b==0:
        r=254
        g=254
        b=254
    return np.array([r, g, b]).astype(np.uint8)
def bijective(im,cg):
    imcon=measure.label(im[:,:]>0)
    imcon_prop=measure.regionprops(imcon)
    newim=np.zeros((im.shape[0],im.shape[1],3),np.uint8)
    for i in range(len(imcon_prop)):
        x1, y1, x2, y2 = imcon_prop[i].bbox  # bbox of 250*250 p map
        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)
        color = cg[x,y,:]
        if np.max(color)==0:
            color=random_rgb()
        newim[imcon==i+1,0]=color[0]
        newim[imcon==i+1,1]=color[1]
        newim[imcon==i+1,2]=color[2]
    return newim
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
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        #here we save the mat labelmap:
        path=dataset.image_info[image_id]['path']
        # mat_path=path.replace('Images','Labels').replace('.png','.mat')
        # mat = loadmat(mat_path)
        # ins_map=mat['instance_map']
        # color_ins_map=np.zeros((ins_map.shape[0],ins_map.shape[1],3),dtype=np.uint8)
        # for i in range(1, np.max(ins_map)+1):
        #     color = random_rgb()
        #     color_ins_map[ins_map == i, 0] = color[0]
        #     color_ins_map[ins_map == i, 1] = color[1]
        #     color_ins_map[ins_map == i, 2] = color[2]
        mask_path=path.replace('Images','Colormask')
        # skimage.io.imsave(mask_path,color_ins_map)

        COLORMASK=skimage.io.imread(mask_path)
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        # visualize.display_instances(
        #     image, r['rois'], r['masks'], r['class_ids'],
        #     dataset.class_names, r['scores'],
        #     show_bbox=False, show_mask=True,
        #     title="Predictions")
        # plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))
        mask_result=np.zeros((image.shape[0],image.shape[1]),dtype=np.int)
        masks=r['masks']
        if TAKE_5_STEP==True:
            scores=r['scores']
            for i in range(masks.shape[2]):

                 # mask_count[masks[:, :, i] == 1] = mask_count[masks[:, :, i] == 1]+1
                # if mask_count.max()>1:
                #     swit=False
                mask_temp=np.array(masks[:,:,i],dtype=np.uint8)
                mask_temp[mask_temp==1]=255
                try:
                    os.mkdir(r'./masks')
                except:
                    pass
                ID=dataset.image_info[image_id]["id"]
                name_path = r'../../datasets/MoNuSAC/stage1_train/' + ID + '/masks'
                name_backpath=r'../../datasets/MoNuSAC-BACK/stage1_train/'+ID+'/masks'
                if scores[i]>0.85 and i <40:
                    mask_result[masks[:, :, i] == 1] = 255
                    # skimage.io.imsave("{}/{}.png".format(os.path.abspath(name_backpath),
                    #                                      dataset.image_info[image_id]["id"] + '_iteration_' + str(
                    #                                          iter) + '_mask_' + str(i)), mask_temp)
                    skimage.io.imsave("{}/{}".format(os.path.abspath(name_path),dataset.image_info[image_id]["id"]+'_iteration_'+str(iter)+'_mask_'+str(i)),
                                      mask_temp)
        for i in range(masks.shape[2]):
            mask_result[masks[:,:,i]==1]=i+1
        mask_result=bijective(mask_result,COLORMASK)
        skimage.io.imsave("{}/{}".format(submit_dir, dataset.image_info[image_id]["id"]).replace(' ','_'),mask_result)

# test.test(submit_dir,r'D:\BaiduNetdiskDownload\BrestCancer\MoNuSAC-master\MoNuSAC_mask')
    dice=testcc.test_XMetric(submit_dir,r'/data1/wyj/M/datasets/ccrcccrop/Test/Colormask/')
    return dice,submit_dir
    # Save to csv file
    # slic2mask2.union(submit_dir)
    # slic2mask2.slic2mask2(submit_dir,iteration_num=iter)
    #savdir=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\datasets'
    #submit_dir=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results\nucleus\submit_20200920T133550'
    #seg_mask.seg_mask(submit_dir,os.path.join(savdir,'masks'),iteration_num=iter)


############################################################
#  Command Line
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
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = NucleusConfig()
    else:
        config = NucleusInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
    #model.keras_model.summary()
    # plot_model(model, to_file='test.png', show_shapes=True)
    # names = [layer.name for layer in model.layers]
    # for layer in model.keras_model.layers:
    #     for weight in layer.weights:
    #         print(weight.name, weight.shape)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "lasts":
        # Find last trained weights
        weights_path = model.find_lasts()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights

    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        print("Loading weights ", weights_path)
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

    else:
        print("Loading weights ", weights_path)
        # model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
        #     "mrcnn_class_logits", "mrcnn_bbox_fc",
        #     "mrcnn_bbox", "mrcnn_mask"])
        # model.load_weights(model.get_imagenet_weights(), by_name=True)
        try:
            model.load_weights(weights_path, by_name=True)
        except:
            print("Loading pass over : ", weights_path)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset,iter=int(args.iteration))
    elif args.command == "detect":
        if args.weights.lower() == "lasts":
            print('TEST LOOP contain {} steps______________________________________________________________'.format(len(weights_path)))
            scores=np.zeros(len(weights_path),dtype=np.float)
            submits = []
            TSn=os.path.join(args.LRPTS_DIR,'TS_of_loop'+str(args.LOOP))
            if not os.path.exists(args.LRPTS_DIR):
                os.mkdir(args.LRPTS_DIR)
            if not os.path.exists(TSn):
                os.mkdir(TSn)
            for i in range(len(weights_path)):
                print("Loading weights ", weights_path[i])
                model.load_weights(weights_path[i], by_name=True)
                score,submit=detect(model, args.dataset, args.subset, iter=int(args.iteration))
                scores[i]=score
                submits.append(submit)
                shutil.copyfile(weights_path[i],os.path.join(TSn,'Student_num_'+str(i)))
                shutil.copytree(submits[i],TSn+'/'+os.path.basename(submits[i])+'_student_num_'+str(i))
            best_id=np.argmax(scores)
            best_weight_path=weights_path[best_id]
            shutil.copyfile(best_weight_path,'../../best.h5')
        else:
            detect(model, args.dataset, args.subset,iter=int(args.iteration))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
