import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
warnings.filterwarnings("ignore")
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import nucleus


# Directory to save logs and trained model
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Dataset directory
DATASET_DIR = os.path.join(ROOT_DIR, "datasets/MoNuSAC")
#DATASET_DIR = os.path.join(ROOT_DIR, "datasets/nucleus")

# Inference Configuration
config = nucleus.NucleusInferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# Only inference mode is supported right now
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    fig.tight_layout()
    return ax

# Load validation dataset
dataset = nucleus.NucleusDataset()
dataset.load_nucleus(DATASET_DIR, "train")
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference",
                              model_dir=LOGS_DIR,
                              config=config)
# Path to a specific weights file
# weights_path = "/path/to/mask_rcnn_nucleus.h5"

# Or, load the last model you trained
weights_path = model.find_last()#'../../logs/default20200930T0804/mask_rcnn_default_0002.h5'#

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

image_id = random.choice(dataset.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset.image_reference(image_id)))
print("Original image shape: ", modellib.parse_image_meta(image_meta[np.newaxis,...])["original_image_shape"][0])

# Run object detection
results = model.detect_molded(np.expand_dims(image, 0), np.expand_dims(image_meta, 0), verbose=1)

# Display results
r = results[0]
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

# Compute AP over range 0.5 to 0.95 and print it

utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                       r['rois'], r['class_ids'], r['scores'], r['masks'],
                       verbose=1)

visualize.display_differences(
    image,
    gt_bbox, gt_class_id, gt_mask,
    r['rois'], r['class_ids'], r['scores'], r['masks'],
    dataset.class_names, ax=get_ax(1,1),
    show_box=False, show_mask=False,
    iou_threshold=0.5, score_threshold=0.5)
plt.savefig('iteration_demo.png')
# Display predictions only
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            dataset.class_names, r['scores'],ax=get_ax(1,1))
plt.savefig('result.png')
# Display Ground Truth only
visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id,
                            dataset.class_names,title="Ground Truth",ax=get_ax(1,1))
plt.savefig('GT.png')

def compute_batch_ap(dataset, image_ids, verbose=1):
    APs = []
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect_molded(image[np.newaxis], image_meta[np.newaxis], verbose=0)
        # Compute AP over range 0.5 to 0.95
        r = results[0]
        ap = utils.compute_ap_range(
            gt_bbox, gt_class_id, gt_mask,
            r['rois'], r['class_ids'], r['scores'], r['masks'],
            verbose=0)
        APs.append(ap)
        if verbose:
            info = dataset.image_info[image_id]
            meta = modellib.parse_image_meta(image_meta[np.newaxis,...])
            print("{:3} {}   AP: {:.2f}".format(
                meta["image_id"][0], meta["original_image_shape"][0], ap))
    return APs

# Run on validation set

limit = 10
APs = compute_batch_ap(dataset, dataset.image_ids[:limit])
print("Mean AP overa {} images: {:.4f}".format(len(APs), np.mean(APs)))

# # Get anchors and convert to pixel coordinates
# anchors = model.get_anchors(image.shape)
# anchors = utils.denorm_boxes(anchors, image.shape[:2])
# log("anchors", anchors)
#
# # Generate RPN trainig targets
# # target_rpn_match is 1 for positive anchors, -1 for negative anchors
# # and 0 for neutral anchors.
# target_rpn_match, target_rpn_bbox = modellib.build_rpn_targets(
#     image.shape, anchors, gt_class_id, gt_bbox, model.config)
# log("target_rpn_match", target_rpn_match)
# log("target_rpn_bbox", target_rpn_bbox)
#
# positive_anchor_ix = np.where(target_rpn_match[:] == 1)[0]
# negative_anchor_ix = np.where(target_rpn_match[:] == -1)[0]
# neutral_anchor_ix = np.where(target_rpn_match[:] == 0)[0]
# positive_anchors = anchors[positive_anchor_ix]
# negative_anchors = anchors[negative_anchor_ix]
# neutral_anchors = anchors[neutral_anchor_ix]
# log("positive_anchors", positive_anchors)
# log("negative_anchors", negative_anchors)
# log("neutral anchors", neutral_anchors)
#
# # Apply refinement deltas to positive anchors
# refined_anchors = utils.apply_box_deltas(
#     positive_anchors,
#     target_rpn_bbox[:positive_anchors.shape[0]] * model.config.RPN_BBOX_STD_DEV)
# log("refined_anchors", refined_anchors, )
#
# Display positive anchors before refinement (dotted) and
# after refinement (solid).
# visualize.draw_boxes(
#     image, ax=get_ax(),
#     boxes=positive_anchors,
#     refined_boxes=refined_anchors)

    # Run RPN sub-graph
#pillar = model.keras_model.get_layer("ROI").output  # node to start searching from
#
# # TF 1.4 and 1.9 introduce new versions of NMS. Search for all names to support TF 1.3~1.10
# nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression:0")
# if nms_node is None:
#     nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV2:0")
# if nms_node is None: #TF 1.9-1.10
#     nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV3:0")
#
# rpn = model.run_graph(image[np.newaxis], [
#     ("rpn_class", model.keras_model.get_layer("rpn_class").output),
#     ("pre_nms_anchors", model.ancestor(pillar, "ROI/pre_nms_anchors:0")),
#     ("refined_anchors", model.ancestor(pillar, "ROI/refined_anchors:0")),
#     ("refined_anchors_clipped", model.ancestor(pillar, "ROI/refined_anchors_clipped:0")),
#     ("post_nms_anchor_ix", nms_node),
#     ("proposals", model.keras_model.get_layer("ROI").output),
# ], image_metas=image_meta[np.newaxis])
#
# # Show top anchors by score (before refinement)
# limit = 100
# sorted_anchor_ids = np.argsort(rpn['rpn_class'][:,:,1].flatten())[::-1]
# # visualize.draw_boxes(image, boxes=anchors[sorted_anchor_ids[:limit]], ax=get_ax())
#
# # Show top anchors with refinement. Then with clipping to image boundaries
# limit = 50
# ax = get_ax(1, 2)
# pre_nms_anchors = utils.denorm_boxes(rpn["pre_nms_anchors"][0], image.shape[:2])
# refined_anchors = utils.denorm_boxes(rpn["refined_anchors"][0], image.shape[:2])
# refined_anchors_clipped = utils.denorm_boxes(rpn["refined_anchors_clipped"][0], image.shape[:2])
# visualize.draw_boxes(image, boxes=pre_nms_anchors[:limit],
#                      refined_boxes=refined_anchors[:limit], ax=ax[0])
# visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[:limit], ax=ax[1])
#
# # Show refined anchors after non-max suppression
# limit = 50
# ixs = rpn["post_nms_anchor_ix"][:limit]
# visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[ixs], ax=get_ax())
#
# # Show final proposals
# # These are the same as the previous step (refined anchors
# # after NMS) but with coordinates normalized to [0, 1] range.
# limit = 50
# # Convert back to image coordinates for display
# # h, w = config.IMAGE_SHAPE[:2]
# # proposals = rpn['proposals'][0, :limit] * np.array([h, w, h, w])
# visualize.draw_boxes(
#     image, ax=get_ax(),
#     refined_boxes=utils.denorm_boxes(rpn['proposals'][0, :limit], image.shape[:2]))
#
#     # Get input and output to classifier and mask heads.
# mrcnn = model.run_graph([image], [
#     ("proposals", model.keras_model.get_layer("ROI").output),
#     ("probs", model.keras_model.get_layer("mrcnn_class").output),
#     ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),
#     ("masks", model.keras_model.get_layer("mrcnn_mask").output),
#     ("detections", model.keras_model.get_layer("mrcnn_detection").output),
# ])
#
# # Get detection class IDs. Trim zero padding.
# det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
# det_count = np.where(det_class_ids == 0)[0][0]
# det_class_ids = det_class_ids[:det_count]
# detections = mrcnn['detections'][0, :det_count]
#
# print("{} detections: {}".format(
#     det_count, np.array(dataset.class_names)[det_class_ids]))
#
# captions = ["{} {:.3f}".format(dataset.class_names[int(c)], s) if c > 0 else ""
#             for c, s in zip(detections[:, 4], detections[:, 5])]
# visualize.draw_boxes(
#     image,
#     refined_boxes=utils.denorm_boxes(detections[:, :4], image.shape[:2]),
#     visibilities=[2] * len(detections),
#     captions=captions, title="Detections",
#     ax=get_ax())
#
#     # Proposals are in normalized coordinates
# proposals = mrcnn["proposals"][0]
#
# # Class ID, score, and mask per proposal
# roi_class_ids = np.argmax(mrcnn["probs"][0], axis=1)
# roi_scores = mrcnn["probs"][0, np.arange(roi_class_ids.shape[0]), roi_class_ids]
# roi_class_names = np.array(dataset.class_names)[roi_class_ids]
# roi_positive_ixs = np.where(roi_class_ids > 0)[0]
#
# # How many ROIs vs empty rows?
# print("{} Valid proposals out of {}".format(np.sum(np.any(proposals, axis=1)), proposals.shape[0]))
# print("{} Positive ROIs".format(len(roi_positive_ixs)))
#
# # Class counts
# print(list(zip(*np.unique(roi_class_names, return_counts=True))))
#
# # Display a random sample of proposals.
# # Proposals classified as background are dotted, and
# # the rest show their class and confidence score.
# limit = 200
# ixs = np.random.randint(0, proposals.shape[0], limit)
# captions = ["{} {:.3f}".format(dataset.class_names[c], s) if c > 0 else ""
#             for c, s in zip(roi_class_ids[ixs], roi_scores[ixs])]
# visualize.draw_boxes(
#     image,
#     boxes=utils.denorm_boxes(proposals[ixs], image.shape[:2]),
#     visibilities=np.where(roi_class_ids[ixs] > 0, 2, 1),
#     captions=captions, title="ROIs Before Refinement",
#     ax=get_ax())
#
#     # Class-specific bounding box shifts.
# roi_bbox_specific = mrcnn["deltas"][0, np.arange(proposals.shape[0]), roi_class_ids]
# log("roi_bbox_specific", roi_bbox_specific)
#
# # Apply bounding box transformations
# # Shape: [N, (y1, x1, y2, x2)]
# refined_proposals = utils.apply_box_deltas(
#     proposals, roi_bbox_specific * config.BBOX_STD_DEV)
# log("refined_proposals", refined_proposals)
#
# # Show positive proposals
# # ids = np.arange(roi_boxes.shape[0])  # Display all
# limit = 5
# ids = np.random.randint(0, len(roi_positive_ixs), limit)  # Display random sample
# captions = ["{} {:.3f}".format(dataset.class_names[c], s) if c > 0 else ""
#             for c, s in zip(roi_class_ids[roi_positive_ixs][ids], roi_scores[roi_positive_ixs][ids])]
# visualize.draw_boxes(
#     image, ax=get_ax(),
#     boxes=utils.denorm_boxes(proposals[roi_positive_ixs][ids], image.shape[:2]),
#     refined_boxes=utils.denorm_boxes(refined_proposals[roi_positive_ixs][ids], image.shape[:2]),
#     visibilities=np.where(roi_class_ids[roi_positive_ixs][ids] > 0, 1, 0),
#     captions=captions, title="ROIs After Refinement")
#
#     # Remove boxes classified as background
# keep = np.where(roi_class_ids > 0)[0]
# print("Keep {} detections:\n{}".format(keep.shape[0], keep))
#
# # Remove low confidence detections
# keep = np.intersect1d(keep, np.where(roi_scores >= config.DETECTION_MIN_CONFIDENCE)[0])
# print("Remove boxes below {} confidence. Keep {}:\n{}".format(
#     config.DETECTION_MIN_CONFIDENCE, keep.shape[0], keep))
#
#     # Apply per-class non-max suppression
# pre_nms_boxes = refined_proposals[keep]
# pre_nms_scores = roi_scores[keep]
# pre_nms_class_ids = roi_class_ids[keep]
#
# nms_keep = []
# for class_id in np.unique(pre_nms_class_ids):
#     # Pick detections of this class
#     ixs = np.where(pre_nms_class_ids == class_id)[0]
#     # Apply NMS
#     class_keep = utils.non_max_suppression(pre_nms_boxes[ixs],
#                                             pre_nms_scores[ixs],
#                                             config.DETECTION_NMS_THRESHOLD)
#     # Map indicies
#     class_keep = keep[ixs[class_keep]]
#     nms_keep = np.union1d(nms_keep, class_keep)
#     print("{:22}: {} -> {}".format(dataset.class_names[class_id][:20],
#                                    keep[ixs], class_keep))
#
# keep = np.intersect1d(keep, nms_keep).astype(np.int32)
# print("\nKept after per-class NMS: {}\n{}".format(keep.shape[0], keep))
#
# # Show final detections
# ixs = np.arange(len(keep))  # Display all
# # ixs = np.random.randint(0, len(keep), 10)  # Display random sample
# captions = ["{} {:.3f}".format(dataset.class_names[c], s) if c > 0 else ""
#             for c, s in zip(roi_class_ids[keep][ixs], roi_scores[keep][ixs])]
# visualize.draw_boxes(
#     image,
#     boxes=utils.denorm_boxes(proposals[keep][ixs], image.shape[:2]),
#     refined_boxes=utils.denorm_boxes(refined_proposals[keep][ixs], image.shape[:2]),
#     visibilities=np.where(roi_class_ids[keep][ixs] > 0, 1, 0),
#     captions=captions, title="Detections after NMS",
#     ax=get_ax())
#
# limit = 8
# display_images(np.transpose(gt_mask[..., :limit], [2, 0, 1]), cmap="Blues")
#
# # Get predictions of mask head
# mrcnn = model.run_graph([image], [
#     ("detections", model.keras_model.get_layer("mrcnn_detection").output),
#     ("masks", model.keras_model.get_layer("mrcnn_mask").output),
# ])
#
# # Get detection class IDs. Trim zero padding.
# det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
# det_count = np.where(det_class_ids == 0)[0][0]
# det_class_ids = det_class_ids[:det_count]
#
# print("{} detections: {}".format(
#     det_count, np.array(dataset.class_names)[det_class_ids]))
#
#     # Masks
# det_boxes = utils.denorm_boxes(mrcnn["detections"][0, :, :4], image.shape[:2])
# det_mask_specific = np.array([mrcnn["masks"][0, i, :, :, c]
#                               for i, c in enumerate(det_class_ids)])
# det_masks = np.array([utils.unmold_mask(m, det_boxes[i], image.shape)
#                       for i, m in enumerate(det_mask_specific)])
# log("det_mask_specific", det_mask_specific)
# log("det_masks", det_masks)
#
# display_images(det_mask_specific[:4] * 255, cmap="Blues", interpolation="none")
#
# display_images(det_masks[:4] * 255, cmap="Blues", interpolation="none")
#
# # Get activations of a few sample layers
# activations = model.run_graph([image], [
#     ("input_image",        model.keras_model.get_layer("input_image").output),
#     ("res2c_out",          model.keras_model.get_layer("res2c_out").output),
#     ("res3c_out",          model.keras_model.get_layer("res3c_out").output),
#     ("rpn_bbox",           model.keras_model.get_layer("rpn_bbox").output),
#     ("roi",                model.keras_model.get_layer("ROI").output),
# ])
#
# # Backbone feature map
# display_images(np.transpose(activations["res2c_out"][0,:,:,:4], [2, 0, 1]), cols=4)