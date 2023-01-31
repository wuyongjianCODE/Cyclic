import shutil
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("--iteration_num",
                        metavar="number",
                        help="number")
    # parser.add_argument('--dataset', required=False,
    #                     metavar="/path/to/dataset/",
    #                     help='Root directory of the dataset')
    # parser.add_argument('--weights', required=True,
    #                     metavar="/path/to/weights.h5",
    #                     help="Path to weights .h5 file or 'coco'")
    # parser.add_argument('--logs', required=False,
    #                     default=DEFAULT_LOGS_DIR,
    #                     metavar="/path/to/logs/",
    #                     help='Logs and checkpoints directory (default=logs/)')
    # parser.add_argument('--subset', required=False,
    #                     metavar="Dataset sub-directory",
    #                     help="Subset of dataset to run prediction on")
    # parser.add_argument('--iteration', required=False,
    #                     default=0,
    #                     metavar="the iteration num",
    #                     help='as shown')
    args = parser.parse_args()

    # Validate arguments
    # if args.command == "train":
    #     assert args.dataset, "Argument --dataset is required for training"
    # elif args.command == "detect":
    #     assert args.subset, "Provide --subset to run prediction on"
    src=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results/iteration_demo.png'
    shutil.move(src,r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\results/iteration_'+str(args.iteration_num)+'.png')