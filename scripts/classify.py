import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import  tensorflow as tf
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
import numpy as np
import skimage.io
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img,ImageDataGenerator
from keras import optimizers
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
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
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

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)
    # f=open("/data1/wyj/M/datasets/dataset-master/labels.csv", "rb")


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
    # model = resnet101(weights='imagenet', include_top=True)
    model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
    model.summary()
    # x=model0.output
    # # x = GlobalAveragePooling2D()(x)
    # # # x = Flatten()(x)
    # # # 添加一个全连接层
    # # x = Dense(1024, activation='relu')(x)
    # #
    # # # 添加一个分类器，假设我们有2个类
    # # predictions = Dense(5, activation='softmax')(x)
    # model = Model(inputs=model0.input, outputs=predictions)
    # sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd,#'rmsprop',
    #               loss='categorical_crossentropy',
    #               metrics=[metrics.mae, metrics.categorical_accuracy,metric_precision,metric_recall,metric_F1score,precision,recall,fmeasure,fbeta_score])
    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    # weights_path = COCO_WEIGHTS_PATH
    # if int(args.LOOP)>0:
    #     weights_path = '../../best.h5'
    # if not os.path.exists(args.LRPTS_DIR):
    #     os.mkdir(args.LRPTS_DIR)
    # tpc=os.path.join(args.LRPTS_DIR,'TS_of_loop'+str(args.LOOP))
    # tp=os.path.join(args.LRPTS_DIR,'TS_of_loop'+str(int(args.LOOP)-1)+'/')
    # if not os.path.exists(tpc):
    #     os.mkdir(tpc)
    # print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True)
    # else:
    # model.load_weights(weights_path, by_name=True)
    model.summary()
    # model.uses_learning_phase=True
    #
    # model=modellib.remodel(model)
    # for layer in model.layers[:-3     z]:
    #     layer.trainable = False
    train_datagen = ImageDataGenerator(
        horizontal_flip=True)

    test_datagen = ImageDataGenerator()
    # train_generator = train_datagen.flow_from_directory(
    #     NP,
    #     target_size=(2000, 2000),
    #     batch_size=10)
    # train_generator=train_datagen.flow(alldata[:300],labs[:300],batch_size=10)
    # validation_generator = test_datagen.flow(alldata[300:],labs[300:],batch_size=10)
    #MaskrcnnPath = '../../logs/prcc/resnet_PRCC10.h5'

    # model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=4000,
    #     epochs=5,
    #     validation_data=validation_generator,
    #     validation_steps=10)
    # model.save('resnet_PRCC5.h5')

    # score = model.evaluate_generator(generator=validation_generator,
    #                                  workers=1,
    #                                  use_multiprocessing=False,
    #                                  verbose=0)
    #
    # print('%s: %.2f' % (model.metrics_names[0], score[0]))  # Loss
    # print('%s: %.2f%%' % (model.metrics_names[1], score[1] * 100))  # metrics1
    # print('%s: %.2f%%' % (model.metrics_names[2], score[2] * 100))  # metrics2
    # print('%s: %.2f%%' % (model.metrics_names[3], score[3] * 100))  # metrics1

    # names = [weight.name for layer in model.layers for weight in layer.weights]
    # weights = model.get_weights()
    # for name, weight in zip(names, weights):
    #     print(name, weight.shape)
    # MaskrcnnPath = '../../logs/prcc/resnet_PRCC0507vgg3.h5'
    # MaskrcnnPath = '../../logs/prcc/resnet_PRCC0508vgg1.h5'
    # model.load_weights(os.path.abspath(MaskrcnnPath), by_name=True)
    if args.command == "train":
        # for it in range(100):
        #     print('iter:::::::::{}:::::::::'.format(it))
        #     model.fit_generator(
        #         train_generator,
        #         steps_per_epoch=4000,
        #         epochs=1,
        #         validation_data=validation_generator,
        #         validation_steps=10)
        #     model.save('../../logs/prcc/resnet_PRCC0508vgg{}.h5'.format(it))
            # model.save(os.path.join(args.LRPTS_DIR,'classification_model_of_loop_'+str(args.LOOP)+'.h5'))
        score = model.evaluate_generator(generator=train_generator,
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
    if args.command == "detect":
        # MaskrcnnPath = '../../logs/prcc/resnet_PRCC0508vgg1.h5'
        # model.load_weights(os.path.abspath(MaskrcnnPath), by_name=True)
        orig_imgs=[]
        oser=os.listdir("/data1/wyj/M/datasets/T0815/")
        for fname in oser:
            print("/data1/wyj/M/datasets/T0815/{}".format(fname))
            orig_imgs.append(img_to_array(load_img("/data1/wyj/M/datasets/T0815/{}".format(fname), target_size=(224, 224))))
        orig_imgs=np.array(orig_imgs)
        predr=model.predict(np.array(orig_imgs))
        for i in range(len(oser)):
            ks = print(np.argmax(predr[i, :]))