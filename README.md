# Cyclic Learning: Bridging Image-level Labels and Nuclei Instance Segmentation
Official implementation of Cyclic Learning: Bridging Image-level Labels and Nuclei Instance Segmentation
The original paper link is here:
[arXiv link](to be update), [TMI link](to be update)

Nuclei instance segmentation on histopathology images is of great clinical value for disease analysis. Generally, fully-supervised algorithms for this task require pixel-wise manual annotations, which is especially time-consuming and laborious for the high nuclei density. To alleviate the annotation burden, we seek to solve the problem through image-level weakly supervised learning, which has not been achieved by any previous work. Compared with most existing methods using other weak annotations (scribble, point, etc.) for nuclei instance segmentation, our method is more labor-saving. The obstacle to using image-level annotations in nuclei instance segmentation is the lack of adequate location information, leading to severe nuclei omission or overlaps.Cyclic learning comprises a front-end classification task and a back-end semi-supervised instance segmentation task to benefit from multi-task learning (MTL). We utilize the interpretability of a CNN as the front-end to convert image-level labels to sets of high-confidence pseudo masks and establish a semi-supervised architecture as the back-end to conduct nuclei instance segmentation under the supervision of these pseudo masks. Most importantly, cyclic learning is designed to circularly share knowledge between the front-end CNN and the back-end semi-supervised part, which allows the whole system to fully extract the underlying information from image-level labels and converge to a better optimum. Experiments on Three datasets validate that our method outperforms other image-level weakly supervised methods on nuclei instance segmentation and is close to the fully-supervised one in performance.

## Installation

- Our project is developed on [Mask_RCNN](https://github.com/mssatterport/Mask_RCNN).
Please first build up this project, then put our code repository under the directory `Mask_RCNN/samples/`.

- Create an environment meets the requirements as listed in `requirements.txt`

## Data Preparation
- Download the [Monusac dataset](https://pan.baidu.com/s/1ALRjHBQ7LwY-stIW1NzMRA?pwd=mseg) (pwd：mseg) and [cropped Monusac dataset](https://pan.baidu.com/s/1D9F1pLcu2bHwglE1oafmZA?pwd=mseg) (pwd : mseg), and put it in the `Mask_RCNN/datasets/` directory.

- Download the [ccrcc dataset](https://pan.baidu.com/s/1RiuaRxxgXWEa2wNYf58bmw?pwd=mseg)
  (pwd：mseg), and put it in the `Mask_RCNN/datasets/` directory.

- Download the [consep dataset](https://pan.baidu.com/s/1zPPOQI9ZTKpvTlNkePIxmw?pwd=mseg) (pwd：mseg), and put it in the `Mask_RCNN/datasets/` directory.
- Download the positive-and-negative nucleus image [classification dataset](https://pan.baidu.com/s/1CjcIfT2k92gmaLW17noFMw?pwd=mseg) (pwd : mseg) which is obtained by cropping out tile images from TCGA WSI(whole slide image). 
Datasets are organized in the following way:
```bazaar
datasets/
    MyNP/
        negative/
        positive/
    MoNuSACGT/
    MoNuSACCROP/
        stage1_train/
        images/
        masks/
    ccrcccrop/    
        Test/
        Train/
        Valid/
    consepcrop/
        Test/
        Train/
        Valid/
```


## Training And Testing
Before training, please download pretrain weights of big nature image datasets, for which we use [COCO pretrain weights](https://cocodataset.org/#home). Remember to change the path in the code.
```bash 
python nucleus_recursion_50.py
```
Training Cyclic Learning on ccrcc dataset:
```bash 
python nucleus_recursion_ccrcc.py
```
Training Cyclic Learning on consep dataset:
```bash 
python nucleus_recursion_consep.py
```
## Citing Cyclic Learning
If you use Cyclic Learning in your work or wish to refer to the results published in this repo, please cite our paper:
```BibTeX

```




