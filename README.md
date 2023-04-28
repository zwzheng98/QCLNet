## Quaternion-valued Correlation Learning for Few-Shot Semantic Segmentation
This is the implementation of our paper [**Quaternion-valued Correlation Learning for Few-Shot Semantic Segmentation**](https://ieeexplore.ieee.org/document/9954424) that has been accepted to IEEE Transactions on Circuits and Systems for Video Technology (TCSVT). 

<p align="middle">
    <img src="figure/Figure2.jpg">
</p>

## Requirements

- Python 3.7
- PyTorch 1.5.1
- cuda 10.1
- tensorboard 1.14

## Datasets

- PASCAL-5<sup>i</sup>:  [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) + [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)

- COCO-20<sup>i</sup>:  [COCO2014](https://cocodataset.org/#download)

  Please see [OSLSM](https://arxiv.org/abs/1709.03410) and [FWB](https://openaccess.thecvf.com/content_ICCV_2019/html/Nguyen_Feature_Weighting_and_Boosting_for_Few-Shot_Segmentation_ICCV_2019_paper.html) for more details on datasets. 

## Directory Structure
Create a directory '../Datasets_HSN' for the above two few-shot segmentation datasets and appropriately place each dataset to have the following directory structure:

    ../                         # parent directory
    ├── ./                      # current (project) directory
    │   ├── common/             
    │   ├── data/               
    │   ├── model/              # (dir.) implementation of Quaternion-valued Correlation Learning Network model 
    │   ├── README.md         
    │   ├── train.py            # code for training QCLNet
    │   └── test.py             # code for testing QCLNet
    └── Datasets_HSN/
        ├── VOC2012/            # PASCAL VOC2012 devkit
        │   ├── Annotations/
        │   ├── ImageSets/
        │   ├── ...
        │   └── SegmentationClassAug/
        └── COCO2014/           
            ├── annotations/
            │   ├── train2014/   
            │   ├── val2014/    
            │   └── ..some json files..
            ├── train2014/
            └── val2014/
## Test and Train
Using PASCAL-5i as an example
### Training
> #### PASCAL-5<sup>i</sup>
> ```bash
> python train.py --backbone {vgg16, resnet50, resnet101}  
>                 --fold {0, 1, 2, 3} 
>                 --benchmark pascal
>                 --lr 1e-3
>                 --bsz 20
>                 --logpath "your_experiment_name"
> ```
> * Take a look at train.py 's main function, where you can set different parameters
> * When training another dataset, you only need to change the corresponding parameters

### Testing

> #### PASCAL-5<sup>i</sup>
> Load the trained model weights and start testing
> ```bash
> python test.py --backbone {vgg16, resnet50, resnet101}  
>                --fold {0, 1, 2, 3} 
>                --benchmark pascal
>                --nshot {1, 5} 
>                --load "path_to_trained_model/best_model.pt"
> ```

## References

This repo is mainly built based on [HSNet](https://github.com/juhongm999/hsnet). Thanks for their great work!

# Citation

If you find this project useful, please consider citing:
```
@article{zheng2022qclnet,
  title={Quaternion-valued Correlation Learning for Few-Shot Semantic Segmentation},
  author={Zheng, Zewen and Huang, Guoheng and Yuan, Xiaochen and Pun, Chi-Man and Liu, Hongrui and Ling, Wing-Kuen},
  journal={TCSVT},
  year={2022}
}
```
