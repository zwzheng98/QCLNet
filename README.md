## Quaternion-valued Correlation Learning for Few-Shot Semantic Segmentation
This is the implementation of our paper [**Quaternion-valued Correlation Learning for Few-Shot Semantic Segmentation**](https://ieeexplore.ieee.org/document/9954424?source=authoralert) that has been accepted to IEEE Transactions on Circuits and Systems for Video Technology (TCSVT). 


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


## Test and Train
Using PASCAL-5i as an example

### Training
> #### PASCAL-5<sup>i</sup>
> ```bash
> python train.py  backbone=$BACKBONE$ fold=$FOLD$  dataset=$DATASET$  batch_size=$BATCH_SIZE$  
> ```

### Testing

> #### PASCAL-5<sup>i</sup>
> ```bash
> python test.py  backbone=$BACKBONE$ fold=$FOLD$  dataset=$DATASET$  batch_size=$BATCH_SIZE$  load=$BEST_MODEL_PTH$
> ```

## Visualization
<p align="middle">
    <img src="figure/vis.png">
</p>

## References

This repo is mainly built based on [HSNet](https://github.com/juhongm999/hsnet). Thanks for their great work!

# Citation

If you find this project useful, please consider citing:
```
@article{zheng2022qclnet,
  title={Quaternion-valued Correlation Learning for Few-Shot Semantic Segmentation},
  author={Zheng, Zewen and Huang, Guoheng and Yuan, Xiaochen and Pun, Chi-Man and Liu, Hongrui and Ling, Wing-Kuen},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2023}
}
```
