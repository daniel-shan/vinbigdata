# VinBigData Chest X-ray Abnormalities Detection
This is my solution for the [VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection) competition hosted on Kaggle.  This solution placed 36th (out of 1329) with a final evaluation metric of .270 mAP.

## Overview
Overall, the solution is simple.  Two VFNet models and two YOLOv5 models were trained to detect abnormalities.  The predictions were then aggregated and passed through a two-class "Finding / No Finding" classifier to remove false positives.

## Hardware
Training for VFNet was done on AWS (1xV100) on the Amazon Linux 2 Deep Learning v42 AMI (ami-001a383ed04e9e6a1).

Training for YOLOv5 was done on Google Colab (1xP100 or 1xV100).

Inference for both models was run on Kaggle.

## Data
The VFNet models were trained on 2x downsampled images from [this dataset](https://www.kaggle.com/sreevishnudamodaran/vinbigdata-fusing-bboxes-coco-dataset)<sup>1</sup>.

The YOLOv5 models were trained on 1024x1024 images from [this dataset](https://www.kaggle.com/xhlulu/vinbigdata-chest-xray-resized-png-1024x1024)<sup>2</sup> with labels from [this dataset](https://www.kaggle.com/awsaf49/vinbigdata-yolo-labels-dataset)<sup>3</sup>.

## Preprocessing
The only preprocessing done was to remove all "No Finding" images and then apply WBF<sup>1,4</sup> @ IoU=.4  

## Training
The VFNet models can be trained by running:
```bash
./vfnet/train_vfnet.sh ${USE_S3}
```
If `USE_S3` is `TRUE`, the values of `S3_BUCKET_NAME` and `DATASET_S3_PATH`, etc. can be changed in `train_vfnet.sh` to download data from S3.  Alternatively, `USE_S3` can be set to `FALSE` to manually put the data in the `~/input/` directory according to `~/vfnet/config.py`:

```python
data = dict(
...
    train=dict(
        type='CocoDataset',
        ann_file='/home/ec2-user/input/train_annotations.json',
```

The YOLOv5 models can be trained by simply running the Colab notebook and either changing the S3 buckets or manually uploading the data.

## Inference
All inference was done on Kaggle to simplify submission.  The notebooks used for submission are included here.

## Postprocessing
The four models were ensembled by aggregating predictions per image and applying WBF @ IoU=.5

I trained a two class classifer primarily to remove false positives.  However, the results were not as good as those provided by @awsaf, so I used his results<sup>5</sup> as a class filter instead. 

## Credits
1. [@sreevishnudamodaran WBF + COCO Kernel](https://www.kaggle.com/sreevishnudamodaran/vinbigdata-fusing-bboxes-coco-dataset)
2. [@xhlulu 1024x1024 images dataset](https://www.kaggle.com/xhlulu/vinbigdata-chest-xray-resized-png-1024x1024)
3. [@awsaf49 YOLO labels](https://www.kaggle.com/awsaf49/vinbigdata-yolo-labels-dataset) 
4. [@ZFTurbo WBF implementation](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
5. [@awsaf two class filter](https://www.kaggle.com/awsaf49/vinbigdata-2class-prediction)
6. [YOLOv5](https://github.com/ultralytics/yolov5)
7. [VarifocalNet](https://github.com/hyz-xmaster/VarifocalNet)
