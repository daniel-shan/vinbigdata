#!/usr/bin/env bash

USE_S3=${1}
S3_BUCKET_NAME="default"
CONFIG_PATH="/home/ec2-user/config.py"

MMDETECTION_VERSION="v2.10.0"
CUDA_VERSION="110"
TORCH_VERSION="1.7.1"

OUTPUT_DIR="/home/ec2-user/vinbig_output"
INPUT_DIR="/home/ec2-user/input"
CHECKPOINTS_DIR="/home/ec2-user/checkpoints"

CONFIG_S3_PATH="s3://${S3_BUCKET_NAME}/vfnet_config.py"
DATASET_S3_PATH="s3://${S3_BUCKET_NAME}/vinbigdata-coco-dataset-with-wbf/"
CHECKPOINTS_S3_PATH="s3://${S3_BUCKET_NAME}/vfnet-checkpoints/"

# install mmcv
pip install mmcv-full -f "https://download.openmmlab.com/mmcv/dist/cu${CUDA_VERSION}/torch${TORCH_VERSION}/index.html"
pip install albumentations

# download data
if [[ ! -f "~/.aws/credentials" ]]; then
    echo "aws credentials have not been set up, please run aws configure before continuing."
    exit 1
fi
mkdir ${OUTPUT_DIR}
mkdir ${INPUT_DIR}
if [[ ${USE_S3} == "TRUE" ]]; then
    aws s3 cp ${DATASET_S3_PATH} ${INPUT_DIR} --recursive
fi

# download config and checkpoints
mkdir ${CHECKPOINTS_DIR}
if [[ ${USE_S3} == "TRUE" ]]; then
    aws s3 cp ${CHECKPOINTS_S3_PATH} ${CHECKPOINTS_DIR} --recursive
    aws s3 cp ${CONFIG_S3_PATH} ${CONFIG_PATH}
fi


# install mmdetection
cd ~
git clone --branch ${MMDETECTION_VERSION} https://github.com/open-mmlab/mmdetection.git

cd mmdetection
pip install -e .

echo "Setup complete."
echo "Starting training script..."

python tools/train.py ${CONFIG_PATH} ${CHECKPOINTS_DIR}/vfnet.pth
