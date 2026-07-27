#!/bin/bash

################################################################################
# Project      : Yelp Business Intelligence Platform
# Layer        : Bronze (Raw Data Ingestion)
# Author       : Nagesh Khichade
# Description  : Downloads Yelp dataset from Kaggle and uploads JSON files
#                into the Bronze Layer (Amazon S3).
################################################################################

set -e

############################
# Configuration
############################

BUCKET_NAME="my-yelp-project-bucket"

DATASET="adamamer2001/yelp-complete-open-dataset-2024"

WORK_DIR="/mnt/ec2-user"

ZIP_FILE="${WORK_DIR}/yelp-complete-open-dataset-2024.zip"

EXTRACT_DIR="${WORK_DIR}/yelp_dataset"

RAW_LAYER="s3://${BUCKET_NAME}/raw"

echo "=========================================================="
echo "Yelp Dataset Bronze Layer Ingestion Started"
echo "=========================================================="

############################
# Step 1 : Verify AWS CLI
############################

echo "[1/10] Checking AWS Access..."

aws s3 ls > /dev/null

echo "AWS connection successful."

############################
# Step 2 : Create Working Directory
############################

echo "[2/10] Creating working directory..."

sudo mkdir -p ${WORK_DIR}
sudo chown ec2-user:ec2-user ${WORK_DIR}

############################
# Step 3 : Download Dataset
############################

echo "[3/10] Downloading dataset from Kaggle..."

cd ${WORK_DIR}

if [ ! -f "${ZIP_FILE}" ]; then
    kaggle datasets download -d ${DATASET}
else
    echo "Dataset already downloaded."
fi

############################
# Step 4 : Extract Dataset
############################

echo "[4/10] Extracting archive..."

mkdir -p ${EXTRACT_DIR}

unzip -o ${ZIP_FILE} -d ${EXTRACT_DIR}

############################
# Step 5 : Upload ZIP Backup
############################

echo "[5/10] Uploading ZIP backup..."

aws s3 cp \
${ZIP_FILE} \
${RAW_LAYER}/

############################
# Step 6 : Upload Business
############################

echo "[6/10] Uploading Business Dataset..."

aws s3 cp \
${EXTRACT_DIR}/yelp_dataset/yelp_academic_dataset_business.json \
${RAW_LAYER}/

############################
# Step 7 : Upload Review
############################

echo "[7/10] Uploading Review Dataset..."

aws s3 cp \
${EXTRACT_DIR}/yelp_dataset/yelp_academic_dataset_review.json \
${RAW_LAYER}/

############################
# Step 8 : Upload User
############################

echo "[8/10] Uploading User Dataset..."

aws s3 cp \
${EXTRACT_DIR}/yelp_dataset/yelp_academic_dataset_user.json \
${RAW_LAYER}/

############################
# Step 9 : Upload Tip, Checkin & Photos
############################

echo "[9/10] Uploading Remaining Datasets..."

aws s3 cp \
${EXTRACT_DIR}/yelp_dataset/yelp_academic_dataset_tip.json \
${RAW_LAYER}/

aws s3 cp \
${EXTRACT_DIR}/yelp_dataset/yelp_academic_dataset_checkin.json \
${RAW_LAYER}/


############################
# Step 10 : Verification
############################

echo "[10/10] Verifying uploaded files..."

aws s3 ls ${RAW_LAYER}/

echo ""
echo "=========================================================="
echo "Bronze Layer Ingestion Completed Successfully"
echo "=========================================================="
