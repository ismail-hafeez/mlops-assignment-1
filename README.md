## MLOps Assignment 1

This repository contains the implementation of a basic ML pipeline on AWS EC2 with EBS persistence and scheduled auto-shutdown.

## EC2 Configuration
```text
Instance Type: t2.micro
Region: us-west-1
```

## Create S3 Bucket
```text
bucket name: mlops-5981
```
Dataset uploaded to S3 and synced to EC2.

## Create IAM Role
Role Name: mlops-s3-role
Permissions:
- AmazonS3FullAccess
- ec2:StopInstances (inline policy)

## Execution Steps

1) Create and launch EC2 instance and download .pem file
2) SSH into EC2
```bash
ssh -i "mlops-assignment.pem" ubuntu@<public-ip>
```
3) Run the following command to set up the environment:
```bash
./setup_ml_env.sh
```
4) Activate venv
```bash
source /mnt/ml-data/venv/bin/activate
```
5) Run the following command to train the model:
```bash
python train_pipeline.py
```
This will train the model and evaluate its accuracy.
6) Verify model saved
```bash
ls /mnt/ml-data/models/

# Verify logs saved
ls /mnt/ml-data/logs/

# Verify cron job
crontab -l
```

## Results

The results will be printed to the console, and the best model will be saved to the S3 bucket.

```text
Model: RandomForest
Accuracy: 0.84475
Timestamp: 2026-02-14 18:40:22
```