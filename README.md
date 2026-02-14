## MLOps Assignment 1

This repository contains the code for the MLOps Assignment 1.

## Create S3 Bucket
```text
bucket name: mlops-5981
```

## Create IAM Role
```text
Role name: mlops-s3-role
```

## Execution Steps

1) Upload the dataset to the S3 bucket.
2) Run the following command to set up the environment:
```bash
./setup_ml_env.sh
```
3) Run the following command to train the model:
```bash
python train_pipeline.py
```

This will train the model and evaluate its accuracy.

## Results

The results will be printed to the console, and the best model will be saved to the S3 bucket.

```text
Model: RandomForest
Accuracy: 0.84475
Timestamp: 2026-02-14 18:40:22
----------------------------------------
```