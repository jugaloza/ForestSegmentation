# ForestSegmentation

This repository contains my work on Forest Segmentation using UNet architecture.

# Dataset Details

I have taken Dataset for this problem from Kaggle. Link given below

<a href=https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation />

This dataset consists of 5108 images with their respective binary mask of forest from aerial view.

# Architecture Details

I have referred UNet architecture for this problem.

You can refer research paper from here : <a href=https://paperswithcode.com/paper/u-net-convolutional-networks-for-biomedical />

# Loss Function

Because this is a task of binary segmentation, I have used binary cross entropy loss for this problem to evaluate performance.

# Evaluation Metric


I have used Binary Jaccard Index as evaluation metric for evaulating model.

# Model Optimization

For model optimization, I used TensorRT for faster inference on GPU with PyCUDA.

# Inference Results

## Inference Result 1

![Pred_mask_01](https://github.com/jugaloza/ForestSegmentation/assets/23360267/9c8422fc-073e-42c7-8b39-26f9849a1653)

## Inference Result 2

![Pred_mask_02](https://github.com/jugaloza/ForestSegmentation/assets/23360267/5ef6817e-9e52-41d6-910f-79d146e89277)
