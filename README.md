# ISIC 2024 - Skin Cancer Detection with 3D-TBP

> Identify cancers among skin lesions cropped from 3D total body photographs

[![License](https://img.shields.io/badge/license-CC_BY--NC_4.0-green)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Lightning](https://img.shields.io/badge/lightning-v2.3.3-red)](#lightning)
[![Pytorch](https://img.shields.io/badge/pytorch-v2.3.1-blue)](#pytorch)
[![Kubeflow](https://img.shields.io/badge/kubeflow-v1.8-orange)](#kubeflow)

This repository contains the code to train a Machine Learning (ML) model on Kubeflow for the
[ISIC 2024 - Skin Cancer Detection with 3D-TBP](https://www.kaggle.com/competitions/isic-2024-challenge)
Kaggle competition.

In this competition, the goal is to develop image-based algorithms to identify histologically
confirmed skin cancer cases using single-lesion crops from 3D total body photos (TBP). The image
quality is similar to close-up smartphone photos, which are often submitted for telehealth purposes.

The binary classification algorithm developed through this project could be used in settings without
access to specialized care, improving triage for early skin cancer detection.

The example code demonstrates how to train a small EfficientNet model variant. The aim is not to
achieve the best accuracy, but to establish a framework for training ML models on Kubeflow in a
distributed manner. This use case can serve as a tutorial or demo for validating the
[VirtML Project](https://github.com/dpoulopoulos/virtml).

## What You'll Need

To run the code in this repository, ensure you have:

* A working Kubeflow deployment. Visit the [VirtML](https://github.com/dpoulopoulos/virtml) project
  page to find out how you can create a local Kubeflow deployment.

## Procedure

To train your own model, follow the steps below:

1. Login to your Kubeflow cluster, using your credentials.
1. Create a new Notebook server and enable access to Kubeflow Pipelines.
1. Connect to the Notebook server, launch a new terminal window, and clone the repository locally.
1. Install the demo prerequisites:
   ```
   pip install -r requirements.txt
   ```
1. Launch the two Notebooks in order and execute the code cells. Select the default kernel for your
   Notebook if prompted.

## Clean Up

To clean up the resources used during this experiment, follow the steps below:

1. Delete the PyTorchJob created by the last step of the Pipeline.
1. Delete the PVCs that store the data and the logs of the experiment.
1. Delete the Jupyter Notebook server, you used to run the code.

## References

1. Nicholas Kurtansky, Veronica Rotemberg, Maura Gillis, Kivanc Kose, Walter Reade, Ashley Chow.
   (2024). ISIC 2024 - Skin Cancer Detection with 3D-TBP. Kaggle.
   https://kaggle.com/competitions/isic-2024-challenge
