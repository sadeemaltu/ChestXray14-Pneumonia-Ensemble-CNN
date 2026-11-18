# Chest X\-ray Pneumonia Identification Using Ensemble CNNs

**ARTI 502 — Deep Learning Final Project**
Imam Abdulrahman Bin Faisal University

## Overview

This project investigates pneumonia\-label identification in the NIH ChestX\-ray14 dataset using two convolutional neural networks &#40;ResNet18 and DenseNet121&#41; and a prediction\-level ensemble\. The task predicts whether the dataset’s **Pneumonia** label is present; images without that label may contain other thoracic abnormalities and are not necessarily healthy\.

## Dataset

- **Source:** NIH ChestX\-ray14
- **Images:** 112,120 from 30,805 patients
- **Pneumonia\-positive images:** 1,353 &#40;approximately 1\.21%&#41;
- **Patient\-level split:** 78,921 training; 16,825 validation; 16,374 test images

The full chest X\-ray image collection is not stored in this repository\. To reproduce training, obtain the NIH images from their official distribution and configure the image paths in the notebooks\. Patient\-level splits and the binary target definition must remain consistent across both models\.

## Repository Structure

|Path        |Contents                                                                                       |
|------------|-----------------------------------------------------------------------------------------------|
|`notebooks/`|Dataset preparation, ResNet18, DenseNet121, and ensemble experiment notebooks (`.ipynb`).      |
|`data/`     |Dataset metadata, patient-level split files, and configuration.                                |
|`Results/`  |Model predictions, evaluation metrics, experimental settings, and comparison tables.           |
|`Figures/`  |Training curves, ROC/precision-recall plots, confusion matrices, and ensemble analysis figures.|
|`Report/`   |Final research report.                                                                         |
|`README.md` |Project overview and reproduction instructions.                                                |
|`.gitignore`|Git exclusions.                                                                                |

The repository also contains `01_NIH_ChestXray14_Dataset_Preparation.ipynb` at its root\. The folder names above follow the current GitHub repository’s capitalization\.

## Workflow

1. Obtain the NIH ChestX\-ray14 images and configure the data paths\.
2. Use the dataset preparation notebook and the provided metadata/split files to establish the binary labels and patient\-level partitions\.
3. Run the ResNet18 and DenseNet121 notebooks on the **same** patient\-level splits\. Google Colab with GPU support is recommended\.
4. Use each model’s validation and test prediction CSV files from `Results/` as inputs to the ensemble notebook in `notebooks/`\. Match predictions by image identifier and verify true\-label agreement\.
5. Apply the ensemble’s validation\-selected normalization, combination weights, and decision threshold to the held\-out test set without retuning on test data\.
6. Consult `Figures/` and `Report/` for the plots, detailed methodology, and discussion\.

Notebooks may require updating file paths for your Google Drive or Colab environment\. Key Python libraries include PyTorch, torchvision, NumPy, pandas, scikit\-learn, and Matplotlib; consult each notebook’s imports for its precise requirements\.

## Test Results

The held\-out test set contains **16,374 images**, including **193 Pneumonia\-positive** images\.

|Model      |ROC-AUC  |Average precision|Recall at validation-selected max-F1 threshold|
|-----------|--------:|----------------:|---------------------------------------------:|
|ResNet18   |0.679    |0.023            |0.083                                         |
|DenseNet121|0.683    |0.021            |0.119                                         |
|Ensemble   |**0.705**|0.023            |0.140                                         |

The ensemble improved test ROC\-AUC over the standalone CNNs in paired patient\-level bootstrap comparisons\. However, average precision remained low, and the models missed many positive images at their max\-F1 thresholds\. These are research findings, **not a clinically validated diagnostic tool**\.

## Team

- **Estabraq AbuGhonaim:** Dataset preparation and exploratory data analysis
- **Zahra AlRowaei:** Literature review and research gap
- **Abrar Almutawa:** ResNet18
- **Ghadah Bohliga:** DenseNet121
- **Jana Alhumaidan:** Ensemble and hyperparameter experiments
- **Sadeem Alturki:** Results analysis, report, and presentation

## Limitations

The dataset has substantial class imbalance and report\-derived labels\. The negative class can include other diseases\. Models were trained with limited hyperparameter exploration, and validation data were reused for checkpoint and ensemble selection\. See the final report for detailed limitations and references\.
