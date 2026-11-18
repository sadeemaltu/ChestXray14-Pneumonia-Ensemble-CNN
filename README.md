Chest X-ray Pneumonia Identification Using Ensemble CNNs

ARTI 502 Deep Learning Final Project — Imam Abdulrahman Bin Faisal University.

Overview

This project studies Pneumonia-label identification in the NIH ChestX-ray14 dataset using two convolutional neural networks, ResNet18 and DenseNet121, and a prediction-level ensemble. The target is the presence or absence of the dataset’s Pneumonia label. A negative label does not necessarily mean a healthy chest X-ray: other thoracic findings may be present.

Dataset

• Source: NIH ChestX-ray14 (112,120 images from 30,805 patients).
• Pneumonia-positive images: 1,353 (approximately 1.21%).
• Patient-level split: training 78,921 images; validation 16,825; test 16,374.
• Full X-ray images are not included in this repository. Obtain them from the NIH dataset source and follow the dataset notebook’s instructions to configure local paths.
• The NIH_Dataset_Handoff.zip file includes metadata, split information, configuration and sample images only.

Project files

|File                                           |Purpose                                                                                          |
|-----------------------------------------------|-------------------------------------------------------------------------------------------------|
|`01_NIH_ChestXray14_Dataset_Preparation.ipynb` |Dataset cleaning, exploratory analysis, patient-level splitting and PyTorch data pipeline.       |
|`NIH_Dataset_Handoff.zip`                      |Dataset metadata, split files, configuration and sample images.                                  |
|`02_ResNet18_Pneumonia_Trainingـnotebook.ipynb`|ResNet18 training and evaluation notebook.                                                       |
|`resnet18_output.zip`                          |ResNet18 predictions, metrics, training history, figures and model checkpoint.                   |
|`DenseNet121.ipynb`                            |DenseNet121 training and evaluation notebook.                                                    |
|`DenseNet121_Colab_Selected_Handoff.zip`       |DenseNet121 validation/test predictions, metrics, settings, training history and checkpoint.     |
|`Member5_Ensemble_Outputs(1).zip`              |Ensemble notebook, evaluation tables, validation weight search, confidence intervals and figures.|
|`ARTI502_Final_Project_Report.docx`            |Final project report.                                                                            |

Extract the ZIP files when working with the data and outputs. The ensemble notebook uses predictions from both CNNs, aligned by image identifier and label. The full dataset and pretrained weights may require additional downloads or path configuration to rerun training.

Reproducing the workflow

1. Obtain NIH ChestX-ray14 images and set their location in the dataset preparation notebook.
2. Run the dataset preparation notebook to produce metadata and patient-level train/validation/test splits.
3. Run the ResNet18 and DenseNet121 notebooks on the same splits and target definition (GPU recommended).
4. Extract the model-output ZIP files and locate each model’s validation and test prediction CSV files.
5. Run 05_Ensemble_and_Hyperparameter_Experiments.ipynb from the ensemble ZIP, updating paths as necessary. Ensemble weights, normalization parameters and decision thresholds are selected on validation, then applied to test.

The notebooks were developed for Google Colab. Their dependencies include Python, PyTorch, torchvision, NumPy, pandas, scikit-learn and plotting libraries; use the notebook imports for the exact environment requirements.

Reported test results

16,374 test images, including 193 with the Pneumonia label.

|Model      |AUROC    |Average precision|Recall at validation-selected max-F1 threshold|
|-----------|--------:|----------------:|---------------------------------------------:|
|ResNet18   |0.679    |0.023            |0.083                                         |
|DenseNet121|0.683    |0.021            |0.119                                         |
|Ensemble   |**0.705**|0.023            |0.140                                         |

The ensemble improved AUROC over either standalone model in the reported paired patient-level bootstrap comparisons. However, average precision remained low and none of the models should be considered clinically validated. The project is a research comparison, not a diagnostic medical device.

Team

• Estabraq AbuGhonaim — Dataset preparation and EDA
• Zahra AlRowaei — Literature review and research gap
• Abrar Almutawa — ResNet18
• Ghadah Bohliga — DenseNet121
• Jana Alhumaidan — Ensemble and hyperparameter experiments
• Sadeem Alturki — Results analysis, report and presentation

Notes

Results are based on a rare and imperfectly annotated label. Performance comparisons with other publications must account for differences in data splits, label definitions and evaluation methods. See the final report for experimental settings, limitations, references and detailed discussion.
