# COVID-19 Chest X-Ray Classification

## Introduction
This project details the development of a deep learning model for the classification of chest X-ray images. The goal is to create a robust and accurate diagnostic tool capable of distinguishing between three distinct conditions: normal lung tissue, Viral Pneumonia, and COVID-19. The notebook showcases a practical application of computer vision and deep learning in a critical healthcare context.

## Problem Statement / Objective
The objective is to build a reliable multi-class classification model for medical image analysis. The core challenges addressed in this project are:
1.  **Imbalanced Data**: The dataset contains a small number of samples, requiring a strategy to prevent model overfitting and ensure fair representation of all classes.
2.  **Model Performance**: The model must achieve a high degree of accuracy and precision across all three classes to be considered a viable aid in the diagnostic process.
3.  **Model Interpretability**: Beyond just making predictions, the project aims to visualize and explain the model's decision-making process, ensuring its trustworthiness in a medical setting.

## Dataset Details
The dataset consists of chest X-ray images and their corresponding labels.
* **Data Size**: 251 images, each 128x128 pixels with 3 color channels.
* **Data Source**: The image data is stored in a `CovidImages.npy` file, and the labels are in a `CovidLabels.csv` file.
* **Target Classes**: The images are classified into three categories: 'Normal', 'Covid', and 'Viral Pneumonia'.
* **Data Preparation**: The images were split into training and testing sets, and class weights were computed to address the dataset's class imbalance. A data augmentation pipeline was implemented to artificially expand the training set and improve the model's ability to generalize.


## Methods and Algorithms
This project employs a deep learning methodology using a transfer learning approach.
* **Transfer Learning**: The project leverages the **VGG16** convolutional base, a model pre-trained on a large-scale image dataset. This allows the model to benefit from pre-existing knowledge of image features, which is highly effective with a smaller dataset.
* **Convolutional Neural Network (CNN)**: A custom CNN is built on top of the VGG16 base. The architecture includes a `GlobalAveragePooling2D` layer to efficiently flatten the feature maps, followed by a `Dense` layer and a final `Dense` layer with `softmax` activation to output the classification probabilities for each of the three classes.
* **Model Explainability**: **Grad-CAM (Gradient-weighted Class Activation Mapping)** is used to generate visual explanations for the model's predictions. This technique creates heatmaps that highlight the specific areas in the X-ray images that the model considered most important for its classification decision.

## Key Results
* **High Accuracy**: The **VGG16-based transfer learning model** achieved an accuracy of **94%** and a recall of **94%**. A custom-trained CNN model achieved an accuracy of **92%** and a recall of **82%**.
* **Performance Metrics**: The **classification report** and **confusion matrix** revealed high precision, recall, and F1-scores across all three classes, confirming the model is robust and not biased towards any single class.
* **Interpretability**: The **Grad-CAM visualizations** successfully highlighted key regions of interest within the X-ray images, providing a clear explanation of the model's focus during its prediction, which is crucial for building trust in an AI-powered diagnostic tool.


## Tech Stack
* **Libraries**:
    * `pandas` and `numpy` for data manipulation.
    * `matplotlib` and `seaborn` for data visualization.
    * `tensorflow.keras` for building, training, and evaluating the deep learning models.
    * `sklearn` for data splitting and performance metrics.
    * `cv2` for image processing.
    * `tf_keras_vis` for Grad-CAM visualization.