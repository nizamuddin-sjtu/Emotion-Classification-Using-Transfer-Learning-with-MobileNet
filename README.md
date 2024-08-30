# Emotion-Classification-Using-Transfer-Learning-with-MobileNet
This repository provides an implementation of an emotion classification model using MobileNet as the backbone network. The model is designed to classify images of emotions using a pre-trained MobileNet model, which is fine-tuned on a custom grayscale emotion dataset.
Overview

The project includes:

    Data Preprocessing: Conversion of grayscale images to RGB, resizing, and augmentation using TensorFlow's ImageDataGenerator.
    Model Architecture: Utilization of MobileNet as the feature extractor with additional dense layers for emotion classification.
    Training and Evaluation: Training the model with data augmentation, evaluating its performance, and visualizing results.

Features

    Data Augmentation: Includes techniques like horizontal flipping, rotation, width and height shifts, and zooming to enhance model generalization.
    Pre-trained MobileNet: Leverages a MobileNet model pre-trained on ImageNet for feature extraction.
    Grayscale to RGB Conversion: Converts grayscale images to RGB format for compatibility with MobileNet.
    Visualization: Plots training history and visualizes sample images to inspect the model's performance.

Code Breakdown

    Data Preparation:
        Uses ImageDataGenerator to preprocess and augment images from specified directories.
        Converts grayscale images to RGB format for compatibility with MobileNet.

    Model Creation:
        Builds a custom model based on MobileNet with additional dense layers.
        Freezes the pre-trained MobileNet layers to retain learned features.

    Training:
        Trains the model with augmented training data and validates on a separate validation set.
        Provides plots for accuracy and loss over epochs.

    Evaluation:
        Evaluates model performance on test data and prints test accuracy.

    Visualization:
        Displays sample images from the validation set along with their corresponding labels.

How to Use

    Prepare Dataset: Place your grayscale images into train and test directories in the structure expected by flow_from_directory.
    Run the Script: Execute the script to preprocess data, train the model, and visualize results.

Requirements

    TensorFlow 2.x
    NumPy
    Matplotlib
