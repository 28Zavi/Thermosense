# Thermosense

# Fruit Maturity Detection Using RGB and Thermal Images

## Project Overview
This project aims to accurately identify the ripeness level of Alphonso mangoes using a combination of RGB and thermal images, applying advanced machine learning models. The goal is to detect fruit maturity without piercing the fruit surface, addressing limitations of traditional methods that rely on subjective and inaccurate visual assessment.

## Problem Statement
The traditional approach of identifying fruit ripeness often lacks accuracy and is biased. For example, a common assumption is that a ripe mango turns yellow, but many mango species remain green even when fully ripe or overripe. Our solution uses machine learning models to assess ripeness through RGB and thermal images to eliminate such subjectivity.

## Dataset Creation
We created a custom dataset of Alphonso mangoes, using a thermal camera (FLIR-One) to capture both RGB and thermal images. Around 200 initial images were collected and later augmented to 995 images. Data preprocessing included resizing, normalization, augmentation, and segmentation to prepare the images for model training.

## Machine Learning Models
We used deep learning models (CNN, AlexNet, VGG16, ResNet50) for feature extraction and traditional machine learning models (k-NN, Random Forest, SVM) for classification. The combination of the InceptionV3 model for feature extraction and the SVM classifier produced the best results, achieving a 97% test accuracy.

## Key Steps
1. **Data Collection**: Captured RGB and thermal images of Alphonso mangoes.
2. **Data Preprocessing**: Applied resizing, normalization, augmentation, and segmentation techniques.
3. **Feature Extraction**: Extracted features using CNN models (AlexNet, VGG16, MobileNetV3, ResNet50).
4. **Model Training**: Trained k-NN, Random Forest, and SVM classifiers.
5. **Best Results**: InceptionV3 + SVM achieved a 97% accuracy.

## Conclusion
This project presents a robust machine learning approach to fruit maturity detection, providing a significant improvement over traditional methods. The combined use of RGB and thermal imaging allows for accurate ripeness detection without damaging the fruit.

