# Ore-Classification-with-Deep-CNN-Models
![dataset-cover](https://github.com/user-attachments/assets/b6269a83-2c4f-4f91-b36d-65d00b0687a8)

This repository contains the code and resources for the classification of ore types using deep learning techniques. The dataset used includes images of seven different ore types: biotite, bornite, chrysocolla, malachite, muscovite, pyrite, and quartz. Four different experimental cases were conducted to find the most efficient approach for ore classification.

# Dataset
The Ore Images Dataset (OID) is publicly available on Kaggle and contains a total of 957 images, divided into seven ore types. The dataset is augmented using techniques like center cropping, edge cropping, zooming, and brightness transformation to enhance model training. 
Dataset Link -> https://www.kaggle.com/datasets/asiedubrempong/minerals-identification-dataset

# Experimental Cases
The project explores four different approaches to improve the classification accuracy:

# Case #1: Transfer Learning and Fine-Tuning
In this case, 17 well-known pre-trained CNN models are fine-tuned on the ore dataset. The models include architectures such as AlexNet, VGG16, DenseNet, and ResNet. The aim is to find the best-performing model in terms of classification accuracy. AlexNet achieved the highest accuracy in this case.

# Case #2: Feature Fusion
The three best-performing CNN models from Case #1 (AlexNet, VGG16, Xception) are used as feature extractors. The extracted features from these models are fused together, and the fused feature set is classified using a Support Vector Machine (SVM) with different kernels (linear, RBF, polynomial, and sigmoid).

# Case #3: Feature Selection with Feature Fusion
After feature fusion, optimization algorithms (such as ABC, GA, FPA, and PSO) are employed to select the most discriminative features from the fused feature set. The selected features are then classified using an SVM. This approach improves the model performance by reducing noise in the feature set.

# Case #4: Ensemble Learning
The three best-performing models (AlexNet, VGG16, Xception) are combined using ensemble learning techniques (hard voting, soft voting, and weighted voting). The ensemble approach yields the highest classification accuracy of all the cases, with weighted voting achieving the best results.

# Results
The ensemble method (Case #4) was the most successful, achieving an accuracy of 98.11%, precision of 98.18%, recall of 98.11%, and F1-score of 98.11%. This demonstrates the advantage of combining multiple models for robust predictions.

# Dependencies
Python 3.x
TensorFlow/Keras
Scikit-learn
meal-py

# Usage
Clone this repository and install the required dependencies and dataset. The code for each experimental case is organized into separate files. Simply navigate to the folder of the experiment you wish to run and follow the instructions in the respective Python scripts.

git clone <repository-url>
cd <case-folder>
python <script>.py

# Contact
For any questions or issues, please contact Şakir Taşdemir (stasdemir@selcuk.edu.tr), Kübra Uyar (kubra.uyar@alanya.edu.tr) or Mustafa Yurdakul (mustafa.yurdakul@kirikkale.edu.tr).

Feel free to adjust it based on your specific needs!
