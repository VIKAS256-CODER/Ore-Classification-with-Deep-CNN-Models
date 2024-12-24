# 🏔️ Ore Classification with Optimized CNN Ensembles and Feature Fusion
![dataset-cover](https://github.com/user-attachments/assets/b6269a83-2c4f-4f91-b36d-65d00b0687a8)

## 🌟 Overview

**Ore Classification Project** utilizes state-of-the-art deep learning techniques to improve the classification of ores based on their mineral composition. This project combines CNN ensembles, feature fusion, and optimization algorithms to deliver robust and accurate classification results, demonstrating significant advancements over traditional and standalone machine learning models.

---

## ✨ Key Features



- **Feature Fusion**:
  - Combines features extracted from multiple CNN models for enhanced class discrimination.

- **Feature Optimization**:
  - Employs meta-heuristic algorithms (e.g., **FPA**, **PSO**) to select the most discriminative features, reducing noise and improving classification performance.
- **Advanced CNN Ensembles**:
  - Utilizes **AlexNet**, **VGG16**, and **Xception** models in an ensemble with weighted voting for superior accuracy.
- **High Classification Metrics**:
  - Achieves **98.11% accuracy**, **98.18% precision**, and **97.80% Cohen’s kappa** on the OID dataset.

  - Achieves **98.11% accuracy**, **98.18% precision**, and **97.80% Cohen’s kappa** on the OID dataset.

---

## 📂 Dataset

The **Ore Images Dataset (OID)** contains **957 images** categorized into seven ore types:

- **Biotite**
- **Bornite**
- **Chrysocolla**
- **Malachite**
- **Muscovite**
- **Pyrite**
- **Quartz**

| Class Name  | Total Images |
| ----------- | ------------ |
| Biotite     | 68           |
| Bornite     | 170          |
| Chrysocolla | 164          |
| Malachite   | 235          |
| Muscovite   | 77           |
| Pyrite      | 98           |
| Quartz      | 145          |
| **Total**   | **957**      |

The dataset is publicly available on Kaggle: [Minerals Identification Dataset](https://www.kaggle.com/asiedubrempong/minerals-identification-dataset).

---

## 📊 Results

The project evaluated four different approaches:

1. **Transfer Learning**: Testing individual CNN models.
2. **Feature Fusion**: Combining features from the best-performing CNNs.
3. **Feature Optimization**: Selecting optimal features using meta-heuristics.
4. **Ensemble Learning**: Combining CNN predictions with weighted voting.

### Comparative Results

| Method                       | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Cohen’s Kappa |
| ---------------------------- | ------------ | ------------- | ---------- | ------------ | ------------- |
| Transfer Learning (AlexNet)  | 95.93        | 96.10         | 95.93      | 95.93        | 95.25         |
| Feature Fusion               | 93.13        | 93.31         | 93.13      | 93.08        | 91.98         |
| Feature Selection (FPA)      | 96.29        | 96.42         | 96.29      | 96.28        | 95.67         |
| **Weighted Voting Ensemble** | **98.11**    | **98.18**     | **98.11**  | **98.11**    | **97.80**     |

---

## 🛠️ Installation

### Prerequisites

- **Python**: 3.9+
- **TensorFlow**: 2.10.0
- **Keras**: 2.10.0
- **Scikit-Learn**: 1.3.2

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/ymyurdakul/Ore-Classification-with-Deep-CNN-Models.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Ore-Classification-with-Deep-CNN-Models
   ```

---

## 👥 Meet the Team

### 👨‍🏫 Prof. Dr. Şakir Taşdemir

- **Affiliation**: Selçuk University, Computer Engineering Department
- **Expertise**: Deep Learning, Ensemble Methods
- **Email**: [stasdemir@selcuk.edu.tr](mailto\:stasdemir@selcuk.edu.tr)

### 👩‍🏫 Assist. Prof. Dr. Kübra Uyar

- **Affiliation**: Alanya Alaaddin Keykubat University, Computer Engineering Department
- **Expertise**: Machine Learning, Feature Optimization
- **Email**: [kubra.uyar@alanya.edu.tr](mailto\:kubra.uyar@alanya.edu.tr)

### 👨‍🎓 Mustafa Yurdakul (PhD Candidate)

- **Affiliation**: Kırıkkale University, Computer Engineering Department
- **Expertise**: CNN Ensembles, Optimization Techniques
- **Email**: [mustafayurdakul@kku.edu.tr](mailto\:mustafayurdakul@kku.edu.tr)

---

## 📬 Contact

For inquiries or collaborations, contact Mustafa Yurdakul: [mustafayurdakul@kku.edu.tr](mailto\:mustafayurdakul@kku.edu.tr).

---

## 📝 License

This project is licensed under the **MIT License**. See the LICENSE file for details.

---

## 🤝 Acknowledgments

Special thanks to the contributors of the OID dataset and the deep learning research community for their valuable resources and tools.

