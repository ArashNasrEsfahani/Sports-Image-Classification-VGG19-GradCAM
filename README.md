# 🤸‍♂️ Advanced Sports Image Classification with VGG-19 and Grad-CAM

A deep learning project for classifying sports images using a fine-tuned VGG-19 model, complete with model interpretability via Grad-CAM visualizations.

## 🌟 About The Project

This project was developed to classify images across 100 different sports using a pre-trained VGG-19 model. The primary goal is to achieve high classification accuracy while also understanding the model's decision-making process using the **Grad-CAM** technique. This technique helps us visualize which parts of an image the model focuses on to make a specific prediction.

### 🎯 Key Objectives

*   Implement and train a powerful image classification model based on VGG-19.
*   Fine-tune the model to achieve a validation accuracy of over 80%.
*   Use Grad-CAM to interpret and visualize the model's predictions.
*   Provide a comprehensive report on the methodology and results.

---

## 💾 Dataset

This project utilizes the **[100 Sports Image Classification](https://www.kaggle.com/datasets/gpiosenka/sports-classification)** dataset. This dataset contains images from 100 different sport.

---

## ⚙️ Methodology

### 1. Model Architecture

The model architecture is a modified VGG-19 network, customized as follows:
*   The convolutional base from a pre-trained VGG-19 model is used as a feature extractor.
*   A custom `1x1` convolutional layer is added to reduce the channel depth from 512 to 256.
*   The original classifier is replaced with a new one tailored for the 100-class sports dataset, which includes `Dropout` layers to prevent overfitting.

### 2. Training and Fine-Tuning Strategy

A two-phase training strategy was employed to effectively leverage the pre-trained weights:
*   **Phase 1: Feature Extraction.** Initially, the entire pre-trained convolutional base was frozen. Only the weights of the newly added `1x1` convolution and the final classifier were updated. This allows the model to learn based on the powerful, general-purpose features extracted by VGG-19.
*   **Phase 2: Fine-Tuning.** After the initial training, the entire network was unfrozen. The learning rate was reduced to make smaller, more careful adjustments to the pre-trained weights, adapting them specifically to the sports dataset.

### 3. Explainability with Grad-CAM

To interpret the model's decision-making process, the **Gradient-weighted Class Activation Mapping (Grad-CAM)** technique was implemented. Grad-CAM produces a coarse localization map (a heatmap) that highlights the important regions in the input image that the model used to predict a specific class.

---

## 📊 Results

The model was successfully trained and achieved the following results:
*   **Peak Validation Accuracy:** **87.6%**, which greatly surpasses the required 80% target.
*   **Final Test Accuracy:** **89.60%**, demonstrating very strong generalization performance.

The Grad-CAM visualizations confirmed that the model learns to focus on semantically relevant objects for each sport, such as the player, ball, or key equipment, providing confidence that it is learning correct features.

---

## 🚀 How To Use

To run this project, follow these steps:

1.  Clone the repository:
    ```bash
    git clone https://github.com/ArashNasrEsfahani/Sports-Image-Classification-VGG19-GradCAM

    ```
2.  Install the required dependencies using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Python notebook (`.ipynb`) and adjust the data paths according to your project structure.

---

## 📦 Dependencies

All necessary libraries are listed in the `requirements.txt` file.
