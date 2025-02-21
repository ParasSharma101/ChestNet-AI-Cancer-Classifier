# Chest CT Scan Classification using VGG16

## 📌 Project Overview
This project leverages **transfer learning** with the **VGG16** model to classify **chest CT scans** into four categories. The dataset is sourced from Kaggle, and the model is fine-tuned using **TensorFlow/Keras**.

## 🚀 Technologies Used
- Python
- TensorFlow/Keras
- OpenCV
- Matplotlib
- NumPy
- Pandas

## 📂 Dataset
The dataset consists of chest CT scan images categorized into four classes:
1. **Class 1**
2. **Class 2**
3. **Class 3**
4. **Class 4**

The dataset is preprocessed and augmented using **image augmentation techniques** to enhance generalization.

## 🔧 Model Architecture
- **Pretrained Model**: VGG16 (with ImageNet weights)
- **Modified Fully Connected Layers**:
  - Flatten layer
  - Dense (256 neurons, ReLU activation)
  - Dropout (0.5)
  - Output layer (4 neurons, Softmax activation)
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Evaluation Metrics**: Accuracy

## 📊 Training Details
- **Epochs**: 50
- **Batch Size**: 32
- **Early Stopping**: Implemented to prevent overfitting
- **Best Validation Accuracy**: 88.89%

## 🔬 Results & Evaluation
- Accuracy improved progressively across epochs
- Model generalization was ensured via **data augmentation**
- Evaluation performed using a **confusion matrix** and accuracy/loss plots

## 📌 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python train.py
   ```
4. Evaluate the model:
   ```bash
   python evaluate.py
   ```

## 📈 Future Improvements
- Experiment with different architectures (ResNet, EfficientNet, etc.)
- Hyperparameter tuning for better performance
- Deploy the model as a web application

## 🔗 References
- [Kaggle Dataset](#) (https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)
- [VGG16 Paper](https://arxiv.org/abs/1409.1556)

---
📢 **Contributions are welcome!** Feel free to raise issues or submit pull requests.

👨‍💻 Developed by **Paras Sharma**

