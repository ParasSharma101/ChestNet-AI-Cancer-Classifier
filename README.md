
## **Chest CT-Scan Image Classification ** ##

Chest CT-Scan Image Classification
This project focuses on classifying chest CT scan images into four categories: Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, and Normal, using a Convolutional Neural Network (CNN) with transfer learning from the VGG16 model.


->Table of Contents


    ->Installation
    
    
    ->Dataset
    
    
    ->Model Architecture
    
    
    ->Training
    
    
    ->Evaluation
    
    
    ->Results
    
    
    ->References
    

->Installation
    Follow these steps to set up the project:


->Clone the repository:


  git clone https://github.com/username/repository-name.git


->Install required dependencies:


  pip install -r requirements.txt


->Set up Kaggle API:


->Download kaggle.json from your Kaggle account.


->Copy the file to your system:
 

    mkdir -p ~/.kaggle
    cp kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json


->Download the dataset:

 
kaggle datasets download -d mohamedhanyyy/chest-ctscan-images -p /content --unzip


Dataset


The dataset consists of CT scan images categorized into four classes:
    
    
    ->Adenocarcinoma

    
    ->Large Cell Carcinoma
    
    
    ->Squamous Cell Carcinoma
    
    
    ->Normal


It is divided into:
  
  
  train: for training the model
  
  
  test: for evaluating performance
  
  
  valid: for validation during training

->Model Architecture


The project uses transfer learning based on the VGG16 model:


VGG16 (pre-trained on ImageNet) as the base model.


Custom layers added on top:


BatchNormalization and Dropout for regularization.


Dense layers with ReLU activation.


Softmax output layer for classifying into 4 categories.


Key components:


Optimizer: Adam


Loss Function: Categorical Crossentropy


Metrics: Accuracy


Training


The model is trained with data augmentation to improve generalization.


To start training the model, run:


    history = pretrained_model.fit(
        train_ds,
        epochs=50,
        validation_data=validation_ds,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5),
            tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
            tf.keras.callbacks.TensorBoard(log_dir='./logs')
        ]
    )


Evaluation


After training, the model is evaluated on the test dataset:

    pretrained_model.evaluate(test_ds)

You can also generate a confusion matrix and a classification report:

    from sklearn.metrics import classification_report, confusion_matrix
    y_pred = pretrained_model.predict(test_ds)
    print(classification_report(y_true, y_pred))
    Results
Training Accuracy: ~90%


Validation Accuracy: ~88%


Test Accuracy: ~87%


The model achieves high accuracy, with well-separated classifications for each type of chest cancer and healthy cases.

References


Dataset: Kaggle Chest CT-Scan Images
VGG16 Paper: Very Deep Convolutional Networks for Large-Scale Image Recognition
TensorFlow Documentation: TensorFlow
