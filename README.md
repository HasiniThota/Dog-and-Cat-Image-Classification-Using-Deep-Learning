# Dog and Cat Image Classification

This project involves classifying images of dogs and cats using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model is trained on a dataset downloaded from Kaggle and is designed to distinguish between images of dogs and cats with high accuracy.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Introduction
This project uses a CNN model to classify images of dogs and cats. The goal is to achieve optimal performance using deep learning techniques, including batch normalization, dropout, and data preprocessing.

## Dataset
The dataset used in this project is the "Dogs vs. Cats" dataset, which contains 25,000 images (12,500 images of dogs and 12,500 images of cats). The dataset is split into 20,000 training images and 5,000 validation images.

- **Download the dataset** from Kaggle: [Dogs vs. Cats](https://www.kaggle.com/salader/dogs-vs-cats)
- The dataset is automatically extracted and loaded using TensorFlow's `image_dataset_from_directory`.

## Model Architecture
The CNN model used in this project consists of:
- **3 Convolutional Layers** with 32, 64, and 128 filters respectively, followed by batch normalization and max pooling.
- **Flattening Layer** to convert the 3D outputs into 1D.
- **2 Dense Layers** with 128 and 64 units, each followed by a dropout layer to prevent overfitting.
- **Output Layer** with a sigmoid activation function to output a probability score for binary classification.

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
])
```

## Preprocessing
- Images are resized to 256x256 pixels and normalized to the range [0, 1].
- Data augmentation techniques can be applied to improve model generalization.

## Training
The model is compiled with the Adam optimizer and binary cross-entropy loss function. It is trained for 10 epochs using a batch size of 32.

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, epochs=10, validation_data=validation_ds)
```

### Training Performance
- Achieved a training accuracy of 96.4% and a validation accuracy of 82.7% after 10 epochs.
- Loss and accuracy metrics are plotted to visualize the model's performance over time.

## Results
The model demonstrates good performance on the validation set, though there is potential for further improvement by fine-tuning hyperparameters or applying additional regularization techniques.

### Sample Prediction
A sample prediction on a test image correctly classified it as a dog.

```python
model.predict(test_input)
# Output: array([[1.]], dtype=float32)
```

## Requirements
- Python 3.7+
- TensorFlow 2.x
- Keras
- OpenCV
- Kaggle API (for dataset download)
- Matplotlib
- NumPy

Install the required libraries using:
```bash
pip install tensorflow keras opencv-python matplotlib numpy kaggle
```

## Usage
1. **Set up Kaggle API**: Place `kaggle.json` in the appropriate directory to access the dataset.
2. **Run the code**: Use Google Colab or a local Jupyter Notebook to execute the code.
3. **Make predictions**: Modify the code to test the model on custom images.

## Acknowledgments
- **Kaggle**: For providing the "Dogs vs. Cats" dataset.
- **Google Colab**: For facilitating an easy and efficient training environment.
- **TensorFlow/Keras Community**: For providing extensive documentation and resources.
