# Skin cancer Classification using custom VGG16 and ResNet

## Overview
This repository contains code for classifying skin lesion images into different categories using various machine learning models.

# Table of Contents
1. [Overview](#overview)
2. [Introduction](#introduction)
3. [Dependencies](#dependencies)
4. [Usage](#usage)
    1. [K-Nearest Neighbors (KNN) Model](#k-nearest-neighbors-knn-model)
    2. [Custom Convolutional Neural Network (CNN) Model](#custom-convolutional-neural-network-cnn-model)
    3. [VGG-like Convolutional Neural Network Model](#vgg-like-convolutional-neural-network-model)
    4. [Residual Network (ResNet)-like Model](#residual-network-resnet-like-model)
    5. [Machine Learning Models Comparison](#machine-learning-models-comparison)
5. [Training and Evaluation](#training-and-evaluation)
    1. [Performance Plotting](#performance-plotting)
    2. [Confusion Matrix](#confusion-matrix)
6. [Saving Trained Models and Training History](#saving-trained-models-and-training-history)
7. [Visualization and Analysis](#visualization-and-analysis)
8. [Notes](#notes)
9. [File Descriptions](#file-descriptions)
10. [Usage](#usage)
11. [Contributions](#contributions)

![Screenshot 2023-11-19 at 3 47 08 PM](https://github.com/Shobhit-Singhh/Skin_cancer/assets/117563572/abbebfba-1f23-4030-aed3-98f605d76fcb)


## Introduction <a name="introduction"></a>
This the HAM10000 ("Human Against Machine with 10000 training images") dataset.It consists of 10015 dermatoscopicimages which are released as a training set for academic machine learning purposes and are publiclyavailable through the ISIC archive. This benchmark dataset can be used for machine learning and for comparisons with human experts.

It has 7 different classes of skin cancer which are listed below :
- 'akiec' (actinic keratoses and intraepithelial carcinoma)
- 'bcc' (basal cell carcinoma)
- 'bkl' (benign keratosis-like lesions)
- 'df' (dermatofibroma)
- 'nv' (melanocytic nevi)
- 'vasc' (pyogenic granulomas and hemorrhage)
- 'mel' (melanoma)

## Dependencies <a name="dependencies"></a>
- Python 3.x
- TensorFlow
- scikit-learn
- Matplotlib
- NumPy
- PIL

## Usage <a name="usage"></a>
1. Clone the repository:
    ```bash
    git clone https://github.com/Shobhit-Singhh/Skin_cancer.git
    cd Skin_cancer
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Execute the code by running the provided scripts.

## Models

### K-Nearest Neighbors (KNN) Model <a name="k-nearest-neighbors-knn-model"></a>

Code demonstrating the implementation of a K-Nearest Neighbors (KNN) classifier for a machine learning task.

## Description

K-Nearest Neighbors (KNN) is a simple and intuitive algorithm used for classification and regression. It is a type of instance-based learning where the model makes predictions based on the majority class of its k nearest neighbors in the feature space. In this, we showcase the use of the KNN algorithm for classification tasks.

## Usage

The provided code initializes and trains a KNN model using the scikit-learn library. It then performs predictions on test data and calculates the accuracy of the model.

### Code Example

```python
# Initialize and train KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_flattened, y_train_flattened)

# Predictions using KNN
knn_predictions = knn_model.predict(X_test_flattened)

# Calculate accuracy
accuracy = accuracy_score(y_test, knn_predictions)
print(f"KNN Accuracy: {accuracy}")
```


### Custom Convolutional Neural Network (CNN) Model <a name="custom-convolutional-neural-network-cnn-model"></a>

This code demonstrating the implementation of a Convolutional Neural Network (CNN) using TensorFlow for image classification tasks.

## Description

The Convolutional Neural Network (CNN) is a type of deep neural network that is particularly adept at analyzing visual imagery. It consists of multiple layers of convolutional and pooling operations, followed by fully connected layers. This repository showcases the construction and training of a CNN model using TensorFlow for image classification.

## Model Architecture

The model architecture consists of several convolutional blocks followed by dense layers for classification:

- Convolutional Block 1:
  - Conv2D layer (16 filters, kernel size 3x3, ReLU activation)
  - MaxPooling2D layer (2x2)
  - Batch Normalization

- Convolutional Block 2:
  - Conv2D layer (32 filters, kernel size 3x3, ReLU activation)
  - Conv2D layer (64 filters, kernel size 3x3, ReLU activation)
  - MaxPooling2D layer (2x2)
  - Batch Normalization

- Convolutional Block 3:
  - Conv2D layer (128 filters, kernel size 3x3, ReLU activation)
  - Conv2D layer (256 filters, kernel size 3x3, ReLU activation)

- Flatten and Dense Layers:
  - Flatten layer
  - Dropout layer (20%)
  - Dense layer (256 neurons, ReLU activation)
  - Batch Normalization
  - Dropout layer (20%)
  - Dense layer (128 neurons, ReLU activation)
  - Batch Normalization
  - Dense layer (64 neurons, ReLU activation)
  - Batch Normalization
  - Dropout layer (20%)
  - Dense layer (32 neurons, ReLU activation)
  - Batch Normalization
  - Output layer (7 classes, Softmax activation)

## Usage

The provided code defines the CNN model architecture, compiles the model, and sets up callbacks for model checkpointing. The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss function.

### Code Example

```python
# Define the model
model = Sequential()

# Convolutional Block 1
model.add(Conv2D(16, kernel_size=(3, 3), input_shape=(28, 28, 3), activation='relu', padding='same', name='conv1'))
model.add(MaxPool2D(pool_size=(2, 2), name='maxpool1'))
model.add(BatchNormalization(name='batchnorm1'))

# Convolutional Block 2
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', name='conv2'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv3'))

model.add(MaxPool2D(pool_size=(2, 2), name='maxpool2'))
model.add(BatchNormalization(name='batchnorm2'))

# Convolutional Block 3
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', name='conv4'))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', name='conv5'))

# Flatten and Dense Layers
model.add(Flatten(name='flatten'))
model.add(tf.keras.layers.Dropout(0.2, name='dropout1'))
model.add(Dense(256, activation='relu', name='dense1'))

model.add(BatchNormalization(name='batchnorm3'))
model.add(tf.keras.layers.Dropout(0.2, name='dropout2'))
model.add(Dense(128, activation='relu', name='dense2'))

model.add(BatchNormalization(name='batchnorm4'))
model.add(Dense(64, activation='relu', name='dense3'))

model.add(BatchNormalization(name='batchnorm5'))
model.add(tf.keras.layers.Dropout(0.2, name='dropout3'))
model.add(Dense(32, activation='relu', name='dense4'))

model.add(BatchNormalization(name='batchnorm6'))
model.add(Dense(7, activation='softmax', name='output'))

# Display model summary
model.summary()

# ModelCheckpoint callback
callback = tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5',monitor='val_accuracy',mode='max',verbose=1,save_best_only=True)

# Optimizer
optimizer = tf.keras.optimizers.Adam(lr=0.001)

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
```

### VGG-like Convolutional Neural Network Model <a name="vgg-like-convolutional-neural-network-model"></a>

This code illustrating the implementation of a VGG-like Convolutional Neural Network (CNN) using TensorFlow for image classification tasks.

## Description

The VGG-like CNN architecture is inspired by the VGG models, characterized by its stacked convolutional layers with small 3x3 filters and max-pooling layers. This repository showcases the construction and training of a VGG-like CNN model using TensorFlow for image classification.

## Model Architecture

The model architecture consists of several convolutional blocks followed by fully connected layers:

- Block 1:
  - Conv2D layer (64 filters, kernel size 3x3, ReLU activation)
  - Conv2D layer (64 filters, kernel size 3x3, ReLU activation)
  - MaxPooling2D layer (2x2)
  - Batch Normalization

- Block 2:
  - Conv2D layer (128 filters, kernel size 3x3, ReLU activation)
  - Conv2D layer (128 filters, kernel size 3x3, ReLU activation)
  - MaxPooling2D layer (2x2)
  - Batch Normalization

- Block 3:
  - Conv2D layer (256 filters, kernel size 3x3, ReLU activation)
  - Conv2D layer (256 filters, kernel size 3x3, ReLU activation)
  - Conv2D layer (256 filters, kernel size 3x3, ReLU activation)
  - MaxPooling2D layer (2x2)
  - Batch Normalization

- Flatten and Fully Connected Layers:
  - Flatten layer
  - Dropout layer (20%)
  - Dense layer (256 neurons, ReLU activation)
  - Batch Normalization
  - Dropout layer (20%)
  - Dense layer (128 neurons, ReLU activation)
  - Batch Normalization
  - Dropout layer (20%)
  - Dense layer (64 neurons, ReLU activation)
  - Batch Normalization
  - Dropout layer (20%)
  - Dense layer (32 neurons, ReLU activation)
  - Batch Normalization

- Output Layer:
  - Dense layer (7 classes, Softmax activation)

## Usage

The provided code defines the VGG-like CNN model architecture, compiles the model using the Adam optimizer, and sets up the model for image classification tasks.

### Code Example

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

# New input size: (64, 64, 3)
input_shape = (28, 28, 3)

# Create a VGG-like model
vgg_model = Sequential()

# Block 1
vgg_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape, name='block1_conv1'))
vgg_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
vgg_model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
vgg_model.add(BatchNormalization())

# Block 2
vgg_model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
vgg_model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
vgg_model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
vgg_model.add(BatchNormalization())

# Block 3
vgg_model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
vgg_model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
vgg_model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
vgg_model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
vgg_model.add(BatchNormalization())

# Flatten and fully connected layers
vgg_model.add(Flatten(name='flatten'))
vgg_model.add(Dropout(0.2))
vgg_model.add(Dense(256, activation='relu', name='fc1'))
vgg_model.add(BatchNormalization())
vgg_model.add(Dropout(0.2))
vgg_model.add(Dense(128, activation='relu', name='fc2'))
vgg_model.add(BatchNormalization())
vgg_model.add(Dropout(0.2))
vgg_model.add(Dense(64, activation='relu', name='fc3'))
vgg_model.add(BatchNormalization())
vgg_model.add(Dropout(0.2))
vgg_model.add(Dense(32, activation='relu', name='fc4'))
vgg_model.add(BatchNormalization())

# Output layer
vgg_model.add(Dense(7, activation='softmax', name='output'))

# Display model summary
vgg_model.summary()

# Optimizer
optimizer = tf.keras.optimizers.Adam(lr=0.001)

# Compile the model
vgg_model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
```

### Residual Network (ResNet)-like Model <a name="residual-network-resnet-like-model"></a>

This code showcasing the implementation of a Residual Network (ResNet)-like Convolutional Neural Network (CNN) using TensorFlow for image classification tasks.

## Description

The code demonstrates the construction of a ResNet-like CNN model, featuring residual blocks and skip connections. ResNet architectures are known for their ability to handle deeper networks effectively and mitigate vanishing gradient problems.

## Model Architecture

The model architecture comprises an initial convolutional layer, followed by residual blocks and fully connected layers:

- Initial Convolutional Layer:
  - Conv2D layer (64 filters, kernel size 7x7, strides 2x2, ReLU activation)
  - Batch Normalization
  - MaxPooling2D layer (pool size 3x3, strides 2x2)

- Residual Blocks:
  - The code iterates through different stages and creates residual blocks according to the specified numbers. For example:
    - Stage 2 has 3 residual blocks
    - Stage 3 has 4 residual blocks
    - Stage 4 has 6 residual blocks
    - Stage 5 has 3 residual blocks

- Fully Connected Layers:
  - Flatten layer
  - Dropout layer (20%)
  - Dense layer (256 neurons, ReLU activation)
  - Batch Normalization
  - Dropout layer (20%)
  - Dense layer (128 neurons, ReLU activation)
  - Batch Normalization
  - Dropout layer (20%)
  - Dense layer (64 neurons, ReLU activation)
  - Batch Normalization

- Output Layer:
  - Dense layer (7 classes, Softmax activation)

## Usage

The provided code defines the ResNet-like CNN model architecture and compiles it using the Adam optimizer for image classification tasks.



## Residual Blocks Construction

```python
for stage, num_blocks in enumerate([3, 4, 6, 3]):
    for block in range(num_blocks):
        x = residual_block(x, 64, block_name=f'stage{stage + 2}_block{block + 1}')
```

**Iteration through Stages and Blocks:**
`enumerate([3, 4, 6, 3])` creates an iterator that goes through the list `[3, 4, 6, 3]`.
Each value in this list corresponds to the number of residual blocks in a particular stage of the network.
- `stage` represents the index of the current stage.
- `num_blocks` stores the number of residual blocks for the current stage.

**Construction of Residual Blocks:**
- The outer loop iterates over the stages.
- The inner loop iterates through each block within the current stage.
- For each iteration, it calls the `residual_block()` function to construct a residual block and passes the following parameters:
  - `x`: Represents the input tensor or the output from the previous layer.
  - `64`: Indicates the number of filters for the convolutional layers within the residual block.
  - `block_name`: Assigns a name to the residual block based on the stage and block number.

**Building Residual Blocks:**
- The `residual_block()` function constructs each residual block by:
  - Implementing two convolutional layers with ReLU activations and batch normalization.
  - Handling the skip connection (shortcut connection) for identity mapping using the Add layer.
  - Activating the output using a ReLU activation function.

**Elaboration:**
- This code segment efficiently constructs the ResNet-like architecture by creating the residual blocks as specified by the list `[3, 4, 6, 3]`.
- It iterates through stages and creates the desired number of residual blocks for each stage, allowing the model to learn complex features while mitigating vanishing gradient issues.
- By organizing blocks into stages and creating residual connections, the model can effectively learn representations at different levels of abstraction, enhancing its ability to capture intricate patterns within the data.
- Adjusting the numbers within the list can alter the depth and complexity of the network architecture to suit specific requirements or optimize performance for different datasets.


### Full Code Example

```python

input_shape = (28, 28, 3)

# Input layer
inputs = Input(shape=input_shape)

# Initial convolutional layer
x = Conv2D(64, kernel_size=7, strides=2, padding='same', name='initial_conv')(inputs)
x = BatchNormalization(name='initial_bn')(x)
x = Activation('relu', name='initial_relu')(x)
x = MaxPooling2D(pool_size=3, strides=2, padding='same', name='initial_maxpool')(x)

# Residual blocks
def residual_block(x, filters, kernel_size=3, strides=1, block_name='block'):
    shortcut = x

    # First convolutional layer
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', name=f'{block_name}_conv1')(x)
    x = BatchNormalization(name=f'{block_name}_bn1')(x)
    x = Activation('relu', name=f'{block_name}_relu1')(x)

    # Second convolutional layer
    x = Conv2D(filters, kernel_size=kernel_size, padding='same', name=f'{block_name}_conv2')(x)
    x = BatchNormalization(name=f'{block_name}_bn2')(x)

    # Shortcut connection
    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=1, strides=strides, padding='same', name=f'{block_name}_shortcut_conv')(shortcut)
        shortcut = BatchNormalization(name=f'{block_name}_shortcut_bn')(shortcut)

    x = Add(name=f'{block_name}_add')([x, shortcut])
    x = Activation('relu', name=f'{block_name}_relu2')(x)

    return x

# Add residual blocks
for stage, num_blocks in enumerate([3, 4, 6, 3]):
    for block in range(num_blocks):
        x = residual_block(x, 64, block_name=f'stage{stage + 2}_block{block + 1}')

# Fully connected layers
x = Flatten(name='flatten')(x)
x = Dropout(0.2, name='dropout1')(x)
x = Dense(256, activation='relu', name='fc1')(x)
x = BatchNormalization(name='fc1_bn')(x)
x = Dropout(0.2, name='dropout2')(x)
x = Dense(128, activation='relu', name='fc2')(x)
x = BatchNormalization(name='fc2_bn')(x)
x = Dropout(0.2, name='dropout3')(x)
x = Dense(64, activation='relu', name='fc3')(x)
x = BatchNormalization(name='fc3_bn')(x)

# Output layer
outputs = Dense(7, activation='softmax', name='output')(x)

# Build the model
resnet_model = Model(inputs, outputs)

# Display model summary
resnet_model.summary()

# Optimizer
optimizer = tf.keras.optimizers.Adam(lr=0.001)

# Compile the model
resnet_model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
```

### Machine Learning Models Comparison <a name="machine-learning-models-comparison"></a>

Showcases the implementation and comparison of various machine learning models for image classification tasks.

### Models Included
1. **KNN model**
   - K-Nearest Neighbors (KNN) is a simple and intuitive algorithm used for classification and regression.
     
2. **Costum CNN model:** 
   - A basic neural network implemented using TensorFlow's Keras API.
   
3. **VGG Model:**
   - Implementation of a VGG-like Convolutional Neural Network using TensorFlow's Keras API.
   
4. **ResNet Model:**
   - Implementation of a ResNet-like architecture using TensorFlow's Keras API.

## Training and Evaluation <a name="training-and-evaluation"></a>

Each model is trained and evaluated using a similar methodology:

- **Training:**
  - Training is performed on the provided dataset (`x_train` and `y_train`) for 100 epochs with a batch size of 128.
  - The training process records the training duration and saves the training history and trained models.

![Screenshot 2023-11-14 at 7 21 04 PM](https://github.com/Shobhit-Singhh/Skin_cancer/assets/117563572/a8afb771-da53-414f-a229-48c8298fee7d)
**Costum CNN model:**

![Screenshot 2023-11-14 at 7 21 14 PM](https://github.com/Shobhit-Singhh/Skin_cancer/assets/117563572/0b0223de-c80c-4518-b549-307faf70d845)
**VGG Model:**

![Screenshot 2023-11-14 at 7 21 29 PM](https://github.com/Shobhit-Singhh/Skin_cancer/assets/117563572/f438b459-7a46-477b-a89a-66fe477b8376)
**ResNet Model:**


- **Performance Plotting:**
  - A function `plot_performance()` is used to visualize the training and validation accuracy/loss curves for each model.

- **Confusion Matrix:**
  - The code generates confusion matrices for each model to analyze classification performance on the test dataset.
 
    
![Screenshot 2023-11-14 at 7 26 29 PM](https://github.com/Shobhit-Singhh/Skin_cancer/assets/117563572/1dcd7d07-9dcc-4186-857f-c2e7d33f5509)
**KNN model**

![Screenshot 2023-11-14 at 7 22 02 PM](https://github.com/Shobhit-Singhh/Skin_cancer/assets/117563572/0367f043-5ff3-4034-b885-96cff2d3423a)
**Costum CNN model:**

![Screenshot 2023-11-14 at 7 22 19 PM](https://github.com/Shobhit-Singhh/Skin_cancer/assets/117563572/e4f2528d-9e7d-4994-a152-a3485cd7cfc3)
**VGG Model:**

![Screenshot 2023-11-14 at 7 22 29 PM](https://github.com/Shobhit-Singhh/Skin_cancer/assets/117563572/efb09b6e-8203-4bd1-a040-b5fa39b4366f)
**ResNet Model:**

## Saving Trained Models and Training History <a name="saving-trained-models-and-training-history"></a>

- Trained models and their corresponding training history are saved in separate files for future reference or further analysis.
- Each model's training history includes loss, accuracy, validation loss, and validation accuracy metrics.

## Visualization and Analysis <a name="visualization-and-analysis"></a>

- Heatmaps of confusion matrices are plotted using seaborn for better understanding the model's performance on the test dataset.

## Notes <a name="notes"></a>

- Ensure necessary data and dependencies are available before running the code.
- Modify the dataset paths, hyperparameters, or callbacks as required for your use case.

## File Descriptions <a name="file-descriptions"></a>

- `training_history_model.pkl`: Training history and metrics for the Simple Model.
- `trained_model.h5`: Saved trained model for the Simple Model.
- `training_history_vgg.pkl`: Training history and metrics for the VGG Model.
- `trained_model_vgg.h5`: Saved trained model for the VGG Model.
- `training_history_resnet.pkl`: Training history and metrics for the ResNet Model.
- `trained_model_resnet.h5`: Saved trained model for the ResNet Model.

## Usage <a name="usage"></a>

1. Ensure necessary libraries are installed (`tensorflow`, `scikit-learn`, `seaborn`, `numpy`, `PIL`).
2. Run the provided code sections in a Python environment.

## Contributions <a name="contributions"></a>

Contributions are welcome! If you'd like to enhance the code or documentation, feel free to fork the repository and submit a pull request.
