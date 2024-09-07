# Handwritten Digit Recognition using Pygame and Keras

This project is an interactive application that allows users to draw digits on a canvas. The drawn digit is then processed by a pre-trained Convolutional Neural Network (CNN) model built using Keras, and the predicted digit is displayed on the screen.

## Table of Contents
- Introduction
- Features
- Installation
- Usage
- Model
- Contributing
- License
- Introduction
This project demonstrates the power of deep learning in recognizing handwritten digits, a key problem in the field of computer vision. Using the MNIST dataset, a CNN model has been trained to classify digits (0-9). The application allows users to draw digits on the screen and predicts the digit based on the drawing.

## Features
- Interactive canvas using Pygame where users can draw digits.
- Real-time prediction using a pre-trained Keras CNN model.
- Option to save drawn images.
- Clears the screen on demand to allow for new drawings.
- Installation
-  Prerequisites

### Make sure you have the following installed:
- Python 3.x
- pip (Python package installer)
- Dependencies
  
### The following libraries are required to run the project:

- pygame
- numpy
- keras
- tensorflow
- opencv-python

### You can install them using pip:

```bash
pip install pygame numpy keras tensorflow opencv-python
```

### Clone the Repository

Clone the GitHub repository to your local machine using the following command:

```bash
git clone https://github.com/sadhanshetty/Handwritten-Digit-Recognition.git
```

### Model
Make sure you have a pre-trained Keras model named bestmodel.h5. If you don't have the model:

You can train a CNN on the MNIST dataset using the code in ```main.py```.
Once trained, save the model as ```bestmodel.h5``` in the same directory as the main script.

then run the ```app.py``` which provides GUI to interact

### Usage
To run the application, navigate to the directory where the project was cloned and execute the Python script:

```bash
python main.py
```
### Controls:
Mouse Left Click: Draw on the screen.
Mouse Release: Stop drawing and process the drawing for prediction.
'n' key: Clears the canvas, allowing you to draw a new digit.
Once you draw a digit, the CNN model predicts the digit, and the result will be displayed on the screen.

## Saving images
To enable image saving, set the IMAGESAVE variable in the code to True. This will save the drawn images in the directory as image.png.

## Model
The model used for predicting digits is a Convolutional Neural Network (CNN) trained on the MNIST dataset. 

The architecture consists of:

Input Layer: Accepts 28x28 grayscale images.
Convolutional Layers: Extracts features from the input image.
Average Pooling Layer: Reduces the dimensionality.
Fully Connected Layers: Classifies the image into one of the 10 digits.
You can see the full model definition and training procedure in the model_training.py file (if provided).


## License
This project is licensed under the MIT License - see the LICENSE file for details.

# Screenshots

![image](https://github.com/user-attachments/assets/4ff47b33-0581-4c6c-bc72-cd55821d04b0)
![image](https://github.com/user-attachments/assets/e9c2f24a-b6ce-4914-9d66-9ff90d8989f8)


