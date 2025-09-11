![Progress](https://img.shields.io/badge/Progression-100%25-blue)


This project implements a handwritten digit recognition system using the MNIST dataset
. It uses TensorFlow/Keras to build and train a simple neural network model that classifies digits (0–9) from images.

📌 Features

Loads and preprocesses the MNIST dataset

Normalizes image pixel values

One-hot encodes labels for classification

Defines a Sequential Neural Network with:

Flatten input layer (28×28 → 784)

Dense layer (128 neurons, ReLU)

Dense layer (64 neurons, ReLU)

Output layer (10 neurons, Softmax)

Compiles model with Adam optimizer and Categorical Crossentropy loss

Trains the model for 10 epochs with validation on test data

Evaluates accuracy on both training and test sets

🚀 Results

Achieves ~97–98% accuracy on test data (depending on run).

Can classify unseen handwritten digits.

├── Hndwriting_recognition.ipynb   # Jupyter notebook with full implementation
├── README.md                      # Project documentation


🛠️ Requirements

Make sure you have the following installed:

Python 3.8+

TensorFlow

NumPy

Matplotlib

Jupyter Notebook

You can install dependencies with:
pip install tensorflow numpy matplotlib jupyter


▶️ Usage

1.Clone the repository:
git clone https://github.com/your-username/handwriting-recognition.git
cd handwriting-recognition

2.Open the notebook:
jupyter notebook Hndwriting_recognition.ipynb


3.Run all cells to train and evaluate the model.

📊 Example Output

Training accuracy and validation accuracy improve over 10 epochs. Final test accuracy is around 97–98%.

✨ Future Improvements

Add Convolutional Neural Networks (CNNs) for higher accuracy

Deploy the model with Flask/Streamlit for a web interface

Experiment with data augmentation
