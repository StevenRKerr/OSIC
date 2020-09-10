# OSIC Pulmonary Fibrosis Progression

This repository contains my baseline code for this <a href="https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression">Kaggle challenge</a>. The goal is to use a 
dataset consisting of patients' clinical information and a three dimensional CT images of their lungs to predict their FVC (Forced Volume Capacity). This is a useful diagnostic tool in pulmonary fibrosis. I do this using a simple convolutional neural network.

- formatData.py puts the data in a suitable format to be used in the neural network.
- ANN.py contains the code that trains the neural network using TensorFlow
