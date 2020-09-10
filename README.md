# OSIC Pulmonary Fibrosis Progression

This repository contains my baseline code for this <a href="https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression">Kaggle challenge</a>. The goal is to use a 
dataset consisting of clinical information and three dimensional CT images of patients to predict their FVC (Forced Volume Capacity). This is a useful diagnostic tool 
in pulmonary fibrosis.

This is done using a simple deep convolutional neural network.

formatData.py puts the data in a suitable format to be used in the neural network.

ANN.py contains the code that trains the neural network using TensorFlow
