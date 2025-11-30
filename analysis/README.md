# Fashion MNIST Classification in R

This project builds Perceptron and SVM models to classify the Fashion-MNIST dataset. Each image is a 28×28 grayscale pixel grid representing clothing categories such as t-shirt, trousers, pullover, coat, sandal, bag, etc.

## Problem Statement

The goal is to predict the correct fashion category from pixel values using two classification methods: Perceptron and Support Vector Machines. Model performance is evaluated using misclassification rate, confusion matrices, tuning experiments, and class-wise difficulty analysis.

## Steps Performed

• Load Fashion-MNIST training and testing data  
• Inspect dataset size and class distribution  
• Convert label column to factor  
• Train a Perceptron model and compute misclassification rate  
• Train an SVM classifier with linear kernel  
• Print confusion matrices for both models  
• Tune Perceptron (hidden units, iterations)  
• Tune SVM using different kernels  
• Identify which fashion classes are easier or harder to classify  
• Optionally visualize a random training image  

## Tools

R, simpleNeural, e1071
