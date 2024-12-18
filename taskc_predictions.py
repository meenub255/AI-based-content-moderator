# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:57:03 2024

@author: Lenovo
"""
import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.metrics import precision_recall_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from transformers import BertForSequenceClassification,BertTokenizer
from sklearn.metrics import classification_report,confusion_matrix
test_data=pd.read_csv("E:/sharedtasks capstone/subtaskC edited/taskC_test_edited.csv")
model=BertForSequenceClassification.from_pretrained("E:/sharedtasks capstone/subtaskC edited/taskc_mBERT model")
tokenizer=BertTokenizer.from_pretrained("E:/sharedtasks capstone/subtaskC edited/taskc_mBERT model")
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():  # Disable gradient calculations for inference
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)
    return predictions.item()

# Make predictions and store them in a new column
predictions_taskC = []
for tweet in test_data['tweet']:
    predictions_taskC.append(predict(tweet))

 test_data['prediction_taskC'] = predictions_taskC
print(test_data.head())

truth_labels=test_data['truth_label']
predicted_labels=test_data['prediction_taskC']
print(classification_report(truth_labels,predicted_labels))

conf_matrix = confusion_matrix(truth_labels, predicted_labels)


# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(truth_labels), 
            yticklabels=np.unique(truth_labels))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import label_binarize

# Assuming your truth labels are in `truth_labels` and are integers representing class labels
# Binarize the truth labels for multiclass
n_classes = len(np.unique(truth_labels))
y_true_binarized = label_binarize(truth_labels, classes=np.arange(n_classes))

# Get the predicted probabilities for each class
with torch.no_grad():
    inputs = tokenizer(test_data['tweet'].tolist(), return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1).numpy()  # Get probabilities for all classes

# Initialize lists to hold precision, recall, and AUC for each class
precision = {}
recall = {}
pr_auc = {}

# Calculate precision-recall for each class
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], probabilities[:, i])
    pr_auc[i] = auc(recall[i], precision[i])

# Plot Precision-Recall curve for each class
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(recall[i], precision[i], marker='.', label='Class {} (area = {:.2f})'.format(i, pr_auc[i]))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Multiclass')
plt.legend()
plt.grid()
plt.show()