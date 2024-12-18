# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 10:21:41 2024

@author: Admin
"""

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve, auc,f1_score,confusion_matrix

# Load the fine-tuned model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("E:/fine_tuned_model")
tokenizer = DistilBertTokenizer.from_pretrained("E:/fine_tuned_model")
model.eval()  # Set the model to evaluation mode

# Load the test dataset
test = pd.read_csv("E:/sharedtasks capstone/subtask B_edited/taskB_combined_test_edit.csv")
test_texts = test["tweet"].tolist()

# Define batch size
batch_size = 16  # Adjust based on your memory capacity

# Prepare a label map
label_map = {0: "Non-Hate", 1: "Hate"}  # Adjust based on your labels

# List to store predictions
predicted_labels = []

# Process the data in batches
for i in range(0, len(test_texts), batch_size):
    batch_texts = test_texts[i:i + batch_size]
    
    # Tokenize the batch
    inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(**inputs)

    # Get the predicted class indices
    predictions = torch.argmax(outputs.logits, dim=1)

    # Map predictions to labels
    predicted_labels.extend([label_map[pred.item()] for pred in predictions])

test['predicted_hate_speech'] = predicted_labels

# Print the results
test.to_csv("subtaskB_with_predictions.csv", index=False)

from sklearn.metrics import classification_report,accuracy_score
test_label=pd.read_csv("E:/sharedtasks capstone/subtask B_edited/subtaskB_with_predictions.csv")
predicted_label= test_label['predicted_1_speech'].tolist()  # Adjust this line to get the true labels
truth_label=test_label['truth_label'].tolist()
# Generate the classification report
report = classification_report(truth_label, predicted_label, target_names=list(label_map.values()), output_dict=False)
print(report)

predicted_probabilities = np.random.rand(len(truth_label))  # Replace with actual probabilities

# Calculate precision and recall
precision, recall, _ = precision_recall_curve(truth_label, predicted_probabilities)

# Calculate the area under the precision-recall curve
pr_auc = auc(recall, precision)

# Plot the precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='PR curve (AUC = {:.2f})'.format(pr_auc))
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.grid()
plt.show()

overall_f1 = f1_score(truth_label, predicted_label, average='weighted')  # Use 'macro' or 'micro' as needed
print(f'Overall F1 Score: {overall_f1:.2f}')

cm = confusion_matrix(truth_label, predicted_label)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(truth_label), yticklabels=np.unique(truth_label))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()



user_input = input("enter the text:")  # Replace with the actual input text

# Tokenize the input
inputs = tokenizer(user_input, padding=True, truncation=True, return_tensors="pt")

# Make prediction
with torch.no_grad():  # Disable gradient calculation
    outputs = model(**inputs)

# Get the predicted class index
predicted_class_index = torch.argmax(outputs.logits, dim=1).item()

# Prepare a label map
label_map = {0: "Non-Hate", 1: "Hate"}  # Adjust based on your labels

# Map the predicted class index to the label
predicted_label = label_map[predicted_class_index]

# Print the result
print(f"Predicted label: {predicted_label}")