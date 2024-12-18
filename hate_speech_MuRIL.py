# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 21:01:29 2024

@author: Admin
"""

import os 
import seaborn as sns
import matplotlib.pyplot as plt
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report,confusion_matrix,f1_score,precision_recall_curve, auc
# Load the model and tokenizer
hate_speech_model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/indic-abusive-allInOne-MuRIL")
tokenizer_hate = AutoTokenizer.from_pretrained("Hate-speech-CNERG/indic-abusive-allInOne-MuRIL")
hate_speech_model.eval()
# Load evaluation data
eval_data = pd.read_csv("E:/sharedtasks capstone/subtask B_edited/taskB_combined_eval_edited.csv")
eval_texts = eval_data['tweet'].tolist()

# Define batch size
batch_size = 32  # You can adjust this based on your memory capacity

# Prepare to store results
predicted_labels = []

# Process texts in batches
for i in range(0, len(eval_texts), batch_size):
    batch_texts = eval_texts[i:i + batch_size]
    inputs = tokenizer_hate(batch_texts, padding=True, truncation=True, return_tensors="pt")

    # Make predictions
    with torch.no_grad():
        outputs = hate_speech_model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    # Map predictions to labels
    label_map = {0: "Non-Hate", 1: "Hate"}
    predicted_labels.extend([label_map[pred.item()] for pred in predictions])

# Create a DataFrame with results
results_df = pd.DataFrame({
    'Text': eval_texts,
    'Prediction': predicted_labels
})

# Save the DataFrame to a CSV file
results_df.to_csv('hate_speech_predictions_MuRIL.csv', index=False)
predicted_label=results_df['Prediction'].tolist()
label_map = {"Non-Hate":0, "Hate":1}
predicted_label_string = [label_map[pred] for pred in predicted_label]

truth_label=eval_data['truth_label'].tolist()
print(classification_report(truth_label,predicted_label_string))
cm = confusion_matrix(truth_label, predicted_label_string)

# Step 4: Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Hate", "Hate"], yticklabels=["Non-Hate", "Hate"])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()

test_data = pd.read_csv("E:/sharedtasks capstone/subtask B_edited/taskB_combined_test_edit.csv")
test_texts = test_data['tweet'].tolist()

# Define batch size
batch_size = 32  # You can adjust this based on your memory capacity

# Prepare to store results
predicted_labels = []
predicted_probabilities = []

# Process texts in batches
for i in range(0, len(test_texts), batch_size):
    batch_texts = test_texts[i:i + batch_size]
    inputs = tokenizer_hate(batch_texts, padding=True, truncation=False, return_tensors="pt")

    # Make predictions
    with torch.no_grad():
        outputs = hate_speech_model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        probabilities = torch.softmax(logits, dim=-1)  # Get probabilities

    # Map predictions to labels
    label_map = {0: "Non-Hate", 1: "Hate"}
    predicted_labels.extend([label_map[pred.item()] for pred in predictions])
    predicted_probabilities.extend(probabilities[:, 1].tolist())  # Get probabilities for the "Hate" class

# Create a DataFrame with results
results_df1 = pd.DataFrame({
    'Text': test_texts,
    'Prediction': predicted_labels
})

# Save the DataFrame to a CSV file
results_df1.to_csv('hate_speech_predictions_MuRIL_test.csv', index=False)

# Prepare for classification report
truth_label1 = test_data['truth_label'].tolist()
predicted_labels=results_df1['Prediction'].tolist()
# Convert truth labels to numeric
label_map = {0:"Non-Hate",1:"Hate"}
truth_label_numeric1 = [label_map[label] for label in truth_label1]
# Print the classification report
print(classification_report(truth_label_numeric1, predicted_labels))

# Calculate the confusion matrix
cm1 = confusion_matrix(truth_label_numeric1, predicted_labels)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Hate", "Hate"], yticklabels=["Non-Hate", "Hate"])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()




overall_f1_score = f1_score(truth_label1, predicted_labels, average='weighted')
print(f'Overall F1 Score: {overall_f1_score:.4f}')

precision, recall, _ = precision_recall_curve(truth_label1, predicted_probabilities)
pr_auc = auc(recall, precision)

# Plot the Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='PR Curve (AUC = {:.2f})'.format(pr_auc))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.show()
