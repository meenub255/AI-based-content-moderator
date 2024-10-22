# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 13:02:39 2024

@author: Admin
"""

import langdetect
import pandas as pd

def detect_languages_batch_eval(texts, batch_size=1000):
    """
    Detects languages for a batch of texts using the LangDetect library.

    Args:
        texts (list): List of text strings.
        batch_size (int): Number of texts to process in each batch.

    Returns:
        list: List of predicted language codes.
    """

    predicted_languages = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Convert batch elements to strings and handle NaN values
        # Handle empty strings, numbers, and special characters
        batch = [str(text) if not pd.isnull(text) and isinstance(text, str) and len(text.strip()) > 0 and any(c.isalpha() for c in text) else '' for text in batch]

        # Use LangDetect to predict languages
        # Skip empty strings
        language_codes = [langdetect.detect(text) for text in batch if text]

        predicted_languages.extend(language_codes)

        print(f"Processed {i + batch_size} out of {len(texts)} texts")  # Progress tracking

    return predicted_languages

if __name__ == "__main__":
    df1 = pd.read_csv('E:/sharedtasks capstone/subtask A_edited/taskA_(index,text(eval.csv')

    # Get all sentences from the DataFrame
    all_sentences = df1['text'].tolist()

    # Predict languages for all sentences in batches
    predicted_languages = detect_languages_batch_eval(all_sentences)

    # Add the predicted languages to the DataFrame
    df1['predicted_language'] = pd.Series(predicted_languages)

    # Save the DataFrame with predicted languages
    df1.to_csv('output_predicted_onevaldata_langdetect.csv', index=False)

from sklearn.metrics import classification_report
predicted_data = pd.read_csv('/content/output_predicted_onevaldata_langdetect.csv')
evaluation_data = pd.read_csv('/content/taskA_(index,label)eval.csv')
# Merge the two DataFrames based on the index
merged_data = pd.merge(predicted_data, evaluation_data, on='index')

# Extract true and predicted labels
true_labels = merged_data['label'].tolist()
predicted_labels = merged_data['predicted_language'].tolist()
report = classification_report(true_labels, predicted_labels)
print(report)

import langdetect
import numpy as np
import pandas as pd

def detect_languages_batch_test(texts, batch_size=1000):
    """
    Detects languages for a batch of texts using the LangDetect library.

    Args:
        texts (list): List of text strings.
        batch_size (int): Number of texts to process in each batch.

    Returns:
        list: List of predicted language codes.
    """

    predicted_languages = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Convert batch elements to strings and handle NaN values
        batch = [str(text) if not pd.isnull(text) else '' for text in batch]

        # Use LangDetect to predict languages
        language_codes = [langdetect.detect(text) for text in batch]

        predicted_languages.extend(language_codes)

        print(f"Processed {i + batch_size} out of {len(texts)} texts")  # Progress tracking

    return predicted_languages

if __name__ == "__main__":
    df1 = pd.read_csv('/content/taskA(index,text)test.csv')

    # Get all sentences from the DataFrame
    all_sentences = df1['text'].tolist()

    # Predict languages for all sentences in batches
    predicted_languages = detect_languages_batch_test(all_sentences)

    # Add the predicted languages to the DataFrame
    df1['predicted_language'] = pd.Series(predicted_languages)

    # Save the DataFrame with predicted languages
    df1.to_csv('output_predicted_ontestdata_langdetect.csv', index=False)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Assuming true_labels and predicted_labels are defined as in your previous code

# Calculate the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Create a heatmap of the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

# Assuming true_labels and predicted_labels are defined as in your previous code

# Convert labels to binary format
lb = LabelBinarizer()
lb.fit(true_labels)
y_true = lb.transform(true_labels)
y_pred = lb.transform(predicted_labels)

# Calculate AUC score for each label
auc_scores = roc_auc_score(y_true, y_pred, average=None)

# Print AUC scores for each label
for i, label in enumerate(lb.classes_):
    print(f"AUC for {label}: {auc_scores[i]}")

# Create a bar chart for AUC scores for each label
plt.figure(figsize=(10, 6))
plt.bar(range(len(auc_scores)), auc_scores)
plt.xlabel('Label')
plt.ylabel('AUC Score')
plt.title('AUC Score for Each Label')
plt.xticks(range(len(auc_scores)), lb.classes_)
plt.show()

macro_auc=roc_auc_score(y_true, y_pred, average='macro')
print("Macro AUC Score:", macro_auc)

weighted_auc=roc_auc_score(y_true, y_pred, average='weighted')
print("Weighted AUC Score:", weighted_auc)

# Calculate the overall AUC score (macro-average)
overall_auc = roc_auc_score(y_true, y_pred, average='macro')
print("Overall AUC Score:", overall_auc)
auc_types = ['Macro-average AUC', 'Weighted-average AUC', 'Overall AUC']
auc_values = [macro_auc, weighted_auc, overall_auc]

plt.figure(figsize=(8, 5))
plt.bar(auc_types, auc_values, color=['skyblue', 'lightgreen', 'coral'])
plt.ylim(0, 1)  # AUC values are between 0 and 1
plt.ylabel('AUC Score')
plt.title('Comparison of Macro-average, Weighted-average, and Overall AUC')
plt.show()
