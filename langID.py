# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 12:25:25 2024

@author: Admin
"""

import langid
import numpy as np
import pandas as pd

def detect_languages_batch_eval_langid(texts, batch_size=1000):
    """
    Detects languages for a batch of texts using the LangID library.

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

        # Use LangID to predict languages
        language_codes = [langid.classify(text)[0] for text in batch]

        # Map language codes to numerical values (if needed)
        label_mapping = {
            'ne': 1,
            'mr': 2,
            'sa': 3,
            'hi': 4
        }
        mapped_language_codes = [label_mapping.get(code, -1) for code in language_codes]  # Map unknown codes to -1
        predicted_languages.extend(mapped_language_codes)

        print(f"Processed {i + batch_size} out of {len(texts)} texts")  # Progress tracking

    return predicted_languages

if __name__ == "__main__":
    df1 = pd.read_csv('/content/taskA_(index,text(eval.csv')

    # Get all sentences from the DataFrame
    all_sentences = df1['text'].tolist()

    # Predict languages for all sentences in batches
    predicted_languages = detect_languages_batch_eval_langid(all_sentences)

    # Add the predicted languages to the DataFrame
    df1['predicted_language'] = pd.Series(predicted_languages)

    # Save the DataFrame with predicted languages
    df1.to_csv('output_predicted_onevaldata_langid.csv', index=False)

from sklearn.metrics import classification_report
df1=pd.read_csv("/content/taskA_(index,label)eval.csv")
df2=pd.read_csv("/content/output_predicted_onevaldata_langid.csv")
# Assuming 'df1' contains the true labels and predictions
y_true = df1['label']  # Replace 'label' with the actual column name for true labels
y_pred = df2['predicted_language']  # Replace 'predicted_language' with the actual column name for predictions

label_mapping = {
    'ne': 1,
    'mr': 2,
    'sa': 3,
    'hi': 4
}
y_true_numerical = y_true.map(label_mapping) #Map string labels to numerical labels


# Generate the classification report
report = classification_report(y_true_numerical, y_pred) #Use numerical labels for both y_true and y_pred

# Print the report
print(report)

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

# ... (rest of your code) ...

# Ensure both y_true and y_pred are numerical
label_mapping = {
    'ne': 1,
    'mr': 2,
    'sa': 3,
    'hi': 4
}
# Convert y_true to numerical labels using the mapping
y_true_numerical = y_true.map(label_mapping)

# Calculate Confusion Matrix using numerical labels for both
confusion_mat = confusion_matrix(y_true_numerical, y_pred)

# Create a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

# Assuming y_true_numerical and y_pred are already defined

# Convert y_true_numerical to one-hot encoded format
y_true_binarized = label_binarize(y_true_numerical, classes=[1, 2, 3, 4])

# Convert y_pred to one-hot encoded format
# (assuming y_pred contains numerical labels corresponding to the classes)
y_pred_binarized = label_binarize(y_pred, classes=[1, 2, 3, 4])

# Calculate AUC for each class and the average AUC
auc_scores = []
for i in range(y_true_binarized.shape[1]):
  try:
    auc = roc_auc_score(y_true_binarized[:, i], y_pred_binarized[:, i])
    auc_scores.append(auc)
  except ValueError:
    print(f"Unable to calculate AUC for class {i+1} due to insufficient data")


if auc_scores:
  average_auc = np.mean(auc_scores)
  print(f"Average AUC: {average_auc:.4f}")

  # Print AUC for each class
  for i, auc in enumerate(auc_scores):
    print(f"AUC for class {i+1}: {auc:.4f}")
    
