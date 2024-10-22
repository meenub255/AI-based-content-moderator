# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 12:08:16 2024

@author: Admin
"""

import pandas as pd
import fasttext
import numpy as np
import requests
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split


df = pd.read_csv('E:/sharedtasks capstone/subtask A_edited/taskA_train.csv')
df.info()

df.isnull().sum()

df.dropna(subset=['text'],inplace=True)

df.isnull().sum()

df['label'].value_counts()

url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
filename = "lid.176.bin" 
response = requests.get(url)
if response.status_code == 200:
  with open(filename, 'wb') as f:
    f.write(response.content)
  print(f"Downloaded {filename}   successfully.")
else:
  print(f"Error downloading the file. Status code: {response.status_code}")
  
  
def train_language_model(train_data, output_model_path):
    try:
        model = fasttext.train_supervised(input=train_data, lr=0.1, epoch=100,wordNgrams=2)
        model.save_model(output_model_path)
        print("Model trained successfully and saved to:", output_model_path)
    except Exception as e:
        print("Error during model training:", e)

if __name__ == "__main__":
    train_data_path = 'E:/sharedtasks capstone/subtask A_edited/taskA_train.csv'
    output_model_path = 'language_model.bin'
    train_language_model(train_data_path, output_model_path)
    


def detect_languages_batch_eval(texts, model, batch_size=1000):
    """
    #Detects languages for a batch of texts using a FastText model.

    #Args:
        #texts (list): List of text strings.
        #model (fasttext.FastText): FastText model.
        #batch_size (int): Number of texts to process in each batch.

    #Returns:
        #list: List of predicted language codes.
    """

    predicted_languages = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        # Convert batch elements to strings and handle NaN values
        batch = [str(text) if not pd.isnull(text) else '' for text in batch]

        labels, probabilities = model.predict(batch, k=1)
        language_codes = [label[0].replace('__label__', '') for label in labels]
        label_mapping = {
            'ne': 1,
            'mr': 2,
            'sa': 3,
            'hi': 4
        }
        # Ensure all language codes are mapped to numerical values
        mapped_language_codes = [label_mapping.get(code, -1) for code in language_codes]  # Map unknown codes to -1
        predicted_languages.extend(mapped_language_codes)

        print(f"Processed {i+batch_size} out of {len(texts)} texts")  # Progress tracking

    return predicted_languages

if __name__ == "__main__":
    df1 = pd.read_csv('E:/sharedtasks capstone/subtask A_edited/taskA_(index,text(eval.csv')
    model = fasttext.load_model('lid.176.bin')

    # Get all sentences from the DataFrame
    all_sentences = df1['text'].tolist()

    # Predict languages for all sentences in batches
    predicted_languages = detect_languages_batch_eval(all_sentences, model)

    # Add the predicted languages to the DataFrame
    df1['predicted_language'] = pd.Series(predicted_languages)

    # Save the DataFrame with predicted languages
    df1.to_csv('output_predicted_onevaldata.csv', index=False)
    
def detect_languages_batch_test(texts, model, batch_size=1000):
    """
    Detects languages for a batch of texts using a FastText model.

    Args:
        texts (list): List of text strings.
        model (fasttext.FastText): FastText model.
        batch_size (int): Number of texts to process in each batch.

    Returns:
        list: List of predicted language codes.
    """

    predicted_languages = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        # Convert batch elements to strings and handle NaN values
        batch = [str(text) if not pd.isnull(text) else '' for text in batch]

        labels, probabilities = model.predict(batch, k=1)
        language_codes = [label[0].replace('__label__', '') for label in labels]
        label_mapping = {
            'ne': 1,
            'mr': 2,
            'sa': 3,
            'hi': 4
        }
        # Ensure all language codes are mapped to numerical values
        mapped_language_codes = [label_mapping.get(code, -1) for code in language_codes]  # Map unknown codes to -1
        predicted_languages.extend(mapped_language_codes)

        print(f"Processed {i+batch_size} out of {len(texts)} texts")  # Progress tracking

    return predicted_languages

if __name__ == "__main__":
    # Load the test dataset
    df_test = pd.read_csv('E:/sharedtasks capstone/subtask A_edited/taskA(index,text)test.csv')

    # Load the FastText model
    model = fasttext.load_model('lid.176.bin')

    # Get all sentences from the test DataFrame
    all_sentences_test = df_test['text'].tolist()

    # Predict languages for all sentences in the test dataset in batches
    predicted_languages_test = detect_languages_batch_eval(all_sentences_test, model)

    # Add the predicted languages to the test DataFrame
    df_test['predicted_language'] = pd.Series(predicted_languages_test)

    # Save the test DataFrame with predicted languages
    df_test.to_csv('output_predicted_test_data.csv', index=False)
    
truth=pd.read_csv("E:/sharedtasks capstone/subtask A_edited/taskA_(index,label)eval.csv")
predicted=pd.read_csv("E:/sharedtasks capstone/output_predicted_onevaldata.csv")
tl=truth['label'].tolist()
pl=predicted['predicted_language'].tolist()
label_mapping={1:1,2:2,3:3,4:4}


tl = truth['label'].tolist()
pl = predicted['predicted_language'].tolist()

# Now both tl and pl should contain numerical labels
print(classification_report(tl, pl))
cm = confusion_matrix(tl, pl)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['nepali', 'marathi', 'sanskrit', 'hindi'], yticklabels=['nepali', 'marathi', 'sanskrit', 'hindi'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


y_true_bin = label_binarize(tl, classes=list(label_mapping.values()))
y_pred_bin = label_binarize(pl, classes=list(label_mapping.values()))

# Calculate AUC for each class
auc_scores = []
for i in range(y_true_bin.shape[1]):
  auc = roc_auc_score(y_true_bin[:, i], y_pred_bin[:, i])
  auc_scores.append(auc)

# Print AUC scores for each class
for i, auc in enumerate(auc_scores):
  print(f"AUC for class {list(label_mapping.keys())[i]}: {auc}")

# Plot AUC scores for each class
plt.figure(figsize=(8, 6))
plt.bar(list(label_mapping.keys()), auc_scores)
plt.xlabel('Classes')
plt.ylabel('AUC Score')
plt.title('AUC Score for Each Class')
plt.show()

# Calculate macro-average AUC (average AUC across all classes)
macro_auc = roc_auc_score(y_true_bin, y_pred_bin, average='macro')
print(f"Macro-average AUC: {macro_auc}")

# Calculate weighted-average AUC (AUC weighted by the number of true instances for each class)
weighted_auc = roc_auc_score(y_true_bin, y_pred_bin, average='weighted')
print(f"Weighted-average AUC: {weighted_auc}")

overall_auc = roc_auc_score(y_true_bin, y_pred_bin)
print(f"Overall AUC: {overall_auc}")

auc_types = ['Macro-average AUC', 'Weighted-average AUC', 'Overall AUC']
auc_values = [macro_auc, weighted_auc, overall_auc]

plt.figure(figsize=(8, 5))
plt.bar(auc_types, auc_values, color=['skyblue', 'lightgreen', 'coral'])
plt.ylim(0, 1)  # AUC values are between 0 and 1
plt.ylabel('AUC Score')
plt.title('Comparison of Macro-average, Weighted-average, and Overall AUC')
plt.show()


