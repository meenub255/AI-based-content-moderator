# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 18:48:46 2024

@author: Admin
"""

import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import numpy as np
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, GlobalMaxPooling1D, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import fasttext  # Import FastText for embeddings
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf

df0 = pd.read_csv('E:/sharedtasks capstone/subtask B_edited/subtaskB_train.csv', delimiter=',')  # Ensure your dataset has Hindi, Marathi, Sanskrit, Nepali
df0.rename(columns={'tweet': 'text', 'class': 'category'}, inplace=True)


# Prepare data
x = df0['text']
y = df0['label']
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(x)
vocab_length = len(word_tokenizer.word_index) + 1

# Padding
max_length = max(x.apply(lambda sentence: len(word_tokenize(sentence))))  # Adjust for your languages
train_padded_sentences = pad_sequences(word_tokenizer.texts_to_sequences(x), maxlen=max_length, padding='post')

def load_vectors(filepath):
  vectors = {}
  with open(filepath, 'r', encoding='utf-8') as f:  # Open the .vec file
    next(f)  # Skip the first line (header)
    for line in f:
      parts = line.rstrip().split(' ')
      word = parts[0]
      vector = np.array(parts[1:], dtype=np.float32)  # Convert vector elements to float
      vectors[word] = vector
  return vectors

fasttext_vectors = {
    'hi': load_vectors('E:/sharedtasks capstone/subtask B_edited/cc.hi.300.vec'),  # Hindi FastText vectors
    'mr': load_vectors('E:/sharedtasks capstone/subtask B_edited/cc.mr.300.vec'),  # Marathi FastText vectors
    'ne': load_vectors('E:/sharedtasks capstone/subtask B_edited/cc.ne.300.vec'),  # Nepali FastText vectors
    'sa': load_vectors('E:/sharedtasks capstone/subtask B_edited/cc.sa.300.vec') }  # Sanskrit FastText vectors

embedding_dim = 300  # Adjust based on the FastText model
embedding_matrix = np.zeros((vocab_length, embedding_dim))

for word, index in word_tokenizer.word_index.items():
    # Try to get the embedding from each language model
    for lang, model in fasttext_vectors.items():
        try:
            embedding_vector = model[word]  # Get the embedding for the word
            embedding_matrix[index] = embedding_vector
            break  # If found, break out of the loop
        except KeyError:
            continue  # If the word is not in the FastText model, skip it

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    train_padded_sentences, 
    y, 
    test_size=0.25,
    random_state=42
)

# Further split the training data into training and validation sets
X_train, x_val, y_train, y_val = train_test_split(
    X_train, 
    y_train,
    test_size=0.1,
    random_state=42
)

# Define the Bi-LSTM model
def create_bi_lstm_model(vocab_length, embedding_dim, embedding_matrix):
    model = Sequential()
    
    # Embedding layer
    model.add(Embedding(input_dim=vocab_length, output_dim=embedding_dim, 
                        weights=[embedding_matrix], trainable=False))
    
    # Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.5))
    
    # Global max pooling layer
    model.add(GlobalMaxPooling1D())
    
    # Fully connected layer
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    
    # Output layer
    model.add(Dense(len(y.unique()), activation='softmax'))  # Adjust based on the number of classes

    return model

# Create the model
model = create_bi_lstm_model(vocab_length, embedding_dim, embedding_matrix)

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Define callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
csv_logger = CSVLogger('training_log.csv', append=True)

# Train the model
history = model.fit(X_train, y_train, 
                    epochs=10,  # Adjust as necessary
                    batch_size=32, # Adjust as necessary
                    validation_data=(x_val, y_val),
                    callbacks=[reduce_lr, csv_logger])

#model.summary()#model summary for the trained model
#tf.keras.utils.plot_model(model, show_shapes=True,to_file='model.png')


model_save_path = 'bilstm_model.keras'  # You can choose any filename and path
model.save(model_save_path)
print(f'Model saved to {model_save_path}')

loaded_model = load_model(model_save_path)

# Verify that the model is loaded correctly
#loaded_model.summary()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Get predicted probabilities for the positive class
y_scores = model.predict(X_test)  # This will give you probabilities for all classes
y_scores_positive = y_scores[:, 1]  # Assuming the positive class is at index 1

# Compute precision and recall
precision, recall, thresholds = precision_recall_curve(y_test, y_scores_positive)

# Calculate the average precision score
average_precision = average_precision_score(y_test, y_scores_positive)

# Plotting the Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='Precision-Recall curve (area = {:.2f})'.format(average_precision))
plt.title('Precision-Recall Curve for Bi-LSTM with fastText Embeddings')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.grid()
plt.show()


import seaborn as sns
df_eval=pd.read_csv("E:/sharedtasks capstone/subtask B_edited/subtask-B_eval.csv")
texts_eval=df_eval['tweet']
eval_sequences = word_tokenizer.texts_to_sequences(texts_eval)
test_padded_sentences = pad_sequences(eval_sequences, maxlen=max_length, padding='post')
tl=df_eval['truth_label']
preds = np.argmax(loaded_model.predict(test_padded_sentences), axis=-1)
print(classification_report(tl, preds))
cm=confusion_matrix(tl, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['True 0', 'True 1'])

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()

df_test=pd.read_csv("E:/sharedtasks capstone/subtask B_edited/subtaskB(index,tweet)t.csv")
texts_test=df_test['tweet']
test_sequences=word_tokenizer.texts_to_sequences(texts_test)
test_data_sequences=pad_sequences(test_sequences,maxlen=max_length,padding='post')
df_test_tl=pd.read_csv("E:/sharedtasks capstone/subtask B_edited/subtaskB(index,label)t.csv")
tl_test=df_test_tl['label']
preds_test=np.argmax(loaded_model.predict(test_data_sequences),axis=-1)
print(classification_report(tl_test, preds_test))#testing classification report
cm1=confusion_matrix(tl_test, preds_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['True 0', 'True 1'])

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()

#for eval data
from sklearn.metrics import precision_score, recall_score, f1_score
overall_precision = precision_score(tl, preds, average='weighted')  # Use 'macro' for unweighted average
overall_recall = recall_score(tl, preds, average='weighted')        # Use 'macro' for unweighted average
overall_f1 = f1_score(tl, preds, average='weighted')                # Use 'macro' for unweighted average

print(f'Overall Precision: {overall_precision:.4f}')
print(f'Overall Recall: {overall_recall:.4f}')
print(f'Overall F1-Score: {overall_f1:.4f}')


#for test data
report = classification_report(tl_test, preds_test, output_dict=True)
overall_precision = report['weighted avg']['precision']
overall_recall = report['weighted avg']['recall']
overall_f1_score = report['weighted avg']['f1-score']

print(f'Overall Precision: {overall_precision:.4f}')
print(f'Overall Recall: {overall_recall:.4f}')
print(f'Overall F1-Score: {overall_f1_score:.4f}')


#AUC SCORE
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# Get predicted probabilities
pred_probs = loaded_model.predict(test_data_sequences)

# Calculate AUC for each label
auc_label_0 = roc_auc_score(tl_test == 0, pred_probs[:, 0])
auc_label_1 = roc_auc_score(tl_test == 1, pred_probs[:, 1])

# Print AUC scores
print(f'AUC Score for Label 0: {auc_label_0:.4f}')
print(f'AUC Score for Label 1: {auc_label_1:.4f}')

# Individual AUC scores
auc_label_0 = 0.5269
auc_label_1 = 0.5258

# Calculate overall AUC score
overall_auc = (auc_label_0 + auc_label_1) / 2

# Print overall AUC score
print(f'Overall AUC Score: {overall_auc:.4f}')


import matplotlib.pyplot as plt

# AUC scores for each label
auc_scores = [auc_label_0, auc_label_1]
labels = ['Label 0', 'Label 1']

# Create a bar chart
plt.figure(figsize=(8, 5))
plt.bar(labels, auc_scores, color=['blue', 'orange'])

# Add labels and title
plt.ylim(0, 1)  # AUC values range from 0 to 1
plt.ylabel('AUC Score')
plt.title('AUC Scores for Each Label')

# Display the AUC values on top of the bars
for i, score in enumerate(auc_scores):
    plt.text(i, score + 0.02, f'{score:.4f}', ha='center')

# Show the plot
plt.show()

#ROC CURVE
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Get predicted probabilities
pred_probs = loaded_model.predict(test_data_sequences)

# Calculate ROC curve for each label
fpr_0, tpr_0, thresholds_0 = roc_curve(tl_test == 0, pred_probs[:, 0])
fpr_1, tpr_1, thresholds_1 = roc_curve(tl_test == 1, pred_probs[:, 1])

# Calculate AUC for each label
roc_auc_0 = auc(fpr_0, tpr_0)
roc_auc_1 = auc(fpr_1, tpr_1)

# Print TPR and FPR at specific thresholds for Label 0
print("Label 0 - TPR and FPR at various thresholds:")
for threshold, fpr, tpr in zip(thresholds_0, fpr_0, tpr_0):
    print(f"Threshold: {threshold:.2f}, FPR: {fpr:.4f}, TPR: {tpr:.4f}")

# Print TPR and FPR at specific thresholds for Label 1
print("\nLabel 1 - TPR and FPR at various thresholds:")
for threshold, fpr, tpr in zip(thresholds_1, fpr_1, tpr_1):
    print(f"Threshold: {threshold:.2f}, FPR: {fpr:.4f}, TPR: {tpr:.4f}")

# Plot ROC curves in the same graph
plt.figure(figsize=(10, 6))
plt.plot(fpr_0, tpr_0, label=f'Label 0 (AUC = {roc_auc_0:.2f})')
plt.plot(fpr_1, tpr_1, label=f'Label 1 (AUC = {roc_auc_1:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Plot ROC curves in different graphs
plt.figure(figsize=(10, 6))
plt.plot(fpr_0, tpr_0, label=f'Label 0 (AUC={roc_auc_0:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Label 0')
plt.legend(loc='lower right')
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(fpr_1, tpr_1, label='Label 1 (AUC={roc_auc_1:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Label 1')
plt.legend(loc='lower right')
plt.grid()
plt.show()
overall_auc = (roc_auc_0 + roc_auc_1) / 2

# Print overall AUC score
print(f'Overall AUC Score: {overall_auc:.4f}')

def preprocess_input(user_text, tokenizer, max_length):
    # Tokenize the input text
    tokenized_text = tokenizer.texts_to_sequences([user_text])
    # Pad the tokenized text
    padded_text = pad_sequences(tokenized_text, maxlen=max_length, padding='post')
    return padded_text

# Get user input
user_input = input("Enter your text: ")

# Preprocess the user input
padded_user_input = preprocess_input(user_input, word_tokenizer, max_length)

# Make prediction
predicted_class_probabilities = loaded_model.predict(padded_user_input)
predicted_label_index = np.argmax(predicted_class_probabilities, axis=1)  # Get the index of the class with the highest probability

# Map the predicted label back to the original class
# Assuming you have a mapping from label index to actual class names
label_mapping = {0: 'non-hate', 1: 'hate'}  # Adjust based on your actual classes
predicted_class_name = label_mapping.get(predicted_label_index[0], 'Unknown Class')

# Output the prediction
print(f"Predicted class: {predicted_class_name}")

