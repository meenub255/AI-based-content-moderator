# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 16:09:30 2024

@author: Admin
"""
import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, GlobalMaxPooling1D, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import fasttext  # Import FastText for embeddings
from keras.callbacks import ReduceLROnPlateau, CSVLogger,EarlyStopping
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,log_loss
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
df0 = pd.read_csv("E:/sharedtasks capstone/subtask B_edited/subtaskB_train_edited.csv", delimiter=',', encoding='utf-8', on_bad_lines='skip')
df0.rename(columns={'tweet': 'text', 'class': 'category'}, inplace=True)

# Load evaluation dataset
df_eval = pd.read_csv('E:/sharedtasks capstone/subtask B_edited/taskB_combined_eval_edited.csv', delimiter=',')  # Adjust the path
df_eval.rename(columns={'tweet': 'text', 'class': 'category'}, inplace=True)

# Prepare evaluation data
x_eval = df_eval['text']
y_eval = df_eval['truth_label']

# Prepare data
x = df0['text']
y = df0['label']
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(x)
vocab_length = len(word_tokenizer.word_index) + 1

# Padding
max_length = max(x.apply(lambda sentence: len(word_tokenize(sentence))))  # Adjust for your languages
train_padded_sentences = pad_sequences(word_tokenizer.texts_to_sequences(x), maxlen=max_length, padding='post')
eval_padded_sentences = pad_sequences(word_tokenizer.texts_to_sequences(x_eval), maxlen=max_length, padding='post')


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
    test_size=0.25,  # Adjust the test size as needed
    random_state=42
)

# Further split the training data into training and validation sets
X_train, x_val, y_train, y_val = train_test_split(
    X_train, 
    y_train,
    test_size=0.1,  # 10% of the training data for validation
    random_state=42
)

# Define the Bi-LSTM model
def create_fine_tuned_model(vocab_length, embedding_dim, embedding_matrix):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_length, output_dim=embedding_dim, 
                        weights=[embedding_matrix], trainable=True))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(64)))  # Adding another LSTM layer
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(np.unique(y)), activation='softmax'))  # Adjust output layer based on unique classes
    return model

# Create and compile the fine-tuned model
fine_tuned_model = create_fine_tuned_model(vocab_length, embedding_dim, embedding_matrix)
fine_tuned_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
csv_logger = CSVLogger('training_log.csv', append=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the fine-tuned model
fine_tuned_history = fine_tuned_model.fit(X_train, y_train, 
                                           epochs=10,  # Adjust as necessary
                                           batch_size=32,  # Adjust as necessary
                                           validation_data=(x_val, y_val),
                                           callbacks=[reduce_lr, csv_logger, early_stopping])
# Save the model
fine_tuned_model.save('fine_tuned_model_hs.keras')  # Save in HDF5 format








#Testing data analysis
from keras.models import load_model
import numpy as np

# Load the saved model
loaded_model = load_model('C:/Users/Admin/fine_tuned_model_hs.keras')
tf.keras.utils.plot_model(loaded_model, show_shapes=True)
loaded_model.summary()
# Load the test data
test_data_truth = pd.read_csv("E:/sharedtasks capstone/subtask B_edited/taskB_combined_test_edit.csv")

tweets = test_data_truth['tweet'].values  # Replace with actual column name
test_padded_sentences = pad_sequences(word_tokenizer.texts_to_sequences(tweets), maxlen=max_length, padding='post')
predictions = loaded_model.predict(test_padded_sentences)
predicted_classes = np.argmax(predictions, axis=1)
true_labels = test_data_truth['truth_label'].values  # Replace 'label' with the actual column name
conf_matrix = confusion_matrix(true_labels, predicted_classes)
print(classification_report(true_labels, predicted_classes))
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Assuming you have already loaded your model and have the training and evaluation data
# Combine the training and evaluation data
combined_texts = pd.concat([df0['text'], df_eval['text']])
combined_labels = pd.concat([df0['label'], df_eval['truth_label']])

# Tokenize and pad the combined data
combined_padded_sentences = pad_sequences(word_tokenizer.texts_to_sequences(combined_texts), maxlen=max_length, padding='post')

# Get predictions for the combined dataset
predictions_proba_combined = loaded_model.predict(combined_padded_sentences)

# For multi-class, you may want to plot PRC for each class
num_classes = len(np.unique(combined_labels))
plt.figure(figsize=(10, 7))

for i in range(num_classes):
    # Binarize the output for the current class
    y_true_binary = (combined_labels == i).astype(int)
    precision, recall, _ = precision_recall_curve(y_true_binary, predictions_proba_combined[:, i])
    average_precision = average_precision_score(y_true_binary, predictions_proba_combined[:, i])
    
    # Plotting the Precision-Recall curve
    plt.plot(recall, precision, label=f'Class {i} (AP = {average_precision:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.show()




# Function to preprocess user input
def preprocess_input(user_input):
    word_tokenizer = Tokenizer()
    vocab_length = len(word_tokenizer.word_index) + 1
    processed_input = word_tokenizer.texts_to_sequences([user_input])
    processed_input = pad_sequences(processed_input, maxlen=max_length)  # Adjust max_length as necessary
    return processed_input

# Get user input
user_input = input("Enter your text: ")

# Preprocess the input
processed_input = preprocess_input(user_input)

# Make predictions
predictions = loaded_model.predict(processed_input)
predicted_class = np.argmax(predictions, axis=1)

# Output the prediction
print(f'Predicted class: {predicted_class[0]}')






from sklearn.metrics import f1_score

predicted_classes_combined = np.argmax(predictions_proba_combined, axis=1)

# Calculate the overall F1 score
overall_f1_score = f1_score(combined_labels, predicted_classes_combined, average='weighted')

print(f'Overall F1 Score: {overall_f1_score:.4f}')