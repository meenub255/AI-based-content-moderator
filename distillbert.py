# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:33:44 2024

@author: Admin
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
data_train = pd.read_csv('E:/sharedtasks capstone/subtask B_edited/subtaskB_train.csv')
data_val = pd.read_csv('E:/sharedtasks capstone/subtask B_edited/subtask-B_eval.csv')
data_test = pd.read_csv('E:/sharedtasks capstone/subtask B_edited/subtaskB(index,tweet)t.csv')

data_train.info()
data_train['label'].value_counts()

data_val.info()

data_test.info()

data_train.isna().sum()
data_val.isna().sum()
data_test.isna().sum()

train_text=data_train['tweet']
val_text=data_val['tweet']
test_text=data_test['tweet']

class_label_train=data_train['label']
class_label_val=data_val['truth_label']
#class_label_test=data_test['predicted_label']

class_label_train.value_counts()
class_label_val.value_counts()

classes_list=['non_hate','hate']
label_index_train=class_label_train.replace({0: 'non_hate', 1: 'hate'})
label_index_val=class_label_val.replace({0: 'non_hate', 1: 'hate'})


import os
os.environ['TF_USE_LEGACY_KERAS'] = "True"
import ktrain
from ktrain import text
#MODEL_NAME = 'google-bert/bert-base-multilingual-cased'
MODEL_NAME = 'roberta-base'
#MODEL_NAME = 'distilbert-base-multilingual-cased'
t=text.Transformer(MODEL_NAME,maxlen=30,class_names=classes_list)
trn=t.preprocess_train(np.array(train_text),np.array(class_label_train))
test=t.preprocess_test(np.array(val_text),np.array(class_label_val))
model_hs_rb=t.get_classifier()
learner=ktrain.get_learner(model_hs_rb,train_data=trn,val_data=test,batch_size=32)

from tensorflow.keras.callbacks import ModelCheckpoint
filepath = "Subtask-B roberta"
checkpoint=ModelCheckpoint(filepath,monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')
callbacks_list=[checkpoint]
learner.fit_onecycle(0.005, 10, verbose=2, callbacks=callbacks_list)


filepath = "Subtask-B roberta-base"
learner.save_model('Subtask-B roberta-base')
model_hs=t.get_classifier()
model_hs.load_weights(filepath)

learner.validate(class_names=t.get_classes())#training classification report

predictor = ktrain.get_predictor(learner.model, preproc=t)

predict = predictor.predict(val_text.values)
labels = [1 if label == 'hate' else 0 for label in predict]

predictions_df = pd.DataFrame({
    'index': data_val['index'],  # Use the index from test_text DataFrame
    'label_predicted': labels
})

predictions_df['label_predicted'].value_counts()
predictions_df.to_csv('subtask-B_distilbert_prediction_test.csv', index=False)

tl=data_val['truth_label']
val=pd.read_csv("E:/sharedtasks capstone/subtask B_edited/subtask-B_distilbert_prediction_eval.csv")
pl=val['label_predicted']
print(classification_report(tl, pl))#validation classification report


