# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:21:44 2024

@author: Admin
"""
#ये लोग हमारे देश के लिए खतरा हैं, इन्हें यहाँ से निकाल दो
#हे लोक आपल्या संस्कृतीसाठी धोकादायक आहेत, त्यांना येथे येऊ देऊ नका
#अस्माकं जातिं निन्दन्ति, तेषां प्रति क्रोधः प्रकटयामः।
#तिमीहरू सबै एकै जातका हो, अरूलाई यहाँ नआउने भन!
import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import fasttext  # Import FastText for embeddings

# Load the FastText model
model_path = "E:/lid.176.bin"
model = fasttext.load_model(model_path)

# Define the label mapping
label_mapping = {
    "__label__sa": "Sanskrit",
    "__label__hi": "Hindi",
    "__label__ne": "Nepali",
    "__label__mr": "Marathi"
}

# Get user input
user_input = input("Please enter the text: ")

# Check if input is empty
if not user_input.strip():
    print("Input text cannot be empty.")
else:
    # Predict language
    language = model.predict(user_input)
    
    # Extract the predicted label
    predicted_label = language[0][0]  # Get the first predicted label
    confidence_score = language[1][0]  # Get the confidence score

    # Map the predicted label to a user-friendly name
    language_name = label_mapping.get(predicted_label, "Unknown Language")

    # Print the result
    print(f"Language of the sentence: {language_name} (Confidence: {confidence_score:.2f})")

    # Load the fine-tuned model for hate speech detection
    hate_speech_model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/indic-abusive-allInOne-MuRIL")
    tokenizer_hate = AutoTokenizer.from_pretrained("Hate-speech-CNERG/indic-abusive-allInOne-MuRIL")
    
    # Tokenize the input for hate speech detection
    inputs_hate = tokenizer_hate(user_input, return_tensors="pt", padding=True, truncation=False)
    
    # Perform inference for hate speech detection
    with torch.no_grad():
        outputs_hate = hate_speech_model(**inputs_hate)

    # Get the predicted class for hate speech
    logits_hate = outputs_hate.logits
    predicted_hate_class = torch.argmax(logits_hate, dim=1).item()

    # Assuming class 1 indicates hate speech
    if predicted_hate_class == 1:  # Change this condition based on your model's output for hate speech
        print("Hate speech detected. Proceeding to target identification...")

        # Load the target identification model
        model_c = AutoModelForSequenceClassification.from_pretrained("E:/sharedtasks capstone/subtaskC edited/taskc_mBERT model")
        tokenizer_c = AutoTokenizer.from_pretrained("E:/sharedtasks capstone/subtaskC edited/taskc_mBERT model")
        
        # Tokenize the input for target identification
        inputs_target = tokenizer_c(user_input, return_tensors="pt", padding=True, truncation=True)
        
        # Perform inference for target identification
        with torch.no_grad():
            outputs_target = model_c(**inputs_target)

        # Get the predicted class for target identification
        logits_target = outputs_target.logits
        predicted_target_class = torch.argmax(logits_target, dim=1).item()

        # Print the predicted class for target identification
        print(f"Predicted target class: {predicted_target_class}")

        # Adjust moderation logic based on your understanding of the classes
        if predicted_target_class in [1, 2]:  # Assuming class 1 and 2 indicate content that needs to be moderated
            print("This content needs to be removed due to identified hate speech.")
        else:
            print("No specific target identified for moderation.")

    else:
        print("There is no hate speech in this sentence.")