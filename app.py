import streamlit as st
import torch
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import matplotlib.pyplot as plt

# Load YOLOv5 model (pretrained)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load LSTM model and tokenizer
caption_model = load_model('caption_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_length = 10  # Set based on your training
vocab_size = len(tokenizer.word_index) + 1

# Helper function to get word from token id
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Generate caption using LSTM model
def generate_caption(model, tokenizer, seed_text, max_length):
    in_text = seed_text
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict(sequence, verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    final_caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    return final_caption

# Detect objects using YOLO
def detect_objects(image_path):
    results = yolo_model(image_path)
    labels = results.pandas().xyxy[0]['name'].tolist()
    return labels

# Streamlit app
def main():
    st.title("Image Caption Generator with YOLO and LSTM")

    st.write("Upload an image to generate a caption:")

    # Image upload section
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image.", use_column_width=True)
        
        # Detect objects in the image
        labels = detect_objects(uploaded_image)
        st.write("Detected Objects: ", labels)

        if labels:
            # Generate caption from detected objects
            seed_text = "startseq " + ' '.join(labels) + " endseq"
            caption = generate_caption(caption_model, tokenizer, seed_text, max_length)
            st.write("Generated Caption: ", caption)
        else:
            st.write("No objects detected.")

if __name__ == '__main__':
    main()
