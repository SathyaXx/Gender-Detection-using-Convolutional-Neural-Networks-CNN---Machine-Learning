from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)
model_path = r'C:\Users\SATHYA NARAYANAN .B\Downloads\GenderData\gender_detection_model.h5'
model = load_model(model_path)


# Preprocess image function
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (100, 100))
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=3)
    return image

@app.route('/')
def index():
    return render_template('imageform.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    preprocessed_image = preprocess_image(image)

    prediction = model.predict(preprocessed_image)
    if prediction[0][0] < 0.5:
        result = 'Male'
    else:
        result = 'Female'

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run()
    