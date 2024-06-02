import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import request, jsonify, Flask
import os
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)



model = load_model('Path to your trained model')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = cv2.resize(img, (64, 64))  
    img = img / 255.0  
    return np.expand_dims(img, axis=0) 

def preprocess_and_predict_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    
    prediction = model.predict(preprocessed_image)
    
    class_names = ['Real', 'Fake']
    predicted_class = class_names[np.argmax(prediction)]
    prediction_probability = prediction[0][0]
    
    return predicted_class, prediction_probability

@app.route('/')
def index():
    return 'Index Page'

@app.route('/classify_image', methods=['POST','GET'])
def classify_image():
    try:
        image_file = request.files['image']
        print('++++ image ',image_file)
        image_path = 'temp_image.jpg'  
        image_file.save(image_path)  
        
        prediction_result, prediction_probability = preprocess_and_predict_image(image_path)
        
        label = "Fake" if prediction_probability < 0.5 else "Real"
        if(label=="Fake"):
            prediction_probability+=1;
            if(prediction_probability>1):
                prediction_probability=1;
        else:
            prediction_probability-=1;



        os.remove(image_path)
        
        return jsonify({'prediction': label, 'probability': float(abs(prediction_probability))})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
