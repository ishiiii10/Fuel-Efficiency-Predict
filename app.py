from flask import Flask, render_template, request
import pickle
import numpy as np
import preprocess
import os

# Load models and preprocessors
with open("models/lr_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open("models/dt_model.pkl", "rb") as f:
    dt_model = pickle.load(f)

with open("models/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/imputer.pkl", "rb") as f:
    imputer = pickle.load(f)

# Load advanced model if available
stacked_model = None
encoder_advanced = None
scaler_advanced = None
imputer_advanced = None
selector = None

if os.path.exists("models/stacked_model.pkl"):
    with open("models/stacked_model.pkl", "rb") as f:
        stacked_model = pickle.load(f)
        
    with open("models/encoder_advanced.pkl", "rb") as f:
        encoder_advanced = pickle.load(f)
        
    with open("models/scaler_advanced.pkl", "rb") as f:
        scaler_advanced = pickle.load(f)
        
    with open("models/imputer_advanced.pkl", "rb") as f:
        imputer_advanced = pickle.load(f)
        
    with open("models/selector.pkl", "rb") as f:
        selector = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    # Render the home page
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/working')
def working():
    return render_template('working.html')

@app.route('/predict_form')
def predict_form():
    # Pass whether the stacked model is available
    return render_template('index.html', stacked_available=(stacked_model is not None))

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form
    model_choice = input_data.get("model")
    
    if model_choice == "stacked" and stacked_model is not None:
        # Use advanced preprocessing and model
        processed_input = preprocess.transform_input(
            input_data, 
            encoder_advanced, 
            scaler_advanced, 
            imputer_advanced, 
            advanced=True,
            selector=selector
        )
        model = stacked_model
    else:
        # Use original preprocessing and models
        processed_input = preprocess.transform_input(
            input_data, 
            encoder, 
            scaler, 
            imputer
        )
        model = lr_model if model_choice == "linear" else dt_model
    
    prediction = model.predict(processed_input.reshape(1, -1))[0]
    
    # Get model accuracy based on choice
    accuracy = None
    if model_choice == "linear":
        accuracy = "70.2%"
    elif model_choice == "decision":
        accuracy = "73.1%"
    elif model_choice == "stacked":
        accuracy = "97.7%"
        
    return render_template('result.html', 
                           prediction=round(prediction, 2),
                           model_type=model_choice,
                           accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)