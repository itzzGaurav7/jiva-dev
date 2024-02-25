import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from pydantic import BaseModel
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

model = pickle.load(open('cbheart1.pkl', 'rb'))

class Scoringitem(BaseModel):
    Age: int
    Sex: int
    RestingBP: int
    FastingBS: int
    MaxHR: int
    ExerciseAngina: int
    ChestPainType_0: int
    ChestPainType_1: int
    ChestPainType_2: int
    ChestPainType_3: int

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the values from request.form as floats
    int_features = [float(x) for x in request.form.values()]

    # Ensure int_features has exactly 10 values
    if len(int_features) < 10:
        # Pad int_features with default or placeholder values
        default_values = [0.0] * (10 - len(int_features))
        int_features.extend(default_values)

    # Now int_features will have exactly 10 values
    final_features = pd.DataFrame([int_features], columns=['Age', 'Sex', 'RestingBP', 'FastingBS', 'MaxHR', 'ExerciseAngina', 'ChestPainType_0', 'ChestPainType_1', 'ChestPainType_2', 'ChestPainType_3'])

    prediction = model.predict(final_features)
    return render_template('home.html', prediction_text="Heart disease: {}".format(prediction[0]))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)

        # Extract numerical values from JSON data
        data = [float(x) for x in data.values()]

        # Create a DataFrame
        final_features = pd.DataFrame([data], columns=['Age', 'Sex', 'RestingBP', 'FastingBS', 'MaxHR', 'ExerciseAngina', 'ChestPainType_0', 'ChestPainType_1', 'ChestPainType_2', 'ChestPainType_3'])

        # Make predictions
        prediction = model.predict(final_features)
        
        # Convert the NumPy array to a Python list
        prediction_1 = model.predict_proba(final_features)
        prediction_list = prediction_1[0].tolist()

        # Ensure that the list only contains JSON serializable types
        for item in prediction_list:
            if not isinstance(item, (int, float)):
                raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

        # Return the prediction in JSON format
        return jsonify({"prediction_probability": prediction_list},{"prediction": int(prediction[0])})
               
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
