from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from pydantic import BaseModel
import pandas as pd
from flask_cors import CORS



model = pickle.load(open('LGBM.pkl', 'rb'))
model1 = pickle.load(open('LGBM1.pkl', 'rb'))
model2 = pickle.load(open('LGBM2.pkl','rb'))

app = Flask(__name__)

CORS(app)

class Scoringitem(BaseModel):
    
    Age:int
    Gender_Female: bool
    Gender_Male : bool
    Polyuria_No : bool
    Polyuria_Yes: bool
    Polydipsia_No: bool
    Polydipsia_Yes : bool
    suddenweightloss_No: bool
    suddenweightloss_Yes : bool
    weakness_No: bool
    weakness_Yes: bool
    Polyphagia_No: bool
    Polyphagia_Yes: bool
    Genitalthrush_No: bool
    Genitalthrush_Yes: bool
    visualblurring_No: bool
    visualblurring_Yes: bool
    Itching_No: bool
    Itching_Yes: bool
    Irritability_No: bool
    Irritability_Yes: bool
    delayedhealing_No : bool
    delayedhealing_Yes: bool
    partialparesis_No: bool
    partialparesis_Yes: bool
    musclestiffness_No: bool
    musclestiffness_Yes: bool
    Alopecia_No: bool
    Alopecia_Yes: bool
    Obesity_No : bool
    Obesity_Yes: bool
    
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])

def predict():
    data = request.form.to_dict()
    print("Form Data:", data)

    # Ensure all features are present
    for feature in Scoringitem.__annotations__.keys():
        if feature not in data:
            return render_template('home.html', prediction_text="Missing feature: {}".format(feature))

   
    for key, value in data.items():
        if(key == 'Age'):
            data[key]= int(value)
        if key != 'Age':  
            data[key] = bool(int(value))
        
    final_features = pd.DataFrame([data])
    yhat1 = model.predict(final_features)
    yhat2 = model1.predict(final_features)
    yhat3 = model2.predict(final_features)
    yhat = (yhat1 + yhat2 + yhat3) / 3.0

    return render_template('home.html', prediction_text="Do you have diabetes: {}".format(yhat[0]))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        
        data = request.get_json(force=True)

        
        data = [float(x) for x in data.values()]

        # Create a DataFrame
        final_features = pd.DataFrame([data], columns=[ 'Age' , 'Gender_Female',
    'Gender_Male ',
    'Polyuria_No ' ,
    'Polyuria_Yes' ,
    'Polydipsia_No' ,
    'Polydipsia_Yes ', 
    'suddenweightloss_No' ,
    'suddenweightloss_Yes ', 
    'weakness_No' ,
    'weakness_Yes' ,
    'Polyphagia_No' ,
    'Polyphagia_Yes' ,
    'Genitalthrush_No', 
    'Genitalthrush_Yes', 
    'visualblurring_No', 
    'visualblurring_Yes', 
    'Itching_No', 
    'Itching_Yes' ,
    'Irritability_No', 
    'Irritability_Yes', 
    'delayedhealing_No ', 
    'delayedhealing_Yes', 
    'partialparesis_No' ,
    'partialparesis_Yes', 
    'musclestiffness_No', 
    'musclestiffness_Yes', 
    'Alopecia_No', 
    'Alopecia_Yes', 
    'Obesity_No ' ,
    'Obesity_Yes' 
    ])

    
        prediction = model.predict_proba(final_features)
        prediction_1 = model.predict(final_features)
       
        prediction_list = prediction[0].tolist()

        
        for item in prediction_list:
            if not isinstance(item, (int, float)):
                raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

        
        return jsonify({"prediction_probability": prediction_list},{"prediction": int(prediction_1[0])})
               
    

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7000,debug=True)
