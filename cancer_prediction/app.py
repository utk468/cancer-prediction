from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)

# Load your trained model (make sure the path is correct)
model = joblib.load('random_forest_model.pkl')

# Sample dataframe (you can replace this with your actual data loading and preprocessing logic)
data = pd.read_csv('cancer.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get values from the form
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        air_pollution = int(request.form['air_pollution'])
        alcohol_use = int(request.form['alcohol_use'])
        dust_allergy = int(request.form['dust_allergy'])
        occupational_hazards = int(request.form['occupational_hazards'])
        genetic_risk = int(request.form['genetic_risk'])
        chronic_lung_disease = int(request.form['chronic_lung_disease'])
        balanced_diet = int(request.form['balanced_diet'])
        obesity = int(request.form['obesity'])
        smoking = int(request.form['smoking'])
        passive_smoker = int(request.form['passive_smoker'])
        chest_pain = int(request.form['chest_pain'])
        coughing_of_blood = int(request.form['coughing_of_blood'])
        fatigue = int(request.form['fatigue'])
        weight_loss = int(request.form['weight_loss'])
        shortness_of_breath = int(request.form['shortness_of_breath'])
        wheezing = int(request.form['wheezing'])
        swallowing_difficulty = int(request.form['swallowing_difficulty'])
        clubbing_of_finger_nails = int(request.form['clubbing_of_finger_nails'])
        frequent_cold = int(request.form['frequent_cold'])
        dry_cough = int(request.form['dry_cough'])
        snoring = int(request.form['snoring'])

        # Create a feature array for prediction
        features = [[age, gender, air_pollution, alcohol_use, dust_allergy, occupational_hazards, genetic_risk, 
                     chronic_lung_disease, balanced_diet, obesity, smoking, passive_smoker, chest_pain, 
                     coughing_of_blood, fatigue, weight_loss, shortness_of_breath, wheezing, swallowing_difficulty, 
                     clubbing_of_finger_nails, frequent_cold, dry_cough, snoring]]
        
        # Predict the level
        prediction = model.predict(features)
        return render_template('index.html', prediction_text=f'Predicted Level: {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)
