from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and encoders
model = joblib.load(r'Models/model.pkl')
region_ohe = joblib.load(r'scalers/ohe_region.pkl')
sex_ohe = joblib.load(r'scalers/ohe_sex.pkl')
smoker_ohe = joblib.load(r'scalers/ohe_smoker.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    sex = request.form['sex']
    smoker = request.form['smoker']
    region = request.form['region']

    # Encode categorical features
    sex_enc = sex_ohe.transform([[sex]])
    smoker_enc = smoker_ohe.transform([[smoker]])
    region_enc = region_ohe.transform([[region]])

    # Combine all features
    final_input = [[age, bmi, children] + list(sex_enc[0]) + list(smoker_enc[0]) + list(region_enc[0])]

    prediction = model.predict(final_input)[0]
    prediction = round(prediction, 2)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)