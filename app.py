from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("car_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    car_age = int(request.form['car_age'])
    year = int(request.form['year'])
    present_price = float(request.form['present_price'])
    kms_driven = float(request.form['kms_driven'])

    # Build feature vector (4 features)
    features = np.array([[car_age, year, present_price, kms_driven]])

    prediction = model.predict(features)[0]

    return render_template('index.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)