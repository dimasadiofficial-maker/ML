from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model dan scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["Age"])
    sex = int(request.form["Sex"])
    chestpain = int(request.form["ChestPainType"])
    restingbp = float(request.form["RestingBP"])
    cholesterol = float(request.form["Cholesterol"])
    fastingbs = int(request.form["FastingBS"])
    restingecg = int(request.form["RestingECG"])
    maxhr = float(request.form["MaxHR"])
    exerciseangina = int(request.form["ExerciseAngina"])
    oldpeak = float(request.form["Oldpeak"])
    st_slope = int(request.form["ST_Slope"])

    data = pd.DataFrame([[
        age,
        sex,
        chestpain,
        restingbp,
        cholesterol,
        fastingbs,
        restingecg,
        maxhr,
        exerciseangina,
        oldpeak,
        st_slope
    ]], columns=[
        'Age',
        'Sex',
        'ChestPainType',
        'RestingBP',
        'Cholesterol',
        'FastingBS',
        'RestingECG',
        'MaxHR',
        'ExerciseAngina',
        'Oldpeak',
        'ST_Slope'
    ])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]

    if prediction == 1:
        hasil = "⚠️ Berisiko Penyakit Jantung"
    else:
        hasil = "✅ Tidak Berisiko Penyakit Jantung"

    return render_template(
        "index.html",
        hasil=hasil
    )

if __name__ == "__main__":
    app.run(debug=True)