from flask import Flask, render_template,request
import joblib

app = Flask(__name__)

model = joblib.load("hdi_model.pkl")

@app.route("/")
def home():
    return render_template("index.html",prediction=None)

@app.route("/predict", methods=["POST"])
def predict():

    life = float(request.form["life"])
    expected = float(request.form["expected"])
    mean = float(request.form["mean"])
    income = float(request.form["income"])

    prediction = model.predict([[life, expected, mean, income]])

    prediction = round(prediction[0],3)
    if prediction < 0.55:
     category = "Low Human Development"
    elif prediction < 0.70:
     category = "Medium Human Development"
    elif prediction < 0.80:
     category = "High Human Development"
    else:
     category = "Very High Human Development"

    return render_template("index.html", prediction=prediction,category=category)

if __name__ == "__main__":
    app.run(debug=True)