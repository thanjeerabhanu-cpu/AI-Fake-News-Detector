from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    vect = vectorizer.transform([news])

    prediction = model.predict(vect)[0]
    probability = model.predict_proba(vect)[0]

    confidence = round(max(probability) * 100, 2)

    if prediction == 1:
        result = "REAL NEWS"
        color = "green"
    else:
        result = "FAKE NEWS"
        color = "red"

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        color=color
    )

if __name__ == "__main__":
    app.run(debug=True)