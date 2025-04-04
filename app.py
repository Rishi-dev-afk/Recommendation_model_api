from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('course_recommender.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({'predicted_rating': float(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
