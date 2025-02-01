from flask import Flask, request, jsonify
from autogluon.tabular import TabularPredictor
import pandas as pd
import os

# ✅ Set the full absolute path to the saved model
model_path = r"C:\Users\allif\Downloads\Credit+Card+Fraud+Using+Pycaret+-Code+and+Files\Code and Files\AutogluonModels\ag-20250201_220122"

# ✅ Print debug message to verify the correct path
print("Loading model from:", model_path)

# ✅ Load the AutoGluon model
try:
    predictor = TabularPredictor.load(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)  # Stop execution if model fails to load

# ✅ Initialize Flask app
app = Flask(__name__)

# ✅ Home Route
@app.route('/')
def home():
    return "🔥 Fraud Detection API is Running!"

# ✅ Prediction Route (POST Request)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 🟢 Get JSON data from request
        data = request.get_json()
        
        # 🟢 Convert JSON into DataFrame (AutoGluon requires a DataFrame)
        df = pd.DataFrame([data])

        # 🟢 Make predictions
        prediction = predictor.predict(df)

        # 🟢 Return prediction result as JSON
        return jsonify({'prediction': int(prediction.iloc[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

# ✅ Run Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
