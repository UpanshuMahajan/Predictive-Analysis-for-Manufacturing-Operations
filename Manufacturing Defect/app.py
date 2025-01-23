from flask import Flask, jsonify, request
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

app = Flask(__name__)

# Initialize global variables
model = None
FEATURES = ['Temperature', 'Run_Time', 'Hydraulic_Pressure', 'Coolant_Temperature', 'Spindle_Speed', 'Defect_Rate']  # Features for training
TARGET = 'Downtime_Flag'  # The target column for predictions

# In-memory storage for the uploaded dataset
uploaded_data = None

@app.route('/')
def home():
    return "Manufacturing Operations Predictive API is Running!"

# Endpoint for uploading dataset
@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Check if the file is part of the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        
        # Check if a file was selected
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Read the file into a pandas DataFrame (if CSV)
        data = pd.read_csv(file)
        
        # If you want, you can validate your data here (e.g., checking required columns)
        global uploaded_data
        uploaded_data = data
        
        return jsonify({'message': 'Dataset uploaded successfully!'})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Endpoint for training the model
@app.route('/train', methods=['POST'])
def train():
    global model, uploaded_data
    try:
        # Ensure the dataset is uploaded
        if uploaded_data is None:
            return jsonify({"error": "No dataset uploaded yet."}), 400

        # Extract features and target variables (using all available features for training)
        X = uploaded_data[FEATURES]
        y = uploaded_data[TARGET]

        # Train the model
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X, y)

        # Make predictions to evaluate the model
        y_pred = model.predict(X)

        # Calculate performance metrics
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        # Save the trained model for later use
        joblib.dump(model, 'decision_tree_model.pkl')

        return jsonify({
            "message": "Model trained successfully!",
            "accuracy": accuracy,
            "f1_score": f1
        }), 200
    except Exception as e:
        print(f"Error in training: {e}")
        return jsonify({"error": str(e)}), 400

# Endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    global model
    try:
        # Ensure the model is trained and loaded
        if model is None:
            # Try loading the model from disk if it's not in memory
            try:
                model = joblib.load('decision_tree_model.pkl')
            except Exception as e:
                return jsonify({"error": "Model is not trained yet. Please train the model first."}), 400

        # Get the JSON data from the request
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided!"}), 400

        # Ensure that all features are passed and in the correct order
        input_data = pd.DataFrame([{
            "Machine_ID": data.get("Machine_ID", None),
            "Temperature": data.get("Temperature", None),
            "Run_Time": data.get("Run_Time", None),
            "Hydraulic_Pressure": data.get("Hydraulic_Pressure", None),
            "Coolant_Temperature": data.get("Coolant_Temperature", None),
            "Spindle_Speed": data.get("Spindle_Speed", None),
            "Defect_Rate": data.get("Defect_Rate", None)
        }])

        # Ensure the input data has all the required features in the correct order
        input_data = input_data[FEATURES]  # Select all necessary features for prediction

        # Make a prediction using the trained model
        prediction = model.predict(input_data)

        # Return the prediction as a JSON response
        result = {
            'Downtime': 'Yes' if prediction[0] == 1 else 'No',
            'Confidence': round(model.predict_proba(input_data)[0][1], 2)  # Probability for "Yes" (Downtime)
        }

        return jsonify(result), 200
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
