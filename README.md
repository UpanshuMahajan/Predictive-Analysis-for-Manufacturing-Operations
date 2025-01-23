# Predictive-Analysis-for-Manufacturing-Operations
Predictive analysis API for manufacturing operations using machine learning. The API predicts machine downtime or defects based on input data like temperature and run time. Built with Flask, it supports file uploads for training, model evaluation, and real-time predictions with confidence scores.
# Overview
This project implements a Predictive Analysis Model to predict machine downtime or production defects in a manufacturing environment. It uses a simple machine learning model (Decision Tree) and exposes a RESTful API to interact with the model for training and making predictions.

# Features:
Upload Dataset: Upload a CSV file containing manufacturing data.
Train Model: Train a machine learning model using the uploaded dataset.
Predict Downtime: Make predictions based on new data inputs.

## Tech Stack

Backend Framework: Flask

Machine Learning Library: scikit-learn

Programming Language: Python

Endpoints

1. / (GET)

Description: Check if the API is running.

Response: "Manufacturing Operations Predictive API is Running!"

2. /upload (POST)

Description: Upload a CSV file containing manufacturing data.

Input: CSV file with columns: Machine_ID, Temperature, Run_Time, Hydraulic_Pressure, Coolant_Temperature, Spindle_Speed, Defect_Rate, Downtime_Flag.
Output: A success or error message.

3. /train (POST)

Trains a Decision Tree model on the uploaded dataset.

Input: None (uses the uploaded dataset).
Output: Model performance metrics (accuracy, F1 score).

4. /predict (POST)

Makes predictions based on input parameters.

Input: JSON with features such as Temperature, Run_Time, etc.
Output: Prediction (downtime: yes/no) and confidence score.

## Technical Requirements

Programming Language: Python

Framework: Flask

Machine Learning Library: Scikit-learn

Testing Tools: Postman or cURL

etup Instructions

1. Clone the Repository

git clone https://github.com/yourusername/predictive-analysis-api.git
cd predictive-analysis-api

2. Install Dependencies

Ensure Python 3.x and pip are installed.
pip install -r requirements.txt

3. Run the API

Start the Flask server.
python app.py

Access the API at http://127.0.0.1:5000/.

### Dataset

The API expects a CSV dataset with the following columns:

Machine_ID

Temperature

Run_Time

Hydraulic_Pressure

Coolant_Temperature

Spindle_Speed

Defect_Rate

Downtime_Flag (Target variable: 0 or 1)

Example Use Cases

Predicting machine downtime to schedule maintenance.
Identifying defect rates and optimizing manufacturing processes.
