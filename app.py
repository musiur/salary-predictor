from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Initialize the FastAPI app
app = FastAPI()

# Load the pre-trained model
model = joblib.load("model.pkl")  # Ensure the correct path to your model

# Define the expected schema
class PredictionInput(BaseModel):
    age: float
    gender: int  # Assuming 1 for Male, 0 for Female
    education: str
    designation: str
    experience: float

# Initialize encoders for categorical variables
le_education = LabelEncoder()
le_designation = LabelEncoder()

# Fit encoders with all possible values (ensure this matches your training data)
le_education.classes_ = ['high school', 'bachelor', 'master', 'phd']  # Example classes
le_designation.classes_ = ['junior', 'mid-level', 'senior']  # Example classes

@app.get("/")
async def root():
    """
    Welcome endpoint for the Salary Prediction API.
    """
    return {"message": "Welcome to the Salary Prediction API"}

@app.post("/predict")
async def predict(data: PredictionInput):
    """
    Endpoint to predict salary based on input data.
    """
    # Convert input data to a DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Validate and encode categorical features
    try:
        input_data['education'] = le_education.transform([input_data['education'][0]])
        input_data['designation'] = le_designation.transform([input_data['designation'][0]])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid categorical value: {str(e)}")

    # Make prediction
    try:
        prediction = model.predict(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return {"predicted_salary": prediction[0]}
