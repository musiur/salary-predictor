from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import json
import pandas as pd

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load mappings
with open('mappings.json', 'r') as f:
    mappings = json.load(f)

# Extract mappings
gender_mapping = mappings['gender']
education_mapping = mappings['education']
designation_mapping = mappings['designation']

# Initialize FastAPI
app = FastAPI()

# Define input schema
class SalaryPredictionRequest(BaseModel):
    age: int
    gender: str
    education: str
    designation: str
    experience: int

@app.post("/predict")
def predict_salary(request: SalaryPredictionRequest):
    try:
        # Map categorical inputs
        gender = gender_mapping.get(request.gender)
        education = education_mapping.get(request.education)
        designation = designation_mapping.get(request.designation)

        # Validate mappings
        if None in [gender, education, designation]:
            raise HTTPException(status_code=400, detail="Invalid categorical input")

        # Prepare input for the model as a pandas DataFrame
        input_data = pd.DataFrame([{
            "age": request.age,
            "gender": gender,
            "education": education,
            "designation": designation,
            "experience": request.experience
        }])

        # Predict salary
        predicted_salary = model.predict(input_data)

        # Convert numpy.float32 to native Python float
        salary_as_float = float(predicted_salary[0])

        return {"predicted_salary": round(salary_as_float, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
