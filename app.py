from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import json
import pandas as pd
import os
# import modal

# Load the trained model locally
if not os.path.exists("model.pkl"):
    raise Exception("Model file 'model.pkl' not found. Please ensure it's in the project directory.")
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load mappings
if not os.path.exists("mappings.json"):
    raise Exception("Mappings file 'mappings.json' not found. Please ensure it's in the project directory.")
with open('mappings.json', 'r') as f:
    mappings = json.load(f)

# Extract mappings
gender_mapping = mappings['gender']
education_mapping = mappings['education']
designation_mapping = mappings['designation']


#initialize modal app
# app = modal.App("salary-predictor")
# Initialize FastAPI
app = FastAPI()


# image = modal.Image.debian_slim().pip_install("fastapi", "pydantic", "uvicorn", "joblib", "pandas", "numpy", "scikit-learn", "seaborn", "matplotlib", "xgboost", "wordcloud", "flask", "modal")
# dockerfile_image = modal.Image.from_dockerfile("Dockerfile")

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
        # Validate and map categorical inputs
        gender = gender_mapping.get(request.gender)
        education = education_mapping.get(request.education)
        designation = designation_mapping.get(request.designation)

        if gender is None:
            raise HTTPException(status_code=400, detail=f"Invalid gender input: {request.gender}")
        if education is None:
            raise HTTPException(status_code=400, detail=f"Invalid education input: {request.education}")
        if designation is None:
            raise HTTPException(status_code=400, detail=f"Invalid designation input: {request.designation}")

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

        # Optional: Log predictions (e.g., to a file or console)
        # print(f"Prediction made: {request.dict()}, Predicted Salary: {salary_as_float}")

        return {"predicted_salary": round(salary_as_float, 2)}

    except HTTPException as http_ex:
        # Reraise HTTP exceptions for predictable errors
        raise http_ex
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


# @app.function(image=dockerfile_image)
# @modal.asgi_app()
# def fastapi_app():
#     return web_app