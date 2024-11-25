import pandas as pd
import numpy as np
import pickle
import json
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset from CSV
data = pd.read_csv("dataset.csv")

# Data Cleaning Process

# 1. Remove duplicates
data = data.drop_duplicates()
print(f"Duplicates removed. Data shape: {data.shape}")

# 2. Handle missing values
# Fill missing categorical values with the mode and numeric values with the median
for column in data.columns:
    if data[column].isnull().sum() > 0:
        if data[column].dtype == 'object':  # Categorical column
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:  # Numeric column
            data[column].fillna(data[column].median(), inplace=True)

print("Missing values handled.")

# 3. Remove outliers
# Define a function to remove outliers using the IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply outlier removal to numeric columns
numeric_columns = ['age', 'experience', 'salary']
for column in numeric_columns:
    original_shape = data.shape
    data = remove_outliers(data, column)
    print(f"Outliers removed from {column}. Rows before: {original_shape[0]}, after: {data.shape[0]}")

# 4. Map categorical columns
gender_mapping = {"Male": 0, "Female": 1}
education_mapping = {"Bachelor's": 0, "Master's": 1, "PhD": 2}

# Dynamically create designation mapping
unique_designations = data['designation'].unique()
designation_mapping = {designation: idx for idx, designation in enumerate(unique_designations)}

# Apply mappings
data['gender'] = data['gender'].map(gender_mapping)
data['education'] = data['education'].map(education_mapping)
data['designation'] = data['designation'].map(designation_mapping)

# Export mappings to a JSON file
mappings = {
    "gender": gender_mapping,
    "education": education_mapping,
    "designation": designation_mapping
}
with open('mappings.json', 'w') as f:
    json.dump(mappings, f)

print("Mappings exported to mappings.json")

# Split the data into features (X) and target (y)
X = data.drop('salary', axis=1)
y = data['salary']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        # Standardize numeric features
        ('num', StandardScaler(), ['age', 'experience']),
        # No additional transformation needed for categorical columns
    ],
    remainder='passthrough'  # Pass through categorical columns
)

# Build a pipeline with preprocessing and the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42))
])

# Train the model
pipeline.fit(x_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('R-squared:', r2)
print("The R-squared value of our model is {:.2f}%".format(r2 * 100))

# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model saved as model.pkl")
