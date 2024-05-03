from joblib import  load
import pandas as pd


model = load("Deployment/random_forest_model.joblib")

data = pd.read_csv('Deployment/Cleaned_data.csv')
predictions = model.predict(data.values)
print(predictions)