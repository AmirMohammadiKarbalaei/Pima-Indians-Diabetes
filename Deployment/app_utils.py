from joblib import load
import pandas as pd
from scipy.stats import zscore

def preprocess(dataset):

    df = pd.DataFrame(dataset)
    data = df.copy()

    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True) 

    data['HOMA_IR'] = (df['Glucose'] * df['Insulin']) / 405  # Homeostatic Model Assessment for Insulin Resistance

    # Interaction Feature Creation 
    data['Glucose_BMI_Interact'] = df['Glucose'] * df['BMI']
    data['Age_Insulin_Interact'] = df['Age'] * df['Insulin']
    data['Age_BMI_Interact'] = df['Age'] * df['BMI']
    data['Age_Pregnancies_Interact'] = df['Age'] * df['Pregnancies']
    data['Insulin_skin_Interact'] = df['Insulin'] * df['SkinThickness']
    data['SkinThickness_to_BMI_Ratio'] = df['SkinThickness'] * df['BMI']

  
    z_scores = data.apply(zscore)

    print("Data is ready!!")

    return z_scores

def predict(data):
    loaded_rf_model = load('Deployment/random_forest_model.joblib')
    # Make predictions using the loaded model
    predictions = loaded_rf_model.predict(data)
    return predictions