import pandas as pd
import numpy as np
from utils import *



dataset = np.loadtxt('data/pima-indians-diabetes.csv', delimiter=',')

cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
df = pd.DataFrame(dataset,columns=cols)


df.loc[df['BloodPressure'] < 50, 'BloodPressure'] = np.NAN
df.loc[(df.Insulin == 0) & (df.Outcome == 0 ),"Insulin"] = np.NAN  # only replacing 0 insulin values for non diabetics

data = df[(df.Glucose !=0) & (df.BMI !=0)] # removing data points where Glucose and BMI are non zero

data['HOMA_IR'] = (df['Glucose'] * df['Insulin']) / 405  # Homeostatic Model Assessment for Insulin Resistance

# Interaction Feature Creation 
data['Glucose_BMI_Interact'] = df['Glucose'] * df['BMI']
data['Age_Insulin_Interact'] = df['Age'] * df['Insulin']
data['Age_BMI_Interact'] = df['Age'] * df['BMI']
data['Age_Pregnancies_Interact'] = df['Age'] * df['Pregnancies']
data['Insulin_skin_Interact'] = df['Insulin'] * df['SkinThickness']
data['SkinThickness_to_BMI_Ratio'] = df['SkinThickness'] * df['BMI']

# Categorising 
data['Family_History'] = df['DiabetesPedigreeFunction'].apply(lambda x: 1 if x > 0.5 else 0)
data['BloodPressure_Category'] = df['BloodPressure'].apply(encode_blood_pressure) # categorising Blood Pressures
data['BloodPressure_Category'] = df["Glucose"].apply(categorise_glucose) # categorising Glucose

bins = [0, 20, 40, 60, 85]
labels = [0, 1, 2,3]
data['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False) # Creating Age groups from bins
data[data.columns[:18]].dropna().to_csv("Deployment/Cleaned_data.csv",index=False)
print("Data is ready!!")
print(data[data.columns[:18]].shape)



