import streamlit as st
import pandas as pd
from app_utils import *

# Define app title and introduction
def main():
    # Setting page configuration
    st.set_page_config(
        page_title='Diabetes Prediction App',
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded"  # Expand the sidebar by default
    )

    # Setting background color and padding
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f2f6;
        }
        .st-bd {
            padding: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Adding a title with colorful background
    st.markdown(
        """
        <div style="background-color:#96401F;padding:10px;border-radius:10px">
        <h1 style="color:white;text-align:center;">Diabetes Classifier</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Introduction text with colorful background
    st.markdown(
        """
        <div style="background-color:#967243;padding:10px;border-radius:10px;margin-top:20px">
        <p style="color:white;text-align:center;">This app predicts diabetes using input data. Ensure that the input data includes the following features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, and Age.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # File uploader for CSV file
    uploaded_file = st.file_uploader("Upload your data as a CSV file ", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read the uploaded CSV file
            input_data = pd.read_csv(uploaded_file)

            # Show the uploaded data
            st.subheader('Uploaded CSV file:')
            st.write(input_data.head())

            # Make prediction when user clicks the 'Predict' button
            if st.button('Make Prediction'):
                with st.spinner('Making prediction...'):
                    processed_data = preprocess(input_data)
                    prediction = predict(processed_data.values)
                st.subheader('Prediction:')
                st.write(prediction)

                # Provide an option to save the prediction as a CSV file
                st.markdown('### Save Prediction')
                st.write('Click below to save the prediction as a CSV file.')
                st.download_button(label="Save Prediction as CSV",
                                   data=pd.DataFrame(prediction, columns=["Prediction"]).to_csv(index=False),
                                   file_name='prediction.csv',
                                   mime='text/csv')
        except Exception as e:
            st.error(f"An error occurred: {e}")

        
                

    # Quit button with colorful background
    if st.button('Quit'):
        st.stop()

if __name__ == '__main__':
    main()
