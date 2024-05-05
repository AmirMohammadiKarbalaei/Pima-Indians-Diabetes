import streamlit as st
import pandas as pd
from app_utils import *

# Create the Streamlit web app
def main():
    st.title('Diabetes Prediction')
    st.write("This app predicts diabetes using  input data. Ensure that the input data includes the following features: ")

    # File uploader for CSV file
    st.image("/Deployment/Data-Table.PNG")
    uploaded_file = st.file_uploader("Upload your data as a CSV file ", type=["csv"])

    if uploaded_file is not None:
        try:
        # Read the uploaded CSV file
            input_data = pd.read_csv(uploaded_file)
        
        # Show the uploaded data
            st.write('Uploaded CSV file:')
            st.write(input_data.head())

        

        # Make prediction when user clicks the 'Predict' button
            if st.button('Make Prediction'):
                processed_data = preprocess(input_data)
                prediction = predict(processed_data.values)
                st.write('Prediction:')
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

    if st.button('Quit'):
        st.stop()

if __name__ == '__main__':
    main()