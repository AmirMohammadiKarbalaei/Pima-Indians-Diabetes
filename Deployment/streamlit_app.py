import streamlit as st
import pandas as pd
from app_utils import preprocess, predict

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
        <p style="color:white;text-align:center;">This app predicts diabetes using input data. Enter the values for the following features or upload a CSV file:</p>
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

    else:
        st.markdown("""
            | Feature | Description |
            |---------|-------------|
            | Pregnancies   | Number of times pregnant       |
            | Glucose   | Plasma glucose concentration a 2 hours in an oral glucose tolerance test       |
            | BloodPressure  | Diastolic blood pressure (mm Hg)      |
            | SkinThickness   | Triceps skin fold thickness (mm)       |
            | Insulin   | 2-Hour serum insulin (mu U/ml)       |
            | BMI   | Body mass index (weight in kg/(height in m)^2)       |
            | DiabetesPedigreeFunction   | Diabetes pedigree function       |
            | Age   | Age (years)      |
            """)
        # Split the input fields into two columns
        col1, col2 = st.columns(2)

        # Input fields in the first column
        with col1:
            pregnancies = st.number_input('Pregnancies', min_value=0, step=1)
            glucose = st.number_input('Glucose', min_value=0)
            blood_pressure = st.number_input('Blood Pressure', min_value=0)
            skin_thickness = st.number_input('Skin Thickness', min_value=0)

        # Input fields in the second column
        with col2:
            insulin = st.number_input('Insulin', min_value=0)
            bmi = st.number_input('BMI', min_value=0.0)
            diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.0, step=0.05)
            age = st.number_input('Age', min_value=0, step=1)

        # Predict button outside of columns
        submitted = st.button("Predict")

        if submitted:
            # Create a DataFrame from the input data
            input_data = pd.DataFrame({
                'Pregnancies': [pregnancies],
                'Glucose': [glucose],
                'BloodPressure': [blood_pressure],
                'SkinThickness': [skin_thickness],
                'Insulin': [insulin],
                'BMI': [bmi],
                'DiabetesPedigreeFunction': [diabetes_pedigree_function],
                'Age': [age]
            })

            with st.spinner('Making prediction...'):
                try:
                    # Assuming preprocess is a function that preprocesses input data
                    processed_data = preprocess(input_data)
                    # Assuming predict is a function that predicts using processed data
                    prediction = predict(processed_data.values)
                    st.subheader('Prediction:')
                    st.write(prediction)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")



    # Quit button with colorful background
    if st.button('Quit'):
        st.stop()

if __name__ == '__main__':
    main()
