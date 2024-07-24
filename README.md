# Pima Indians Diabetes Classification

This project focuses on developing a predictive model to diagnose diabetes in Pima Indians using machine learning techniques. The dataset includes various medical predictor variables and a target variable indicating diabetes presence.

## Dataset

- **Features**: Includes medical metrics such as the number of pregnancies, BMI, insulin levels, age, and more.
- **Target**: Diabetes outcome (`0` = No, `1` = Yes).

## Process

1. **Exploratory Data Analysis (EDA)**
   - Conducted comprehensive data cleansing.
   - Performed detailed analysis of each feature to ensure data quality and integrity.

2. **Feature Engineering**
   - Created new features and assessed their impact using SHAP (Shapley Additive Explanations).

3. **Synthetic Minority Oversampling Technique (SMOTE)**
   - Applied SMOTE to address class imbalance by generating synthetic samples for the minority class, enhancing model performance.

4. **Model Implementation**
   - **Random Forest**: Achieved 93% accuracy. Feature importance was analyzed using SHAP, revealing that low insulin levels and the interaction between age and insulin are highly influential.
   - **Deep Learning Model**: Evaluated for performance; results were comparable to the Random Forest model.

5. **Deployment**
   - **Model Selection**: Random Forest was chosen for deployment due to its interpretability and efficiency.
   - **Streamlit Application**: Developed for easy prediction, with integrated preprocessing and feature engineering.

## Key Insights

- **Influential Features**: Low insulin values and the interaction between age and insulin significantly affect predictions.
- **Less Impactful Features**: Features like blood pressure and number of pregnancies have minimal effect and can be omitted to reduce computational costs.

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

1. **Run the Streamlit Application**
   ```bash
   streamlit run app.py
   ```

2. **Input Data**
   - Prepare a pandas DataFrame with the same structure as the training dataset.
   - The Streamlit app will handle preprocessing and feature engineering, then provide predictions.

## Contributing

Contributions are welcome! If you have suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
