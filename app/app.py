
import streamlit as st
import pandas as pd
import joblib
import sys
import os
import mlflow
import mlflow.sklearn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(BASE_DIR)


#from Development.feature_engg import create_features
from Development.feature_selection import select_features
from Development.model_monitoring import save_prediction

# #model = joblib.load(
#     os.path.join(BASE_DIR, "model.pkl")
# )

# preprocessing_pipeline = joblib.load(
#     os.path.join(BASE_DIR, "preprocessing_pipeline.pkl")
# )

mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_URI = "models:/HR_Employee_Atrrition1@production"

model = mlflow.sklearn.load_model(MODEL_URI)


# Streamlit Page Configuration
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="�"
)


st.title("HR_Employee_Attrition")

st.write("Enter employee information to get a prediction.")


# Employee Information
st.subheader("Employee Information")


employee_age = st.number_input(
    "Age",
    min_value=1,
    max_value=50,
    value=30
)


BusinessTravel=st.selectbox(
    'BusinessTravel',
    ["Travel_Rarely","Travel_Frequently","Non-Travel"]
)

DailyRate=st.number_input(
    'DailyRate',
    min_value=100,
    max_value=10000,
    value=100

)


HourlyRate=st.number_input(
    'HourlyRate',
    min_value=1,
    max_value=100,
    value=100

)



MonthlyIncome=st.number_input(
    'MonthlyIncome',
    min_value=1,
    max_value=80000,
    value=100

)




MonthlyRate=st.number_input(
    'MonthlyRate',
    min_value=1,
    max_value=50000,
    value=100

)


Department=st.selectbox(
    'Department',
    ["Sales","Research & Development","Human Resources"]
)



DistanceFromHome=st.number_input(
    'DistanceFromHome',
    min_value=1,
    max_value=50,
    value=50

)


Education=st.number_input(
    'Education',
    min_value=1,
    max_value=5,
    value=5

)




EducationField=st.selectbox(
    'EducationField',
    ["Life Sciences","Other","Medical","Marketing","Technical Degree","Human Resources"]
)


EnvironmentSatisfaction=st.number_input(
    'EnvironmentSatisfaction',
    min_value=1,
    max_value=5,
    value=5

)


Gender=st.selectbox(
    'Gender',
    ["Male","Other","Female"]
)


JobInvolvement=st.number_input(
    'JobInvolvement',
    min_value=1,
    max_value=5,
    value=5

)

JobLevel=st.number_input(
    'JobLevel',
    min_value=1,
    max_value=5,
    value=5

)

 
JobRole=st.selectbox(
    'JobRole',
    ["Sales Executive",
     "Research Scientist",
     "Laboratory Technician",
     "Manufacturing Director",
     "Healthcare Representative",
     "Manager",
     "Sales Representative",
     "Research Director",
     "Human Resources"
]
)
 
 
 
JobSatisfaction=st.number_input(
    'JobSatisfaction',
    min_value=1,
    max_value=5,
    value=5

) 
 
 
MaritalStatus=st.selectbox(
    'MaritalStatus',
    ["Single","Married","Divorced"]
)

 
NumCompaniesWorked=st.number_input(
    'NumCompaniesWorked',
    min_value=1,
    max_value=10,
    value=10

) 


OverTime=st.selectbox(
    'OverTime',
    ["No","Yes"]
)
 

 
PercentSalaryHike=st.number_input(
    'PercentSalaryHike',
    min_value=1,
    max_value=50,
    value=50

)
 

PerformanceRating=st.number_input(
    'PerformanceRating',
    min_value=1,
    max_value=5,
    value=5

)


WorkLifeBalance=st.number_input(
    'WorkLifeBalance',
    min_value=1,
    max_value=5,
    value=5

)


YearsAtCompany=st.number_input(
    'YearsAtCompany',
    min_value=1,
    max_value=50,
    value=50

)



YearsInCurrentRole=st.number_input(
    'YearsInCurrentRole',
    min_value=1,
    max_value=50,
    value=50

)



YearsSinceLastPromotion=st.number_input(
    'YearsSinceLastPromotion',
    min_value=1,
    max_value=50,
    value=50

)



YearsWithCurrManager=st.number_input(
    'YearsWithCurrManager',
    min_value=1,
    max_value=50,
    value=50

)
 

# Prediction
# Prediction
if st.button("Predict"):

    try:

        # Create input DataFrame
        input_data = pd.DataFrame({
            "MonthlyIncome": [MonthlyIncome],
            "Age":[employee_age],
            
            "BusinessTravel":[BusinessTravel],
            "DailyRate":[DailyRate],
            "Department":[Department],
            "DistanceFromHome":[DistanceFromHome],
            "Education":[Education],
            "EducationField":[EducationField],
            
            
            "EnvironmentSatisfaction":[EnvironmentSatisfaction],
            "Gender":[Gender],
            "HourlyRate":[HourlyRate],
            "JobInvolvement":[JobInvolvement],
            "JobLevel":[JobLevel],
            "JobRole":[JobRole],
            "JobSatisfaction":[JobSatisfaction],
            "MaritalStatus":[MaritalStatus],
            
            "MonthlyRate":[MonthlyRate],
            "NumCompaniesWorked":[NumCompaniesWorked],
            
            "OverTime":[OverTime],
            "PercentSalaryHike":[PercentSalaryHike],
            "PerformanceRating":[PerformanceRating],
            
            "WorkLifeBalance":[WorkLifeBalance],
            "YearsAtCompany":[YearsAtCompany],
            "YearsInCurrentRole":[YearsInCurrentRole],
            "YearsSinceLastPromotion":[YearsSinceLastPromotion],
            "YearsWithCurrManager":[YearsWithCurrManager]
        })

        # Feature Engineering
       ## input_data = create_features(input_data)

        # Feature Selection
        #input_data = select_features(input_data)

        # Preprocessing
        #input_processed = preprocessing_pipeline.transform(
        #    input_data
        #)

        # Prediction
        prediction = model.predict(input_data)

        probability = model.predict_proba(input_data)

        employee_probability = probability[0][1] * 100

        # Display Prediction
        if prediction[0] == 1:

            st.error("⚠️ The model predicts that the employee is likely to leave the company.")

        else:

            st.success("✅ The model predicts that the employee is not likely to leave the company.")

        st.write(
            "Probability:",
            round(employee_probability, 2),
            "%"
        )

        # Save Prediction Log
        log_file = os.path.join(
            BASE_DIR,
            "prediction_logs.csv"
        )

        save_prediction(
            input_data,
            prediction[0],
            employee_probability,
            "v1",
            log_file
        )

        st.success("Prediction logged successfully.")

    except Exception as e:

        st.error(
            "⚠️ Prediction could not be completed. "
            "Please try again."
        )