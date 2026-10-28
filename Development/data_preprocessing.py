import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,LabelEncoder,StandardScaler,TargetEncoder,MinMaxScaler



def create_preprocessing_pipeline():

    StandardScaler_columns = [
        'DailyRate',
        'DistanceFromHome',
        'HourlyRate',
        'MonthlyRate',
        'NumCompaniesWorked',
        'PercentSalaryHike',
        'YearsAtCompany',
        'YearsInCurrentRole',
        'YearsSinceLastPromotion',
        'YearsWithCurrManager'
    ]

    MinMaxScaler_columns = [
        'Age',
        'MonthlyIncome'
    ]

    onehot_columns = [
        'BusinessTravel',
        'Department',
        'EducationField',
        'Gender',
        'JobRole',
        'MaritalStatus',
        'OverTime'
    ]

    ordinal_columns = [
        'Education',
        'EnvironmentSatisfaction',
        'JobInvolvement',
        'JobLevel',
        'JobSatisfaction',
        'PerformanceRating',
        'WorkLifeBalance'
    ]

    StandardScaler_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    minmax_scaler_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("minmaxscaler", MinMaxScaler())
    ])

    onehot_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder())
    ])

    preprocessing_pipeline = ColumnTransformer([
        ("standard", StandardScaler_pipeline, StandardScaler_columns),
        ("MinMax", minmax_scaler_pipeline, MinMaxScaler_columns),
        ("onehot", onehot_pipeline, onehot_columns),
        ("ordinal", ordinal_pipeline, ordinal_columns)
    ])

    return preprocessing_pipeline