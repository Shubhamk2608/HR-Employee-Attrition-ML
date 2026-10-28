# import pandas as pd

# def create_features(df):

#     df["CheckUp_Date"]=pd.to_datetime(df["CheckUp_Date"])

#     df["checkup_year"]=df["CheckUp_Date"].dt.year

#     df["checkup_month"]=df["CheckUp_Date"].dt.month

#     df["checkup_day"]=df["CheckUp_Date"].dt.day

#     df["checkup_dayofweek"]=df["CheckUp_Date"].dt.dayofweek


#     df=df.drop("CheckUp_Date",axis=1)


#     # return df