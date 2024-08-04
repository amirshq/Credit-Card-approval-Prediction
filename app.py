import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt 
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score, precision_score, f1_score
import xgboost as xgb   
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


application_record_df = pd.read_csv('/Users/amirshahcheraghian/Desktop/Desktop/PHD ETS/Python Codes/Kaggle /Credit Card Approval Prediction/application_record.csv')
credit_card_record_df = pd.read_csv('/Users/amirshahcheraghian/Desktop/Desktop/PHD ETS/Python Codes/Kaggle /Credit Card Approval Prediction/credit_record.csv')

# Merged two DataFrames
merged_df = pd.merge(application_record_df,credit_card_record_df,how='outer',on='ID')

# Fill NaN values with the previous value
merged_df.fillna(method='ffill',inplace=True)

# Encoding 
categorical_columns = ['FLAG_OWN_CAR','FLAG_OWN_REALTY','OCCUPATION_TYPE','CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']

# Initialize LabelEncoder
label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()

# Apply label encoding to the categorical columns
for col in categorical_columns:
    merged_df[col] = label_encoders[col].fit_transform(merged_df[col])

merged_df['TARGET'] = merged_df['STATUS'].apply(lambda x:1 if x in ['1','2','3','4','5'] else 0)

target = merged_df.groupby('ID')['TARGET'].max().reset_index()

# aggregated_df is the merged_df after GroupBy
aggregated_df = merged_df.groupby('ID').agg({
    'CODE_GENDER': 'first',
    'FLAG_OWN_CAR': 'first',
    'FLAG_OWN_REALTY': 'first',
    'CNT_CHILDREN': 'first',
    'AMT_INCOME_TOTAL': 'mean',
    'NAME_INCOME_TYPE': 'first',
    'NAME_EDUCATION_TYPE': 'first',
    'NAME_FAMILY_STATUS': 'first',
    'NAME_HOUSING_TYPE': 'first',
    'DAYS_BIRTH': 'mean',
    'DAYS_EMPLOYED': 'mean',
    'FLAG_MOBIL': 'first',
    'FLAG_WORK_PHONE': 'first',
    'FLAG_PHONE': 'first',
    'FLAG_EMAIL': 'first',
    'OCCUPATION_TYPE': 'first',
    'CNT_FAM_MEMBERS': 'first',
    'MONTHS_BALANCE': 'max',  # assuming we take the most recent month
    'STATUS': 'last'  # assuming we take the latest status
}).reset_index()



# aggregated_df is the merged_df after GroupBy
merged_df_aggregate = merged_df.groupby('ID').agg({
    'CODE_GENDER': 'first',
    'FLAG_OWN_CAR': 'first',
    'FLAG_OWN_REALTY': 'first',
    'CNT_CHILDREN': 'first',
    'AMT_INCOME_TOTAL': 'mean',
    'NAME_INCOME_TYPE': 'first',
    'NAME_EDUCATION_TYPE': 'first',
    'NAME_FAMILY_STATUS': 'first',
    'NAME_HOUSING_TYPE': 'first',
    'DAYS_BIRTH': 'mean',
    'DAYS_EMPLOYED': 'mean',
    'FLAG_MOBIL': 'first',
    'FLAG_WORK_PHONE': 'first',
    'FLAG_PHONE': 'first',
    'FLAG_EMAIL': 'first',
    'OCCUPATION_TYPE': 'first',
    'CNT_FAM_MEMBERS': 'first',
    'MONTHS_BALANCE': 'max',  # assuming we take the most recent month
    'STATUS': 'last',  # assuming we take the latest status
    'TARGET':'max'
}).reset_index()






