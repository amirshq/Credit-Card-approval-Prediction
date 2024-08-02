import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

application_record_df = pd.read_csv('/Users/amirshahcheraghian/Desktop/Desktop/PHD ETS/Python Codes/Kaggle /Credit Card Approval Prediction/application_record.csv')
credit_card_record_df = pd.read_csv('/Users/amirshahcheraghian/Desktop/Desktop/PHD ETS/Python Codes/Kaggle /Credit Card Approval Prediction/credit_record.csv')
