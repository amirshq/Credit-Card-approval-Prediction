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


application_record_df = pd.read_csv('/Users/amirshahcheraghian/Credit-Card-approval-Prediction/Data/application_record.csv')
credit_card_record_df = pd.read_csv('/Users/amirshahcheraghian/Credit-Card-approval-Prediction/Data/credit_record.csv')

print(application_record_df.head())
print(credit_card_record_df.head())

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


target_one = pd.DataFrame(merged_df_aggregate[merged_df_aggregate['TARGET']==1])


#Calculate Information Value (IV) Function

def calc_iv(df, feature, target, pr=False):
    lst = []
    df[feature] = df[feature].fillna("NULL")

    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature,                                                        # Variable
                    val,                                                            # Value
                    df[df[feature] == val].count()[feature],                        # All
                    df[(df[feature] == val) & (df[target] == 0)].count()[feature],  # Good (think: Fraud == 0)
                    df[(df[feature] == val) & (df[target] == 1)].count()[feature]]) # Bad (think: Fraud == 1)

    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])
    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])
    
    data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})

    data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])

    data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
    data.index = range(len(data.index))

    if pr:
        print(data)
        print('IV = ', data['IV'].sum())

    iv = data['IV'].sum()
    print('This variable\'s IV is (*1000):',iv*1000)
    print(df[feature].value_counts())
    return iv, data



feature_names = merged_df.columns[1:]
iv_list = []
for col in feature_names:
    iv, data = calc_iv(merged_df,col,'TARGET')
    iv_list.append((col,iv))


for item in iv_list: 
    print(f'iv = {iv_list[0]}, data= {iv_list[1]}')

# IV values
iv = pd.DataFrame(iv_list,columns=['Attribute','IV'])
filtered_df = iv[iv['IV']>0.02]

# Selected Features
selected_features = filtered_df['Attribute'].tolist()
selected_features


# # Build Model

x = aggregated_df[filtered_df['Attribute'].tolist()]
y = target['TARGET']

# Split the data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

# Apply SMOTE
smote = SMOTE(random_state=42)
x_train_res,y_train_res = smote.fit_resample(x_train,y_train)

# ## Random Forest Classifier

Random_forest_model = RandomForestClassifier(random_state=42, class_weight = "balanced")
Random_forest_model.fit(x_train_res,y_train_res)

y_pred_res = Random_forest_model.predict(x_test)

print(f'Accuracy={accuracy_score(y_test,y_pred_res)}')
print(f'recall={recall_score(y_test,y_pred_res)}')
print(f'precision={precision_score(y_test,y_pred_res)}')
print(f'f1={f1_score(y_test,y_pred_res)}')

