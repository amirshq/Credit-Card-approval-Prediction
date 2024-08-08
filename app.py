import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Load your data
application_record_df = pd.read_csv('Data/application_record.csv')
credit_card_record_df = pd.read_csv('Data/credit_record.csv')

# Merge DataFrames
merged_df = pd.merge(application_record_df, credit_card_record_df, how='outer', on='ID')

# Separate numeric and non-numeric columns
numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
non_numeric_cols = merged_df.select_dtypes(exclude=[np.number]).columns

# Impute missing values
# Numeric columns
numeric_imputer = SimpleImputer(strategy='mean')
merged_df[numeric_cols] = numeric_imputer.fit_transform(merged_df[numeric_cols])

# Non-numeric columns
non_numeric_imputer = SimpleImputer(strategy='most_frequent')
merged_df[non_numeric_cols] = non_numeric_imputer.fit_transform(merged_df[non_numeric_cols])

# Encoding categorical columns
categorical_columns = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'OCCUPATION_TYPE', 'CODE_GENDER', 'NAME_INCOME_TYPE', 
                       'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']

# Initialize LabelEncoder
label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    merged_df[col] = label_encoders[col].fit_transform(merged_df[col])

merged_df['TARGET'] = merged_df['STATUS'].apply(lambda x: 1 if x in ['1', '2', '3', '4', '5'] else 0)

target = merged_df.groupby('ID')['TARGET'].max().reset_index()

# Aggregate features
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
    'MONTHS_BALANCE': 'max',
    'STATUS': 'last'
}).reset_index()
'''
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
'''

iv_list = [('CODE_GENDER', 0.008703776533994529),
 ('FLAG_OWN_CAR', 0.00010730199701853978),
 ('FLAG_OWN_REALTY', 0.008008561629814266),
 ('CNT_CHILDREN', 0.004030524780812981),
 ('AMT_INCOME_TOTAL', 0.11382239749846743),
 ('NAME_INCOME_TYPE', 0.010413597023017808),
 ('NAME_EDUCATION_TYPE', 0.01002765863464529),
 ('NAME_FAMILY_STATUS', 0.005073091522869707),
 ('NAME_HOUSING_TYPE', 0.0019369219868989927),
 ('DAYS_BIRTH', 1.387618877373408),
 ('DAYS_EMPLOYED', 0.9137270200449777),
 ('FLAG_WORK_PHONE', 0.003915477537295826),
 ('FLAG_PHONE', 0.00026787576985636176),
 ('FLAG_EMAIL', 3.6611685551246236e-05),
 ('OCCUPATION_TYPE', 0.027940998925250954),
 ('CNT_FAM_MEMBERS', 0.0036296486413072535),
 ('MONTHS_BALANCE', 0.7429999346706175)]

for item in iv_list: 
    print(f'iv = {iv_list[0]}, data= {iv_list[1]}')

# IV values
iv = pd.DataFrame(iv_list,columns=['Attribute','IV'])
filtered_df = iv[iv['IV']>0.02]

# Handle target and features
x = aggregated_df[filtered_df['Attribute'].tolist()]
y = target['TARGET']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Apply SMOTE
smote = SMOTE(random_state=42)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

# Train Random Forest Classifier
Random_forest_model = RandomForestClassifier(random_state=42, class_weight="balanced")
Random_forest_model.fit(x_train_res, y_train_res)

# Predictions and metrics
y_pred_res = Random_forest_model.predict(x_test)

print(f'Accuracy={accuracy_score(y_test, y_pred_res)}')
print(f'Recall={recall_score(y_test, y_pred_res)}')
print(f'Precision={precision_score(y_test, y_pred_res)}')
print(f'F1 Score={f1_score(y_test, y_pred_res)}')
print(classification_report(y_test, y_pred_res))
print(confusion_matrix(y_test, y_pred_res))

