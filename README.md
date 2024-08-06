# Credit Card Approval Prediction

This project aims to predict credit card approval based on applicants' information and credit records. The dataset is a combination of application records and credit card records.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Features](#features)
- [Model](#model)
- [Evaluation](#evaluation)
- [License](#license)

## Project Overview
The project involves the following steps:
1. Data loading and merging
2. Data preprocessing and feature engineering
3. Model training using Random Forest Classifier
4. Model evaluation and performance metrics

## Installation

To set up this project on your local machine, follow these steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/amirshq/Credit-Card-approval-Prediction.git
   cd credit-card-approval-prediction

# Create a virtual environment (optional but recommended)
    
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install the required packages
    
    pip install -r requirements.txt

# USAGE 
## Run the script
    
    python script.py

## Script Breakdown

The script loads and merges the application and credit card records.
It fills missing values and encodes categorical features.
It performs feature selection based on Information Value (IV).
It trains a Random Forest model to predict credit card approval.
It evaluates the model using accuracy, recall, precision, and F1-score.
## Dataset
The dataset consists of two CSV files:

application_record.csv: Contains application information of credit card applicants.
credit_record.csv: Contains credit records of the applicants.

## Features
    CODE_GENDER: Gender of the applicant
    FLAG_OWN_CAR: Does the applicant own a car
    FLAG_OWN_REALTY: Does the applicant own a property
    CNT_CHILDREN: Number of children the applicant has
    AMT_INCOME_TOTAL: Total income of the applicant
    NAME_INCOME_TYPE: Income type of the applicant
    NAME_EDUCATION_TYPE: Education level of the applicant
    NAME_FAMILY_STATUS: Family status of the applicant
    NAME_HOUSING_TYPE: Housing type of the applicant
    DAYS_BIRTH: Age of the applicant in days
    DAYS_EMPLOYED: Employment duration of the applicant in days
    FLAG_MOBIL: Does the applicant have a mobile phone
    FLAG_WORK_PHONE: Does the applicant have a work phone
    FLAG_PHONE: Does the applicant have a phone
    FLAG_EMAIL: Does the applicant have an email
    OCCUPATION_TYPE: Occupation of the applicant
    CNT_FAM_MEMBERS: Number of family members of the applicant
    MONTHS_BALANCE: Balance of months in the credit record
    STATUS: Status of the credit record

## Model
A Random Forest Classifier is used for this project. The model is trained with oversampling using SMOTE to handle class imbalance.

## Evaluation
The model is evaluated using the following metrics:

Accuracy
Recall
Precision
F1-score

# License
MIT License

Copyright (c) 2024 [Amir shahcheraghian]

## Dataset License

The dataset used in this project is sourced from Kaggle and is available at:
https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction

Please refer to the dataset's page for specific licensing terms and usage guidelines.