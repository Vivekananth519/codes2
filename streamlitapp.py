# Importing relevant libraries
import pandas as pd
import sys
import os
import numpy as np
import seaborn as sns
from sklearn import metrics
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    accuracy_score
)
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score
import lightgbm as lgb
import xgboost as xgb
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import matplotlib.pyplot as plt
import category_encoders as ce
pd.set_option('display.max_columns', None)
from unipath import Path
p = os.getcwd() + '/Documents/Projects/Bank_API_3'
from eda_utils.Model_Summary import Summary_Automation
from eda_utils.eda_report import EDAReport
from model_utils.model_run import model_evaluate
from data.data_preprocess import data_preprocess
from model_utils.model_run import model_evaluate
from eda_utils.eda_to_ppt import *
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import streamlit as st

st.title('Automatic Model Building App for External Partners - Tata AIA')

tab1, tab2 = st.tabs(["Data Pre-Processing", "Model Running"])

with tab1:
    master_file_path = st.file_uploader("Upload your Master file", type=["xlsx"])
    print(master_file_path)
    input_data_path = st.file_uploader('Input Data Path (optional)',type=["csv"])
    target = st.text_input('Target Variable', value='li_flag')
    test_split = st.slider('Test Split', min_value=0.0, max_value=1.0, value=0.3)
    generate_random_data = st.checkbox('Generate Random Data', value=True)
    n_samples = st.number_input('Number of Samples', min_value=1, value=10000)
    validation = st.checkbox('Use Validation', value=False)

    if st.button('Run Preprocessing'):
        data_obj = data_preprocess(
            master_file_path=master_file_path,
            input_data_path=input_data_path if input_data_path else None,
            target=target,
            test_split=test_split,
            generate_random_data=generate_random_data,
            n_samples=n_samples,
            validation=validation
        )
        if validation:
            train, test, valid, s_feature = data_obj.run_pre_processing()
            st.session_state['train'] = train
            st.session_state['test'] = test
            st.session_state['valid'] = valid
            st.session_state['s_feature'] = s_feature
        else: 
            train, test, s_feature = data_obj.run_pre_processing()
            st.session_state['train'] = train
            st.session_state['test'] = test
            st.session_state['s_feature'] = s_feature

        st.write('Training data shape:', train.shape)
        st.write('Testing data shape:', test.shape)
        if validation:
            st.write('Validation data shape:', valid.shape)
        st.write('Selected Features:', s_feature)
    with open(p +'/EDA Report/EDA_Report_formatted.xlsx', 'rb') as my_file:
        st.download_button(label = 'Download the Excel EDA Report', 
                        data = my_file, 
                        file_name = 'EDA_Report_formatted.xlsx', 
                        mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        
    with open(p + '/EDA Report/Data_Analysis_Presentation.pptx', 'rb') as my_file:
        st.download_button(label = 'Download the Power Point EDA Report', 
                        data = my_file, 
                        file_name = 'Data_Analysis_Presentation.pptx')
with tab2:
    model_type = st.selectbox("Select the model for which you want the model explainability tabs:", 
                 ('lightgbm', 'xgboost', 'randomforest', 'logistic'))
    if st.button('Run Models'):
        train = st.session_state['train']
        test = st.session_state['test']
        s_feature = st.session_state['s_feature']
        if validation:
            valid = st.session_state['valid']
            # Example usage
            model_obj = model_evaluate(train = train, eval = test, target = target, s_feature = s_feature, valid=valid)
            result = model_obj.optimize_and_evaluate()
            st.write("Model run complete")
            st.write(f"Training Results ({model_type}): \n Decile Summary: \n", result[model_type][1])
            st.write(f"Training Results ({model_type}): \n Classification Report: \n", result[model_type][4])
            st.write(f"Testing Results ({model_type}): \n Decile Summary: \n", result[model_type][2])
            st.write(f"Testing Results ({model_type}): \n Classification Report: \n", result[model_type][5])
            st.write(f"Testing Results ({model_type}): \n Decile Summary: \n", result[model_type][3])
            st.write(f"Testing Results ({model_type}): \n Classification Report: \n", result[model_type][6])
        else:
            model_obj = model_evaluate(train = train, eval = test, target = target, s_feature = s_feature, valid=None)
            result = model_obj.optimize_and_evaluate()
            st.write("Model run complete")
            st.write(f"Training Results ({model_type}): \n Decile Summary: \n", result[model_type][1])
            st.write(f"Training Results ({model_type}): \n Classification Report: \n", result[model_type][3])
            st.write("Testing Results ({model_type}): \n Decile Summary: \n", result[model_type][2])
            st.write(f"Testing Results ({model_type}): \n Classification Report: \n", result[model_type][4])
        st.session_state['Model_Result'] = result

    with open(f'{p}/Model Summary/{model_type} Model_Summary.xlsx', 'rb') as my_file:
        st.download_button(label = f'Download the {model_type} Model Report', 
                        data = my_file, 
                        file_name = f'{model_type} Model_Summary.xlsx', 
                        mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    if st.button("Run Model Explainability"):
        train = st.session_state['train']
        test = st.session_state['test']
        s_feature = st.session_state['s_feature']
        result = st.session_state['Model_Result']
        explainer = ClassifierExplainer(model = result[model_type][0], 
                                        X = train[s_feature], 
                                        y = train[target],
                                        idxs = train['cust_id'],
                                        labels = ["Non LI Holder", "LI Holder"],
                                        target = 'LI Holder')
        dashboard = ExplainerDashboard(explainer, title=f"{model_type} Explainer Dashboard")
        st.write(f"Dashboard for {model_type} running on port http://localhost:8050/")
        dashboard.run(port=8050)
    else:
        pass
