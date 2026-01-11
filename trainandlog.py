from src.data_fetch import Data_Fetch
from src.data_preprocess import *
from src.model_run import *
import mlflow
mlflow.autolog(disable=True)
import pyodbc
import pandas
from azure.storage.blob import ContainerClient, BlobServiceClient
from io import StringIO, BytesIO
import numpy
import os
import xlsxwriter
from pandarallel import pandarallel
from dateutil.relativedelta import relativedelta
import datetime
import pytz
import warnings
import pickle
import joblib
import json
import re

class LapsationTrain():

    def __init__(self, containerName: str = 'analytics', AZURE_STORAGE_CONNECTION_STRING: str = 'AZURE_STORAGE_CONNECTION_STRING', channel: str = None,
    start_date: str = None, end_date: str = None, sql_version: str = 'v1', version: str = 'v1', only_run_modelling = False):

        # Azure Blob Storage connection part
        self.containerName = containerName
        self.AZURE_STORAGE_CONNECTION_STRING = AZURE_STORAGE_CONNECTION_STRING
        connectionString_TCNP = os.environ.get(self.AZURE_STORAGE_CONNECTION_STRING)
        self.tcnp_vidhi_l0 = ContainerClient.from_connection_string(conn_str=connectionString_TCNP,container_name = self.containerName)
        self.Blob_client = BlobServiceClient.from_connection_string(connectionString_TCNP)

        ## Parameters within Databricks workspace
        self.start_date =  start_date
        self.end_date =  end_date
        self.channel = channel
        self.sql_version = sql_version
        self.version = version
        self.only_run_modelling = only_run_modelling

        # Experiment name
        self.experiment_name = f'Lapsation_{channel}_with_data_period_' + re.sub('-', '_', self.start_date) + '_' + re.sub('-', '_', self.end_date)

    def TrainAndLog(self):

        # Channel dictionary for folder reference
        channel_dict = {
            'HDFC': 'hdfc',
            'IBL': 'ibl',
            'CBI': 'cbi',
            'DBS': 'dbs',
            'AGENCY': 'agency',
            'DSF': 'dsf',
            'BROCA & CORP': 'broca',
            'POLICY BAZAR': 'pb'
        }

        # Channel code as per channels on datalake tables
        channel_code = {
            'HDFC': 'hdfc',
            'IBL': 'ibl',
            'CBI': 'cbi',
            'DBS': 'dbs',
            'AGENCY': 'AGCY',
            'DSF': 'AGCYDS',
            'BROCA & CORP': 'broca',
            'POLICY BAZAR': 'pb'
        }

        channel_cd = channel_code[self.channel]
        base_path = f'{channel_dict[self.channel]}/'
        base_query_path =  base_path+ f'train/{self.sql_version}/sql/base_query_{channel_dict[self.channel]}.sql'

        base_path_train = f'/Databricks/PipelinesData/Lapsation_DL/{channel_dict[self.channel]}/train/{self.version}/'
        raw_output_path = base_path_train + 'data/raw/'
        model_path = base_path_train + 'model/'
        report_path = base_path_train + 'report/'
        processed_path = base_path_train + 'data/processed/'
        config_file_path = base_path_train + 'config/'

        print('Training Period Start Date:', self.start_date)
        print('Training Period End Date:', self.end_date)
        print('Scoring using SQL version:', self.sql_version)
        print('Scoring Channel:', channel)
        print('Base Query Path :',base_query_path)
        print('Train Data Path :',base_path_train)
        print('Raw Data Output Path:', raw_output_path)
        print('Model Ouput Path :', model_path)
        print('Report Ouput Path :', report_path)
        print('Configuration File Path:',config_file_path)

        if self.only_run_modelling == False:
            obj = Data_Fetch(start_date = self.start_date, 
                        end_date = self.end_date, 
                        sql_path = base_query_path,
                        output_path = raw_output_path,
                        channel = channel_cd,
                    #   staging_con = Staging_Connection_String,
                        vidhi_con = self.tcnp_vidhi_l0, 
                        Blob_client = self.Blob_client, 
                        pdd = 'PREMIUM_DUE_DT', 
                        target='TARGET_LAPSATION',
                        config_file_path = config_file_path
                        )

            # Data Prep Part
            obj.data_prep()

            dp = data_preprocess(input_data_path = raw_output_path, 
                                output_data_path = processed_path, 
                                config_file_path = config_file_path,
                                # model_files_path = model_path, 
                                vidhi_con = self.tcnp_vidhi_l0, 
                                report_path = report_path,
                                test_split = 0.3,
                                eda_needed = False,
                                Blob_client = self.Blob_client)

            dp.run_pre_processing()

        model_obj = model_evaluate( processed_path = processed_path,
                                    config_file_path = config_file_path,
                                    model_path = model_path, 
                                    report_path = report_path, 
                                    vidhi_con = self.tcnp_vidhi_l0,
                                    valid = True,
                                    experiment_name = self.experiment_name,
                                    channel = self.channel)

        model_obj.optimize_and_evaluate()
        return print(f'Model Training and loggig completed for channel: {self.channel} and version: {self.version}')
