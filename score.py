from src.data_fetch import Data_Fetch
from src.scoring_data_preprocessing import *
from src.model_predict import *
from src.folder_utils import get_latest_version_info
import mlflow
mlflow.autolog(disable=True)
import pyodbc
import pandas
from azure.storage.blob import ContainerClient, BlobServiceClient
from io import StringIO, BytesIO
from dateutil.relativedelta import relativedelta
import numpy
import os
import xlsxwriter
from pandarallel import pandarallel
import warnings
import pickle
import joblib
import json
import re
import datetime
import pytz

class LapsationScore():

    def __init__(self, containerName: str = 'analytics', AZURE_STORAGE_CONNECTION_STRING: str = 'AZURE_STORAGE_CONNECTION_STRING', channel: str = None,
    start_date: str = None, end_date: str = None, sql_version: str = 'v1', version: str = None, only_run_modelling = False):

        # Azure Blob Storage connection part
        self.containerName = containerName
        self.AZURE_STORAGE_CONNECTION_STRING = AZURE_STORAGE_CONNECTION_STRING
        connectionString_TCNP = os.environ.get(self.AZURE_STORAGE_CONNECTION_STRING)
        self.tcnp_vidhi_l0 = ContainerClient.from_connection_string(conn_str=connectionString_TCNP, container_name=self.containerName)
        self.Blob_client = BlobServiceClient.from_connection_string(connectionString_TCNP)

        ## Parameters within Databricks workspace
        self.start_date =  start_date
        self.end_date =  end_date
        self.channel = channel
        self.sql_version = sql_version
        self.version = version
        self.sql_version = sql_version

    def Score(self):
        run_timestamp = (datetime.datetime.now(pytz.timezone('Asia/Kolkata'))-relativedelta(days=2)).strftime('%Y/%m/%d')

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

        channel_cd = channel_code[channel]

        # start_date =  datetime.datetime(datetime.datetime.today().year, (datetime.datetime.today() - relativedelta(days=1)).month, 1)
        # end_date =  pd.to_datetime(start_date) + relativedelta(months=5) - relativedelta(days=1)

        # Relevant Paths
        base_path = f'{channel_dict[self.channel]}/'
        base_query_path =  base_path+ f'scoring/{self.sql_version}/sql/base_query_{channel_dict[self.channel]}.sql'

        if channel != 'CUB':
            base_path_version = f'/Databricks/PipelinesData/Lapsation_DL/{channel_dict[self.channel]}/train/'
            if self.version is None:
                self.version_info = get_latest_version_info(container = self.tcnp_vidhi_l0, parent_folder = base_path_version)
                self.version = self.version_info.get_latest_version()
            base_path_train = f'/Databricks/PipelinesData/Lapsation_DL/{channel_dict[self.channel]}/train/{self.version}/'
            processed_path_train = f'{base_path_train}/data/processed/'
            config_file_path = base_path_train + 'config/'
            base_path_scoring = f'/Databricks/PipelinesData/Lapsation_DL/{channel_dict[self.channel]}/scoring/{run_timestamp}/{self.version}/'
            raw_output_path = base_path_scoring + 'data/raw/'
            model_path = base_path_train + 'model/'
            processed_path = base_path_scoring + 'data/processed/'
            report_path = base_path_scoring + 'report/'
        else:
            base_path_version = f'/Databricks/PipelinesData/Lapsation_DL/hdfc/train/'
            if self.version is None:
                self.version_info = get_latest_version_info(container = self.tcnp_vidhi_l0, parent_folder = base_path_version)
                self.version = self.version_info.get_latest_version()
            base_path_train = f'/Databricks/PipelinesData/Lapsation_DL/hdfc/train/{self.version}/'
            processed_path_train = f'{base_path_train}/data/processed/'
            config_file_path = base_path_train + 'config/'
            base_path_scoring = f'/Databricks/PipelinesData/Lapsation_DL/{channel_dict[self.channel]}/scoring/{run_timestamp}/{self.version}/'
            raw_output_path = base_path_scoring + 'data/raw/'
            model_path = base_path_train + 'model/'
            report_path = base_path_scoring + 'report/'
            processed_path = base_path_scoring + 'data/processed/'

        config_file = download_yaml_data(config_file_path + 'config.yaml', tcnp_vidhi_l0 = self.tcnp_vidhi_l0)
        tr_start_date= config_file['data_period']['start_date']
        tr_end_date = config_file['data_period']['end_date']
        # cat_col = config_file['features']['cat_col']
        # cont_col = config_file['features']['cont_col']
        # ignore_col = config_file['features']['ignore_col']
        pdd = config_file['PDD']
        s_features = config_file['selected_features']
        model_type = config_file['best_model_type']
        parent_run_id = config_file['best_run_id']

        # Experiment name
        self.experiment_name = f'Lapsation_{channel}_with_data_period_' + re.sub('-', '_', tr_start_date) + '_'  + re.sub('-', '_', tr_end_date)

        print('Scoring Start Date:', self.start_date)
        print('Scoring End Date:', self.end_date)
        print('Scoring using SQL version:', self.sql_version)
        print('Scoring using version:', self.version)
        print('Scoring Channel:', self.channel)

        print('Base Query Path :',base_query_path)
        print('Scoring Data Path :',base_path_scoring)
        print('Raw Data Output Path:', raw_output_path)
        print('Model Ouput Path :', model_path)
        print('Report Ouput Path :', report_path)

        # Data Fetch part
        obj = Data_Fetch(start_date = start_date, 
                end_date = end_date, 
                sql_path = base_query_path,
                output_path = raw_output_path,
                channel = channel_cd,
                #   staging_con = Staging_Connection_String,
                vidhi_con = self.tcnp_vidhi_l0, 
                Blob_client = self.Blob_client, 
                pdd = pdd, 
                target='TARGET_LAPSATION',
                config_file_path = config_file_path,
                is_scoring_call = True
                )
        # Data prep part
        obj.data_prep()


        # Data pre-processing part
        dp = data_preprocess(input_data_path = raw_output_path, 
                            output_data_path = processed_path, 
                            config_file_path = config_file_path,
                            vidhi_con = self.tcnp_vidhi_l0,
                            Blob_client = self.Blob_client)
        dp.run_pre_processing()

        # Model Scoring part
        score_obj = score_data(start_date = self.start_date,
                            end_date = self.end_date,
                            scoring_data_path = processed_path,  
                            train_data_path = processed_path_train, 
                            s_features = s_features, 
                            model_path = model_path, 
                            report_path = report_path, 
                            vidhi_con = self.tcnp_vidhi_l0, 
                            model_type = model_type, 
                            Blob_client = self.Blob_client, 
                            channel = self.channel,
                            pdd = pdd, 
                            experiment_name = self.experiment_name,
                            parent_run_id = parent_run_id)

        drift, retrain, trigger_retrain_loop = score_obj.predict()

        print('Drift Check:', drift)
        print('Retrain Check:', retrain)
        print('Retrain Trigger:', trigger_retrain_loop)

        if trigger_retrain_loop:
            print('Re-training loop has been triggered')
            print('Model training process will be initiated by assuming last 18 months of data')

            self.new_end_date = pd.to_datetime(datetime.datetime.today().replace(day=1) + relativedelta(months=-2, day=31))
            self.new_start_date = str(self.new_end_date + relativedelta(days=1) - relativedelta(months=18))[:10]
            self.new_end_date = str(self.new_end_date)

            Train_obj = LapsationTrain( containerName = self.containerName, AZURE_STORAGE_CONNECTION_STRING = self.AZURE_STORAGE_CONNECTION_STRING, 
                                        start_date = self.new_start_date, end_date = self.new_end_date, sql_version = self.sql_version, 
                                        version = self.version_info().create_new_version(self.version), only_run_modelling = False)
            Train_obj.TrainAndLog()
            print(f'Model Re-training has been completed with the data period: {self.new_start_date} to {self.new_end_date}')
            print(f'New version of the model is: {self.version_info().create_new_version(self.version)}')
            return print('Done')
        return print(f'End to End Scoring process has been completed for the period: {self.start_date} to {self.end_date} with version {self.version}')





start_date = '2025-10-01'
end_date = '2025-12-31'
sql_version = 'v1'
version = 'v1'

channel_list = [
                        # 'CUB', 
                        # 'DBS', 
                        # 'IBL', 
                        # 'CBI', 
                        'AGENCY', 
                        'DSF', 
                        # 'BROCA & CORP', 
                        # 'POLICY BAZAR',
                        # 'HDFC'
                        ]
for channel in channel_list:
    score_obj = LapsationScore( containerName = 'analytics', 
                                AZURE_STORAGE_CONNECTION_STRING = 'AZURE_STORAGE_CONNECTION_STRING', 
                                start_date = start_date, 
                                end_date = start_date, 
                                sql_version = 'v1', 
                                version = None, 
                                only_run_modelling = False)

    score_obj.Score()
