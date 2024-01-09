import os
import sys 
from src.logger import logging 
from src.exception import CustomException
import pandas as pd 
# from src.componets.data_ingestion
from src.componets.data_ingestion import dataIngestion
from src.componets.data_transformation import DataTransformation
from src.componets.model_trainer import ModelTrainer


if __name__=='__main__':
    obj=dataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)

    data_tranformation=DataTransformation()
    

    train_arr,test_arr,_=data_tranformation.initiate_data_transformation(train_data_path,test_data_path)

    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)
