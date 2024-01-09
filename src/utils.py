import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,model):
    try:
        report={}
        # print(model)
        for i in range(len(model)):
            
            mod=list(model.values())[i]
            #train model
            mod.fit(X_train,y_train)

            #predict Testing data
            y_test_pred=mod.predict(X_test)

            #get R2 scores for train and test data
            #train_model_score=r2_score(ytrain,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)

            report[list(model.keys())[i]]=test_model_score
        return report 
    
    except Exception as e:
        logging.info('Exception occured duuring model training ')
        raise CustomException(e,sys)


