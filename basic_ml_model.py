import pandas as pd
import numpy as np
import os
from ucimlrepo import fetch_ucirepo 
import mlflow
import mlflow.sklearn  

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,roc_auc_score
from sklearn.model_selection import train_test_split

from ucimlrepo import fetch_ucirepo 

import argparse
  
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 

def get_data():
    # fetch dataset 
    wine = fetch_ucirepo(id=186) 
    try:
        # data (as pandas dataframes) 
       X = wine_quality.data.features 
       y = wine_quality.data.targets 
       df = pd.concat([X, y],axis=1, join='outer') 
       return df
    except Exception as e:
        raise e

# with mlflow.start_run():
def main(n_estimators,max_depth):
    df = get_data()
    #Train test split
    train,test=train_test_split(df)
    x_train = train.drop(['quality'],axis=1)
    x_test = test.drop(['quality'],axis=1)

    y_train = train['quality']
    y_test = test['quality']
    
    rf=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
    rf.fit(x_train,y_train)
    pred = rf.predict(x_test)
    prob_pred = rf.predict_proba(x_test)
            
    # mlflow.log_param("n_estimators", n_estimators)
        
    # mlflow.log_param("n_estimators", max_depth)
    return pred,y_test,prob_pred,rf

def evaluate(y_pred,y_true,prob_pred):
    
    accuracy = accuracy_score(y_true,y_pred)
    roc = roc_auc_score(y_true,prob_pred,multi_class='ovr')
    
    # mlflow.log_metric("Accuracy", accuracy)
        
    # mlflow.log_metric("ROC_AUC_score Score", roc)
    print(f"Accuracy Score: {accuracy}")
    print(f"ROC_AUC_score Score: {roc}")
    return accuracy,roc
    


if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--n_estimators","-n", default=50, type = int)
    args.add_argument("--max_depth","-m", default=5, type = int)
    parse_args=args.parse_args()
    try:
        with mlflow.start_run():
                
            y_pred,y_true,pred_prob,rf = main(n_estimators = parse_args.n_estimators, max_depth = parse_args.max_depth)
            accuracy,roc = evaluate(y_pred,y_true,pred_prob)  
            mlflow.log_param("n_estimators",parse_args.n_estimators)
            mlflow.log_param("max_depth",parse_args.max_depth)
            mlflow.log_metric("Accuracy_score", accuracy)
            mlflow.log_metric("ROC_AUC_Score", roc)
            mlflow.sklearn.log_model(rf,"Random Forest Model")
    except Exception as e:
        raise e 

