import pandas as pd
import numpy as np
import os
from ucimlrepo import fetch_ucirepo 
import mlflow
import mlflow.sklearn  

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score
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


def main(n_estimators,max_depth):
    df = get_data()
    #Train test split
    train,test=train_test_split(df)
    x_train = train.drop(['quality'],axis=1)
    x_test = test.drop(['quality'],axis=1)

    y_train = train['quality']
    y_test = test['quality']
    # Model training
    # lr = ElasticNet()
    # lr.fit(x_train,y_train)
    # pred = lr.predict(x_test)
    rf=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
    rf.fit(x_train,y_train)
    pred = rf.predict(x_test)
    return pred,y_test

def evaluate(y_pred,y_true):
    #y_pred,y_true = main()
    # mae = mean_absolute_error(y_true,y_pred)
    # mse = mean_squared_error(y_true,y_pred)
    # rmse = np.sqrt(mse)
    # r2 = r2_score(y_true,y_pred)
    accuracy = accuracy_score(y_true,y_pred)
    print(f"Accuracy Score: {accuracy}")
    return accuracy
    #return mae,mse,rmse,r2

# #mae,mse,rmse,r2=evaluate()
# accuracy = evaluate()
# # print(f"Mean Absolute Error: {mae}")
# # print(f"Mean Squared Error: {mse}")
# # print(f"Root Mean Squared Error: {rmse}")
# # print(f"R2 Score: {r2}")
# print(f"Accuracy Score: {accuracy}")




if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--n_estimators","-n", default=50, type = int)
    args.add_argument("--max_depth","-m", default=5, type = int)
    parse_args=args.parse_args()
    try:
        y_pred,y_true = main(n_estimators = parse_args.n_estimators, max_depth = parse_args.max_depth)
        evaluate(y_pred,y_true)    
    except Exception as e:
        raise e

