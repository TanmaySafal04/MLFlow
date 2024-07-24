import mlflow


def cal_sum(x,y):
    return x+y



if __name__ == '__main__':

    with mlflow.start_run():
        #MLFLOW server started    
        x,y=10,20
        sum=cal_sum(x,y)
        #tracking 
        mlflow.log_param("x",x)
        mlflow.log_param("y",y)
        mlflow.log_metric("Sum",sum)