### As the first step use DBSCAN to check the outliers if outlier % is less than 5% remove them and if not fit a model while keeping them 

def load_data_from_s3(**context):
    import os
    import boto3
    import pandas as pd
    import numpy as np
    import mlflow 
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from utils.utils import read_yaml
    from utils.utils import  download_csv_from_s3
    from urllib.parse import urlparse
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    import xgboost as xgb
    from sklearn.utils import resample
    try:
        config = read_yaml('config.yaml')
        bucket_name = 'liverfeaturestore'
        file_prefixes = ['X_train', 'X_test', 'y_train', 'y_test']
        local_files = []

        s3 = boto3.client('s3',
                          aws_access_key_id=config['modelStore']['Access_key'],
                          aws_secret_access_key=config['modelStore']['Secret_Access_Key'])

        for prefix in file_prefixes:
            s3_key = f"feature_store/{prefix}.csv"
            local_file_path = f"{prefix}.csv"
            try:
                s3.download_file(bucket_name, s3_key, local_file_path)
                local_files.append(local_file_path)
            except Exception as e:
                print(f"Error downloading file {s3_key} from S3: {e}")

        
        if len(local_files) == len(file_prefixes):
            X_train = pd.read_csv(local_files[0])
            X_test = pd.read_csv(local_files[1])
            y_train = pd.read_csv(local_files[2])
            y_test = pd.read_csv(local_files[3])

            context['ti'].xcom_push(key='X_train', value=X_train)
            context['ti'].xcom_push(key='X_test', value=X_test)
            context['ti'].xcom_push(key='y_train', value=y_train)
            context['ti'].xcom_push(key='y_test', value=y_test)
        else:
            print("Error: Not all files were downloaded successfully.")
            return None

    except Exception as e:
        print(f"Error loading data from S3: {e}")
        return None





def model_training(**context):
    import os
    import boto3
    import pandas as pd
    import numpy as np
    import mlflow 
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from utils.utils import read_yaml
    from utils.utils import  download_csv_from_s3
    from urllib.parse import urlparse
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    import xgboost as xgb
    from sklearn.utils import resample
    from mlflow.tracking import MlflowClient

   
    try:
        X_train = context['ti'].xcom_pull( key='X_train')
        X_test = context['ti'].xcom_pull(key='X_test')
        y_train = context['ti'].xcom_pull(key='y_train')
        y_test = context['ti'].xcom_pull( key='y_test')







        ### Set the tracking uri
        # 'mysql://username:password@hostname:port/database_name'
        uri = 'ec2-16-171-41-165.eu-north-1.compute.amazonaws.com'
        mlflow.set_tracking_uri(f"http://{uri}:5080")
        bucket_name = 'liverfeaturestore'

        with mlflow.start_run():
            ###########################################################################
            ###########################################################################
            ###########################################################################
            ##################     Approach - 01 : ( original set + RF)    ############
            ###########################################################################
            ###########################################################################

            rf = RandomForestClassifier(random_state=28)
            param_grid = {
                'n_estimators': [50, 60, 70, 80, 90, 100, 150],
                'max_depth': [3, 4, 5, None],
                'min_samples_split': [2, 3, 4],
                'min_samples_leaf': [1, 2, 3],
                'max_features': ['auto', 'sqrt', 'log2']
            }

            grid_search_1 = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
            grid_search_1.fit(X_train, y_train)

            best_model_1 = grid_search_1.best_estimator_
            y_pred_1 = best_model_1.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred_1)
            precision = precision_score(y_test, y_pred_1)
            recall = recall_score(y_test, y_pred_1)
            auc = roc_auc_score(y_test, y_pred_1)

            mlflow.log_params(grid_search_1.best_params_)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("auc", auc)

            mlflow.sklearn.log_model(
                best_model_1,
                registered_model_name="RFNORMALMODEL",
                artifact_path='models'
            )



        
            ###########################################################################
            ###########################################################################
            ###########################################################################
            ############            Approach - 02 : ( with SMOTE + XGB)     ###########
            ###########################################################################
            ###########################################################################
            ###########################################################################

            smote = SMOTE(random_state=28)
            X_resampled_train, y_resampled_train = smote.fit_resample(X_train, y_train)
            X_resampled_train = pd.DataFrame(X_resampled_train)
            y_resampled_train = pd.DataFrame(y_resampled_train)

            xgb_classifier = xgb.XGBClassifier(random_state=28)

            y_resampled_train = y_resampled_train - 1
            y_test_new = y_test - 1
            grid_search_2 = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=5, n_jobs=-1)
            grid_search_2.fit(X_resampled_train, y_resampled_train)

            best_model_2 = grid_search_2.best_estimator_
            y_pred_2 = best_model_2.predict(X_test)

            accuracy = accuracy_score(y_test_new, y_pred_2)
            precision = precision_score(y_test_new, y_pred_2)
            recall = recall_score(y_test_new, y_pred_2)
            auc = roc_auc_score(y_test_new, y_pred_2)

            mlflow.log_params(grid_search_2.best_params_)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("auc", auc)

            mlflow.sklearn.log_model(
                best_model_2,
                registered_model_name="SMOTEXGB",
                artifact_path='models'
            )

            ###########################################################################
            ###########################################################################
            ###########################################################################
            ############       Approach - 03 : ( down sampling  + RF )      ###########
            ###########################################################################
            ###########################################################################
            ###########################################################################

            majority_class = X_train[y_train[0] == 1]
            minority_class = X_train[y_train[0] == 0]

            majority_downsampled = resample(majority_class,
                                            replace=False,
                                            n_samples=minority_class.shape[0],
                                            random_state=42)

            X_downsampled = np.vstack([minority_class, majority_downsampled])
            y_downsampled = np.concatenate([np.zeros(minority_class.shape[0]), np.ones(minority_class.shape[0])])

            shuffle_indices = np.arange(len(X_downsampled))
            np.random.shuffle(shuffle_indices)
            X_downsampled = X_downsampled[shuffle_indices]
            y_downsampled = y_downsampled[shuffle_indices]

            X_resampled_train = pd.DataFrame(X_downsampled)
            y_resampled_train = pd.DataFrame(y_downsampled)

            rf = RandomForestClassifier(random_state=28)
            param_grid = {
                'n_estimators': [50, 60, 70, 80, 90, 100, 150],
                'max_depth': [3, 4, 5, None],
                'min_samples_split': [2, 3, 4],
                'min_samples_leaf': [1, 2, 3],
                'max_features': ['auto', 'sqrt', 'log2']
            }

            grid_search_3 = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
            grid_search_3.fit(X_resampled_train, y_resampled_train)

            best_model_3 = grid_search_3.best_estimator_
            y_pred_3 = best_model_3.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred_3)
            precision = precision_score(y_test, y_pred_3)
            recall = recall_score(y_test, y_pred_3)
            auc = roc_auc_score(y_test, y_pred_3)

            mlflow.log_params(grid_search_3.best_params_)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("auc", auc)

            mlflow.sklearn.log_model(
                best_model_3,
                registered_model_name="DOWNSAMPLINGRF",
                artifact_path='models'
            )
        
        
    except Exception as e:
        print(f"Error in model training: {e}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data_from_s3()
    print(X_train.shape)
    print(y_train.shape)

    if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
        model_training(X_train, X_test, y_train, y_test)
    else:
        print("Failed to load data from S3.")









       
