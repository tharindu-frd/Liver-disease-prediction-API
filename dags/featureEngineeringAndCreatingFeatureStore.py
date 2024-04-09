from utils.utils import read_yaml
from utils.utils import  download_csv_from_s3




def preprocessing(**kwargs):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    from sklearn.preprocessing import StandardScaler
    import os 
    import boto3
    from sklearn.model_selection import train_test_split
    import shutil
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, 'config.yaml')

     
        config = read_yaml(config_file)

        
        folder_location = os.path.join(config['validation']['destination_after_validation']['mainfolder'],
                                       config['validation']['destination_after_validation']['subfolder'])
        csv_files = [file for file in os.listdir(folder_location) if file.endswith('.csv')]
        dfs = []

        for csv_file in csv_files:
            csv_file_path = os.path.join(folder_location, csv_file)
            df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)

        numerical_columns = combined_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = combined_df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

       
        for col in numerical_columns:
            if combined_df[col].isnull().sum() > 0:
                null_percentage = combined_df[col].isnull().mean() * 100
                if null_percentage < 3:
                    combined_df.dropna(subset=[col], inplace=True)
                else:
                    median_value = combined_df[col].median()
                    combined_df[col].fillna(median_value, inplace=True)

        for col in categorical_columns:
            if combined_df[col].isnull().sum() > 0:
                null_percentage = combined_df[col].isnull().mean() * 100
                if null_percentage < 3:
                    combined_df.dropna(subset=[col], inplace=True)
                else:
                    mode_value = combined_df[col].mode()[0]
                    combined_df[col].fillna(mode_value, inplace=True)

        combined_df['Gender of the patient'] = combined_df['Gender of the patient'].replace({'Male': 1, 'Female': 0})
        combined_df.drop_duplicates(inplace=True)

       

        return combined_df  # Return the processed DataFrame

    except Exception as e:
        print("Preprocessing encountered an error:", e)
        return pd.DataFrame()  # Return an empty DataFrame in case of an error


def creating_feature_store(**kwargs):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    from sklearn.preprocessing import StandardScaler
    import os 
    import boto3
    from sklearn.model_selection import train_test_split
    import shutil
    try:
        combined_df = preprocessing()
        y = combined_df['Result']
        X = combined_df.drop(columns=['Result'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)

        if combined_df.empty:
            print("Preprocessing did not produce a valid DataFrame.")
            return

        # Save train and test sets locally as CSV files
        X_train.to_csv('X_train.csv', index=False)
        X_test.to_csv('X_test.csv', index=False)
        y_train.to_csv('y_train.csv', index=False)
        y_test.to_csv('y_test.csv', index=False)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, 'config.yaml')

        # Read the YAML config file
        config = read_yaml(config_file)


        aws_access_key_id = config['featureStore']['Access_key']
        aws_secret_access_key = config['featureStore']['Secret_Access_Key']
        bucket_name = config['featureStore']['bucketname']

        s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

        
        file_prefixes = ['feature_store/X_train.csv', 'feature_store/X_test.csv', 'feature_store/y_train.csv', 'feature_store/y_test.csv']
        for prefix in file_prefixes:
            local_file_path = f"{prefix.split('/')[-1]}"
            with open(local_file_path, "rb") as f:
                s3_client.upload_fileobj(f, bucket_name, prefix)
            print(f"Uploaded {local_file_path} to {prefix}.")

        
        for file_name in ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']:
            if os.path.exists(file_name):
                os.remove(file_name)
                print(f"File {file_name} removed successfully.")
            else:
                print(f"File {file_name} does not exist in the current folder.")

    except Exception as e:
        print(f"Error creating feature store: {e}")



if __name__ == "__main__":
    preprocessing()
    creating_feature_store()



'''
cd src
python featureEngineeringAndCreatingFeatureStore

'''
