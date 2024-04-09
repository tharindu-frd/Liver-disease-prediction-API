import mlflow
from utils.utils import read_yaml
import pickle
import boto3

def find_model_with_lowest_accuracy():
    import mlflow
    from utils.utils import read_yaml
    import pickle
    import boto3
    import os
    from mlflow.tracking import MlflowClient
    try:
        # Connect to the MLflow tracking server
        uri = 'eec2-16-171-41-165.eu-north-1.compute.amazonaws.com'
        mlflow.set_tracking_uri(f"http://{uri}:5080")

        # Get all runs across all experiments
        runs = mlflow.search_runs()

        lowest_accuracy = float('inf')
        lowest_accuracy_run_id = None

       
        for _, run in runs.iterrows():
            accuracy = run["metrics.accuracy"]
            if accuracy < lowest_accuracy:
                lowest_accuracy = accuracy
                lowest_accuracy_run_id = run["run_id"]

       
        model_uri = f"runs:/{lowest_accuracy_run_id}/models"
        model = mlflow.sklearn.load_model(model_uri)

        
        with open("lowest_accuracy_model.pkl", "wb") as f:
            pickle.dump(model, f)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, 'config.yaml')

       
        config = read_yaml(config_file)
        




        access_key =  config['modelStore']['Access_key'] 
        secret_access_key =  config['modelStore']['Secret_Access_Key'] 
        bucketname =  config['modelStore']['bucketname'] 
        
        s3 = boto3.client('s3', 
                          aws_access_key_id=access_key, 
                          aws_secret_access_key=secret_access_key)
        s3.upload_file("lowest_accuracy_model.pkl", bucketname, "lowest_accuracy_model.pkl")

    except Exception as e:
        print(f"Error finding model with lowest accuracy: {e}")


if __name__ == "__main__":
    find_model_with_lowest_accuracy()
