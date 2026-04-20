import mlflow
import os
from dotenv import load_dotenv
from mlflow import client

load_dotenv()

dagshub_token = os.getenv("DAGSHUB_TOKEN")
dagshub_username = os.getenv("DAGSHUB_USERNAME") 

if not dagshub_token or not dagshub_username:
    raise EnvironmentError("DAGSHUB_TOKEN or DAGSHUB_USERNAME not set in env")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

repo_owner = "virajdeshmukh080818"
repo_name = "AML"
tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
mlflow.set_tracking_uri(tracking_uri)

client = mlflow.tracking.MlflowClient(tracking_uri)

model_name = "AML_Laundering_Detector"

def promote():
    print(f"Chechking for new models in {model_name}..")

    versions = client.get_latest_versions(model_name, stages=['None'])
    if not versions:
        print("No new model version found to promote")
        return
    
    new_version = versions[0].version
    print(f"Found model: version {new_version}")

    #promote the latest version to production
    client.set_registered_model_alias(
        name=model_name,
        alias='Production',
        version=new_version
    )

    print(f"Model version {new_version} is now promoted to PRODUCTION")

if __name__== "__main__":
    promote()
