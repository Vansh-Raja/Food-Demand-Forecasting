import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5001")
def fetch_model(model_name, stage="Production"):
    import mlflow.pyfunc
    
    model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{stage}")
    return model

def give_modeluri_command(model_name, stage="Production"):
    return f"mlflow models serve --model-uri models:/{model_name}/{stage} -p 1234 --no-conda"
    

def predict(data):
    import requests
    
    inference_request = {
        "dataframe_records": data
    }
    
    endpoint = "http://localhost:1234/invocations"
    
    response = requests.post(endpoint, json=inference_request).json()
    
    return response

print(give_modeluri_command("RandomForestRegressor"))