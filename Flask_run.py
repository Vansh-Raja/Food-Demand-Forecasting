from flask import Flask, request, jsonify
import mlflow.pyfunc
import mlflow

app = Flask(__name__)

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

@app.route('/predict',methods=['POST'])
def predict():
    try:
        data = request.get_json()
        name = data['name']
        version = data['version']
        modelInp = data['modelInp']

        model = mlflow.sklearn.load_model(model_uri=f"models:/{name}/{version}")
        prediction = model.predict([modelInp])

        return jsonify(prediction[0])

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
if __name__ == '__main__':
    app.run(debug=True)
    
