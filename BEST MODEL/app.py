from flask import Flask, jsonify, request
import pickle
import pandas as pd

app = Flask(__name__)

@app.route('/recommendation', methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        center_id = data.get('center_id')
        meal_id = data.get('meal_id')
        future_period = int(data.get('future_period'))
        
        if meal_id is None or center_id is None:
            return jsonify({"error":'Meal_id or center_id missing'}),400
        
        file_name = f"{meal_id}_{center_id}_model.pkl"

        try:
            with open(file_name, "rb") as input_file:
                loaded_model = pickle.load(input_file)
                
        except:
            return jsonify({"error":'Meal_id or center_id Wrong'}),400

        last_training_date = loaded_model.history['ds'].max()
        # Create a future DataFrame for the next 5 days with WEEK frequency
        future = pd.date_range(start=last_training_date + pd.DateOffset(1), periods=future_period, freq='W')
        future_df = pd.DataFrame({'ds': future})

        # Make sure 'ds' column is in datetime format
        future_df['ds'] = pd.to_datetime(future_df['ds'])
        preds = loaded_model.predict(future_df)['yhat']
        preds = list(preds)

        return jsonify({"predictions":preds})
        

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)