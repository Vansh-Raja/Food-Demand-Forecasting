# app.py

from flask import Flask, render_template, request, send_file
from preprocess import preprocess_data, get_categorical_columns
import pandas as pd
from sklearn.model_selection import train_test_split
from io import BytesIO

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preprocess', methods=['POST'])
def preprocess_route():
    # Check if the POST request has the file part
    if 'dataset_path' not in request.files:
        return render_template('error.html', message="No file part")

    dataset_file = request.files['dataset_path']

    # Check if the user submitted an empty form
    if dataset_file.filename == '':
        return render_template('error.html', message="No selected file")

    # Save the uploaded file to a temporary location (you might want to handle this more securely)
    dataset_path = "/" + dataset_file.filename
    dataset_file.save(dataset_path)

    # Load the dataset
    dataset = pd.read_csv(dataset_path, nrows=1000)

    missing_option = int(request.form['missing_option'])
    categorical_option = int(request.form['categorical_option'])
    scaling_option = int(request.form['scaling_option'])

    # Execute the pre-processing pipeline
    preprocessed_data = preprocess_data(
        dataset,
        missing_option,
        categorical_option,
        scaling_option
    )

    # Split the dataset into train and test sets
    train_data, test_data = train_test_split(preprocessed_data, test_size=0.2, random_state=42)

    # Save train and test sets to CSV files
    train_data.to_csv('train.csv', index=False)
    test_data.to_csv('test.csv', index=False)
    preprocessed_data.to_csv('complete_dataset.csv', index=False)

    return render_template('result.html', preprocessed_data=preprocessed_data.to_html())

@app.route('/download_csv/<filename>')
def download_csv(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
