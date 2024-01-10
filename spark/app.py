from flask import Flask, render_template, request, send_file
from pyspark.sql import SparkSession
from preprocess_spark import preprocess_and_standardize

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preprocess', methods=['POST'])
def preprocess():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        # Save the uploaded file
        dataset_path = 'uploaded_dataset.csv'
        uploaded_file.save(dataset_path)

        # Read the dataset using Spark
        spark = SparkSession.builder.appName('Dataframe').getOrCreate()
        df_pyspark = spark.read.csv(dataset_path, header=True, inferSchema=True)

        # Preprocess the dataset
        final_standardized_df = preprocess_and_standardize(df_pyspark)

        # Save the preprocessed dataset
        preprocessed_path = 'preprocessed_dataset.csv'
        final_standardized_df.write.csv(preprocessed_path, header=True, mode='overwrite')

        return render_template('result.html', dataset_path=preprocessed_path)
    else:
        return render_template('index.html', error='No file selected.')

@app.route('/download/<filename>')
def download(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
