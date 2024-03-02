from flask import Flask, request, render_template, url_for, redirect, session
import json
import mlflow
from mlflow.tracking import MlflowClient
import subprocess
import pickle
import pandas as pd
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

app = Flask(__name__)
app.secret_key = 'super secret key'

all_ml_options=['Time Series', 'Regression Models', 'LSTM']
all_scaling_options=['Standard Scaling', 'Min-Max Scaling', 'Robust Scaling','Power Transformation','Quantile Transformation','MaxAbsScaler','Normalizer']
all_missing_options=['Mean', 'Median', 'Mode', 'Linear Regression Imputation']
all_encoding_options=['None','One Hot Encoding', 'Label Encoding', 'Helmert Encoding', 'Sum Encoding', 'Backward Difference Encoding', 'James-Stein Encoder']

all_model_options={'Time Series':['FB Prophet'], 'Regression Models':['Gradient Bootsting', 'Decision Tree', 'Random Forest', 'XGBoost', 'Extra Trees Regressor', 'AdaBoost Regressor'], 'LSTM':['LSTM']}
all_tuning_options=['Grid Search', 'Random Search', 'No Tuning']


config={}

@app.route('/')
def first():
    return render_template('preface.html')

@app.route('/home')
def home():
    session.clear()
    return render_template('index.html')

@app.route('/new_model_data',methods=['GET','POST'])
def new_data():
    if request.method=="POST":
        kafka_topic_name = request.form['kafka_topic_name'] if request.form['kafka_topic_name'] else 'bigd'
        ml_type = request.form['ml_type'] if request.form['ml_type'] else 'Time Series'
        kafka_connection_url = request.form['kafka_connection_url'] if request.form['kafka_connection_url'] else 'localhost:9092'
        
        # config['kafka_topic_name']=kafka_topic_name
        # config['ml_type']=ml_type
        # config['kafka_connection_url']=kafka_connection_url
        
        session['path']="New Data"

        session['kafka_topic_name']=kafka_topic_name
        session['ml_type']=ml_type
        session['kafka_connection_url']=kafka_connection_url

        return redirect(url_for('preprocessing'))

    else:
        
        return render_template('kafka.html', ml_options=all_ml_options)


@app.route('/preprocessing',methods=['GET','POST'])
def preprocessing():
    if request.method=="POST":
        missing_option = request.form['missing_option']
        encoding_option = request.form['encoding_option']
        scaling_option = request.form['scaling_option']

        # config['missing_option']=missing_option
        # config['encoding_option']=encoding_option
        # config['scaling_option']=scaling_option

        session['missing_option']=missing_option
        session['encoding_option']=encoding_option
        session['scaling_option']=scaling_option
        
        return redirect(url_for('database'))
    
    if 'raw' in session:
        return render_template('preprocessing.html', scaling_options=all_scaling_options, missing_options=all_missing_options, encoding_options=all_encoding_options, btn='Submit')
    return render_template('preprocessing.html', scaling_options=all_scaling_options, missing_options=all_missing_options, encoding_options=all_encoding_options)

###################
@app.route('/database',methods=['GET','POST'])
def database():
    if request.method=="POST":
        db_type = request.form['selecting_database']
        db_name = request.form['database_name']
        session['db_type']=db_type
        session['db_name']=db_name
        
        if db_type=='mongo':
            mongo_host = request.form['sql_mongo']
            session['mongo_host']=mongo_host
        else:
            sqlite_table_name = request.form['sql_mongo']
            session['sqlite_table_name']=sqlite_table_name

        if 'raw' in session:
            return redirect(url_for('final_selection'))
        
        return redirect(url_for('model'))
    
    return render_template('database.html')
###################

@app.route('/model',methods=['GET','POST'])
def model():
    if request.method=="POST":
        model_option = request.form['model_option']
        tuning_option = request.form['tuning_option']
        session['model_option']=model_option
        session['tuning_option']=tuning_option
        return redirect(url_for('final_selection'))
    
    return render_template('model.html', model_options=all_model_options[session['ml_type']], tuning_options=all_tuning_options)

@app.route('/final_selection',methods=['GET','POST'])
def final_selection():
    if request.method=="POST":
        try:
            center_id = request.form['center_id']
            meal_id = request.form['meal_id']
            print(center_id)
            if center_id!="" and meal_id!="":
                session['data'] = "partial"
                session['center_id']=center_id
                session['meal_id']=meal_id
            else:
                session['data'] = "complete"
                session['center_id']="0"
                session['meal_id']="0"
        except:
            session['data'] = "complete"
            session['center_id']="0"
            session['meal_id']="0"
            pass

        if session['path']=="New Data":
            params={"path": session['path'],
                "kafka_topic":session['kafka_topic_name'],
                "kafka_url":session['kafka_connection_url'],
                "model_type":session['ml_type'],
                "null_values":session['missing_option'],
                "encoding":session['encoding_option'],
                "scaling":session['scaling_option'],
                "model_id":session['model_option'],
                "tuning":session['tuning_option'],
                "db_type":session['db_type'],
                "db_name":session['db_name'],
                "db_url":session['mongo_host'] if session['db_type']=='mongo' else session['sqlite_table_name'],
                "data":session['data'],
                "center_id":session['center_id'],
                "meal_id": session['meal_id']
                }
            with open('params.json','w') as f:
                json.dump(params,f)
            with open('last_config.json','w') as f:
                json.dump(params,f)
            command=['dvc','repro','-f']
            subprocess.run(command, check=True)
            print(session)
        
        elif session['path']=='Existing Raw Data':
            for key, value in all_model_options.items():
                if session['model_option'] in value:
                    session['ml_type']=key
                    break
            params={"path": session['path'],
                "model_type":session['ml_type'],
                "null_values":session['missing_option'],
                "encoding":session['encoding_option'],
                "scaling":session['scaling_option'],
                "model_id":session['model_option'],
                "tuning":session['tuning_option'],
                "db_type":session['db_type'],
                "db_name":session['db_name'],
                "db_url":session['mongo_host'] if session['db_type']=='mongo' else session['sqlite_table_name'],
                "data":session['data'],
                "center_id":session['center_id'],
                "meal_id": session['meal_id']
                    }
            with open('params.json','w') as f:
                    json.dump(params,f)
            command=['dvc','repro','-s','-f','transform']
            subprocess.run(command, check=True)
            command=['dvc','repro','-s','-f','train']
            subprocess.run(command, check=True)
            command=['dvc','repro','-s','-f','load']
            subprocess.run(command, check=True)

        elif session['path']=='Existing Preprocessed Data':
            with open('last_config.json','r') as f:
                di=json.load(f)
                model_type=di['model_type']
                
            
            with open('params.json','r') as f:
                di=json.load(f)
            di['path']=session['path']
            di["center_id"]=session['center_id']
            di["meal_id"]= session['meal_id']
            with open('params.json','w') as f:
                json.dump(di,f)
            command=['dvc', 'repro', '-s','-f','train']
            subprocess.run(command, check=True)
        
                
        return redirect(url_for('result'))
    return render_template('final_selection.html')
    

@app.route('/new_model_existing_data',methods=['GET','POST'])
def new_model_existing_data_preprocessed():
    session.clear()
    session['path']="Existing Preprocessed Data"
    if request.method=="POST":
        #Direct add model part here
        session['model_option']=request.form['model_option']
        session['tuning_option']=request.form['tuning_option']
        return redirect(url_for('final_selection'))
    try:
        with open('last_config.json', 'r') as f:
            config = json.load(f)
            return render_template('model.html', model_options=all_model_options[config['model_type']], tuning_options=all_tuning_options)
    except:
        return redirect(url_for('new_data'))

@app.route('/new_model_existing_data_raw',methods=['GET','POST'])
def new_model_existing_data_raw():
    session.clear()
    session['path']="Existing Raw Data"
    if request.method=="POST":
        session['raw']=True
        model_option = request.form['model_option']
        tuning_option = request.form['tuning_option']
        session['model_option']=model_option
        session['tuning_option']=tuning_option

        return redirect(url_for('preprocessing'))
    
    all_opts = []
    for i in all_model_options:
        all_opts.extend(all_model_options[i])
    return render_template('model.html', model_options=all_opts, tuning_options=all_tuning_options, btn='Next')

@app.route('/existing' ,methods=['GET','POST'])
def existing_model():
    client = MlflowClient()
    rm=client.search_registered_models()
    dic={}
    for model in rm:
        versions=[]
        for ver in client.search_model_versions(f"name='{model.name}'"):
            versions.append(ver.version)
        dic[model.name]=versions
    return render_template('existing.html', models= dic)

@app.route('/best', methods=["GET", "POST"])
def bestModel():
    if request.method=="POST":
        center_id=request.form['center_id']
        meal_id=request.form['meal_id']
        session['center_id']=center_id
        session['meal_id']=meal_id
        future_period=5
        file_name = f"BestTS/{meal_id}_{center_id}_model.pkl"
        try:
            with open(file_name, "rb") as input_file:
                loaded_model = pickle.load(input_file)
        except Exception as e:
            print(e)
            return redirect(url_for('home'))
        last_training_date = loaded_model.history['ds'].max()
        # Create a future DataFrame for the next 5 days with WEEK frequency
        future = pd.date_range(start=last_training_date + pd.DateOffset(1), periods=future_period, freq='W')
        future_df = pd.DataFrame({'ds': future})

        # Make sure 'ds' column is in datetime format
        future_df['ds'] = pd.to_datetime(future_df['ds'])
        preds = loaded_model.predict(future_df)['yhat']
        preds = list(preds)
        preds = [round(x, 2) for x in preds]
        return render_template('result.html', preds=preds)
    return redirect(url_for('home'))


@app.route('/result')
def result():
    with open(r"D:\Study\INTERNSHIP\FINALDuplicate\data\result\result.txt",'r') as f:
        preds=f.readlines()

    preds = [round(float(x), 2) for x in preds]
    return render_template('result.html',preds=preds)
if __name__ == '__main__':
    app.run(debug=True)